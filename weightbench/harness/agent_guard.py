"""Runtime guard loaded into every Python process the agent starts (installed as ``sitecustomize.py``).

Standard library only at import time; torch / numpy are patched lazily when the agent imports them.
Configuration comes from environment variables set by the runner:

    WB_TRACE_DIR    directory for per-process trace summaries (JSON, read by the audit)
    WB_SHAPES       JSON file with the 2-D weight shapes of the task's public artifacts
    WB_FLOP_CAP     optional per-process cap on metered matmul FLOPs (torch and numpy functions); a
                    ``flop_cap.json`` next to this file (the task's cap) takes precedence, the variable can only lower it

What it enforces (SPEC §1 *Enforcement*):
1. Network: sockets may only connect to localhost (the Weight API and the egress proxy).
2. No model-execution or simulator libraries (``BLOCKED_MODULES``).
3. No gradient descent: ``Tensor.backward``, ``torch.autograd.grad``, ``torch.func`` / functorch transforms and
   every ``Optimizer.step`` raise. Blocked events are also POSTed to the Weight API (``WB_URL``) as they happen,
   so they land in the append-only ledger and cannot be erased by deleting trace files.
4. Trace: matmul FLOPs, weight-shaped matmuls and forward-pass-like chains (weight matmul → nonlinearity
   → weight matmul) are counted and written to ``WB_TRACE_DIR`` at exit.
Python-level guards can be bypassed by a determined agent; the OS sandbox (network) and the audit
(which also flags bypass attempts) are the other layers.
"""
from __future__ import annotations

import atexit
import builtins
import importlib.abc
import json
import os
import socket
import sys

BLOCKED_MODULES = {
    "transformers", "vllm", "llama_cpp", "accelerate", "peft", "optimum", "onnxruntime", "tensorflow", "keras",
    "jax", "flax", "minatar", "gymnasium", "gym", "kinetix", "sklearn", "lightgbm", "xgboost", "requests",
    "httpx", "urllib3", "aiohttp",
}
NONLINEAR = {"relu", "relu_", "gelu", "silu", "sigmoid", "tanh", "softmax", "log_softmax", "leaky_relu",
             "elu", "softplus", "maximum", "clamp", "clamp_min", "exp", "where", "threshold", "relu6", "hardtanh"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class GuardViolation(PermissionError):
    pass


class BlockedModule(ModuleNotFoundError):
    """Raised for blocked imports. An ImportError subclass, so optional ``try: import x`` keeps working."""


TRACE = {"pid": os.getpid(), "argv": sys.argv[:4], "flops": 0, "weight_matmuls": 0, "forward_chains": 0,
         "blocked": [], "metering_escapes": 0, "unsupported_contractions": 0,
         "raw_weight_opens": 0, "raw_weight_data_bytes": 0, "weight_export_bytes": 0}
_STATE = {"chain": 0, "after_nonlinear": False}


_REPORTING = [False]


def _report(final: bool) -> None:
    """Send this process's trace to the Weight API (WB_URL), which appends it to the run's ledger. Events that
    reach the append-only ledger cannot be removed later, unlike the per-process trace files."""
    url, token = os.environ.get("WB_URL"), os.environ.get("WB_TOKEN", "")
    if not url or _REPORTING[0]:
        return
    _REPORTING[0] = True
    try:
        import http.client
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=3)
        conn.request("POST", "/trace", json.dumps({**TRACE, "final": final}),
                     {"Content-Type": "application/json", "X-WB-Token": token})
        conn.getresponse().read()
        conn.close()
    except Exception:  # noqa: BLE001 - reporting must never break the agent's process
        pass
    finally:
        _REPORTING[0] = False


def _record_block(what: str) -> None:
    if len(TRACE["blocked"]) < 50:
        TRACE["blocked"].append(what)
    _report(final=False)


# ---- 1. network -------------------------------------------------------------------------------
def _host_of(address) -> str:
    return str(address[0]) if isinstance(address, tuple) and address else str(address)


_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _checked(address) -> None:
    host = _host_of(address)
    if hasattr(socket, "AF_UNIX") and isinstance(address, (str, bytes)):
        return   # unix-domain sockets are local by definition
    if host not in LOCAL_HOSTS:
        _record_block(f"socket:{host}")
        raise GuardViolation(f"WeightBench sandbox: no network access ({host}); only the local Weight API is reachable")


def _connect(self, address):
    _checked(address)
    return _orig_connect(self, address)


def _connect_ex(self, address):
    _checked(address)
    return _orig_connect_ex(self, address)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex

_orig_create_connection = socket.create_connection


def _create_connection(address, *args, **kwargs):
    """``create_connection`` resolves the host name before ``connect``. Without network (the OS sandbox) the
    resolution fails first, so the attempt would never reach ``connect`` and leave no trace. Check the host up front."""
    _checked(address)
    return _orig_create_connection(address, *args, **kwargs)


socket.create_connection = _create_connection


# ---- 2. blocked modules + 3./4. lazy torch / numpy patches ------------------------------------
class _BlockedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        root = module.__name__.split(".")[0]
        _record_block(f"import:{root}")
        raise BlockedModule(f"WeightBench sandbox: '{root}' is not available in this benchmark", name=root)


class _Guard(importlib.abc.MetaPathFinder):
    """Blocked packages that are installed resolve to a spec whose loader raises on import, so
    availability checks (``importlib.util.find_spec``) don't crash libraries, but importing fails.
    Packages that aren't installed fall through to the normal 'not found'."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in BLOCKED_MODULES:
            return None
        import importlib.machinery

        real = importlib.machinery.PathFinder.find_spec(fullname, path)
        if real is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _BlockedLoader(), origin=real.origin)



_orig_import = builtins.__import__
_IMPORT_DEPTH = [0]
_PATCHED: set[str] = set()     # kept here, not as module attributes (torch's __getattr__ may import)
_PATCHING = [False]


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    """Patch torch / numpy once the *outermost* import has finished (modules fully initialised)."""
    _IMPORT_DEPTH[0] += 1
    try:
        module = _orig_import(name, globals, locals, fromlist, level)
    finally:
        _IMPORT_DEPTH[0] -= 1
    if _IMPORT_DEPTH[0] == 0 and not _PATCHING[0]:
        _PATCHING[0] = True
        try:
            for mod_name, patch in (("torch", _patch_torch), ("numpy", _patch_numpy),
                                    ("safetensors", _patch_safetensors),
                                    ("safetensors.numpy", _patch_safetensors)):
                if mod_name in sys.modules and mod_name not in _PATCHED and (mod_name != "safetensors.numpy"
                                                                             or "numpy" in _PATCHED):
                    _PATCHED.add(mod_name)
                    patch(sys.modules[mod_name])
            if {"torch", "numpy"} <= _PATCHED and "torch->numpy" not in _PATCHED:
                _PATCHED.add("torch->numpy")
                _patch_torch_numpy_bridge(sys.modules["torch"])
        finally:
            _PATCHING[0] = False
    return module


def _load_shapes() -> set[tuple[int, int]]:
    try:
        with open(os.environ["WB_SHAPES"]) as f:
            shapes = {tuple(s) for s in json.load(f) if len(s) == 2}
    except (KeyError, OSError, ValueError):
        return set()
    return shapes | {(b, a) for a, b in shapes}


def _load_weight_files() -> dict[str, tuple]:
    """Resolved path -> tensor-data byte ranges for every subject weight data file (``weight_files.json`` next to this
    module, written by ``sandbox.install_guard``), at every location the agent can see it (artifact dir and agent
    workdir). Also fills ``WEIGHT_BY_SIZE`` (size -> [(fingerprint, ranges)]) so an exact copy under another name is
    recognised. Legacy files with only ``data_offset`` become one range from that offset to the end."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_files.json")) as f:
            entries = json.load(f)
    except (OSError, ValueError, NameError):
        return {}
    out = {}
    for item in entries if isinstance(entries, list) else []:
        try:
            if "ranges" in item:
                ranges = tuple((int(a), int(b)) for a, b in item["ranges"] if int(b) > int(a))
            else:
                ranges = ((int(item.get("data_offset", 0)), 1 << 62),)
            for path in [item["path"], *(item.get("paths") or [])]:
                out[os.path.realpath(path)] = ranges
            if item.get("fingerprint") and item.get("size"):
                WEIGHT_BY_SIZE.setdefault(int(item["size"]), []).append((str(item["fingerprint"]), ranges))
        except (TypeError, KeyError, ValueError):
            continue
    return out


WEIGHT_BY_SIZE: dict[int, list] = {}
WEIGHT_FILES = _load_weight_files()
_WEIGHT_INODES: dict[tuple[int, int], tuple] = {}   # (st_dev, st_ino) of the declared files -> ranges (hard links)
for _p, _r in WEIGHT_FILES.items():
    try:
        _st = os.stat(_p)
        _WEIGHT_INODES[(_st.st_dev, _st.st_ino)] = _r
    except OSError:
        pass
_GUARD_LOADING = [0]              # >0 while the guard's own metered loaders read a weight file (re-entrancy)


class _guard_loading:
    """Marks the guard's own metered file reads, so the raw-open hook does not flag them."""

    def __enter__(self):
        _GUARD_LOADING[0] += 1
        return self

    def __exit__(self, *exc):
        _GUARD_LOADING[0] -= 1
        return False


WEIGHT_SHAPES = _load_shapes()


def _load_flop_cap() -> float | None:
    """The run's cap: ``flop_cap.json`` next to this file (written by ``sandbox.install_guard``, read-only for the
    agent) wins; ``WB_FLOP_CAP`` can only lower it. Without the file (tests, legacy guards) the variable alone sets
    it. An agent that exports a larger ``WB_FLOP_CAP`` in its shell therefore cannot raise a task's cap."""
    caps = []
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flop_cap.json")) as f:
            value = json.load(f)
        if value is not None:
            caps.append(float(value))
    except (OSError, ValueError, TypeError, NameError):
        pass
    try:
        if os.environ.get("WB_FLOP_CAP"):
            caps.append(float(os.environ["WB_FLOP_CAP"]))
    except ValueError:
        pass
    return min(caps) if caps else None


FLOP_CAP = _load_flop_cap()


def _charge_flops(flops: int) -> None:
    TRACE["flops"] += flops
    if FLOP_CAP is not None and TRACE["flops"] > FLOP_CAP:
        _record_block("flop_cap")
        raise GuardViolation(f"WeightBench sandbox: per-process FLOP cap ({FLOP_CAP:.3g}) exceeded")


def _note_weight_product(shape_a, shape_b, operands=None) -> str | None:
    """Forward-chain heuristic shared by matmul and supported contractions.

    A product is a *layer* when exactly one operand has a weight shape (weight applied to activations). Its result is
    tagged ``act``. A chain is a layer whose activation operand went through a nonlinearity after an earlier layer.

    NumPy passes the operands, and the decision follows the data (tags on ``WeightArray``, see ``_TAG_RANK``):
    - the activation operand is tracked: chain iff it is tagged ``act_nl``;
    - the weight operand is a sign-split / |W| / clipped / masked transform of a weight (tag ``wx``): this is bound
      propagation (``max(W,0) @ u + min(W,0) @ l``), not a layer; the result is tagged ``bound`` and never chains;
    - the activation operand is a plain (untracked) array: the order-based rule below.
    Torch passes no operands and keeps the order-based rule: a nonlinearity seen after a layer, then another layer.
    Returns ``None`` (not a weight-on-activation product), ``"act"`` or ``"bound"``.

    Provenance wins over shape: an operand tagged ``act`` / ``act_nl`` is an activation whatever its shape (a batch of 1
    can make an activation's shape equal a weight's). Only when neither operand's provenance identifies it (torch, or a
    plain untracked activation) does the shape heuristic ("exactly one operand is weight-shaped") decide."""
    a2 = tuple(int(d) for d in shape_a[-2:]) if len(shape_a) >= 2 else None
    b2 = tuple(int(d) for d in shape_b[-2:]) if len(shape_b) >= 2 else None
    oa = operands[0] if operands is not None and len(operands) >= 2 else None
    ob = operands[1] if operands is not None and len(operands) >= 2 else None
    ta, tb = _tag(oa), _tag(ob)
    act_tags = ("act", "act_nl")
    # Provenance: identify the activation operand by its tag, regardless of shape.
    if ta in act_tags and tb not in act_tags:
        weight, activation = ob, oa
    elif tb in act_tags and ta not in act_tags:
        weight, activation = oa, ob
    else:
        # No activation provenance: fall back to the shape heuristic, used by torch and plain operands.
        weight_on_activation = (b2 in WEIGHT_SHAPES and a2 not in WEIGHT_SHAPES) or \
                               (a2 in WEIGHT_SHAPES and b2 not in WEIGHT_SHAPES)
        if not weight_on_activation:
            return None                     # two genuine weights (W2 @ W1), or neither weight-shaped: static analysis
        weight, activation = ((oa, ob) if a2 in WEIGHT_SHAPES else (ob, oa)) if operands is not None else (None, None)
    TRACE["weight_matmuls"] += 1
    if _tag(weight) == "wx":
        return "bound"                      # sign-split / |W| / masked weight: bound propagation, never a layer
    if _tracked(activation):
        chain = _tag(activation) == "act_nl"
    else:
        chain = _STATE["after_nonlinear"] and _STATE["chain"] >= 1
    if chain:
        TRACE["forward_chains"] += 1
    _STATE["chain"] += 1
    _STATE["after_nonlinear"] = False
    return "act"


def _note_matmul(shape_a, shape_b, operands=None) -> str | None:
    """Count FLOPs; detect a weight matrix applied to activations, and chains of them."""
    if len(shape_a) < 1 or len(shape_b) < 1:
        return None
    m = 1
    for d in shape_a[:-1]:
        m *= int(d)
    k = int(shape_a[-1])
    n = int(shape_b[-1]) if len(shape_b) >= 2 else 1
    _charge_flops(2 * m * k * n)
    return _note_weight_product(shape_a, shape_b, operands)


# ---- dataflow tags on metered NumPy arrays (forward chain vs. bound propagation) ---------------------------------
# None: weight or weight-derived value; "wx": a weight transformed by a value-changing nonlinearity (max/min/abs/clip/
# where/boolean mask: a sign split for bounds); "act": output of a weight-on-activation product (or derived linearly);
# "act_nl": an activation that went through a nonlinearity; "bound": anything computed from a "wx" product (interval /
# back-substitution bounds). Mixing tags keeps the highest rank, so a value mixed with a bound is a bound.
_TAG_RANK = {None: 0, "wx": 1, "act": 2, "act_nl": 3, "bound": 4}
NUMPY_TAG_NONLINEAR = {"maximum", "minimum", "fmax", "fmin", "clip", "tanh", "exp", "expm1", "log1p", "logaddexp",
                       "absolute", "fabs", "greater", "greater_equal", "less", "less_equal", "heaviside"}


def _tracked(value) -> bool:
    weight_cls = _NP.get("WeightArray")
    return weight_cls is not None and isinstance(value, weight_cls)


def _tag(value):
    return getattr(value, "_wb_tag", None) if _tracked(value) else None


def _flat(values):
    out = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.extend(v)
        elif isinstance(v, dict):
            out.extend(v.values())
        else:
            out.append(v)
    return out


def _combine(values):
    best = None
    for v in _flat(values):
        t = _tag(v)
        if _TAG_RANK.get(t, 0) > _TAG_RANK.get(best, 0):
            best = t
    return best


def _weight_shaped(value) -> bool:
    shape = getattr(value, "shape", ())
    return len(shape) >= 2 and tuple(int(d) for d in shape[-2:]) in WEIGHT_SHAPES


def _nonlinear_tag(inputs, result):
    """Tag of a nonlinearity's result: on an activation (more than one element) it is ``act_nl``; on a weight it is a
    ``wx`` transform when it changed the weight's values (``max(W, -1e30)`` leaves W a plain weight); bounds stay
    bounds."""
    np = _NP["np"]
    values = _flat(inputs)
    tags = [_tag(v) for v in values]
    if "bound" in tags:
        return "bound"
    acts = [v for v, t in zip(values, tags) if t in ("act", "act_nl")]
    if any(int(np.size(v)) > 1 for v in acts):
        return "act_nl"
    if acts:
        return "act"
    for v, t in zip(values, tags):
        if _tracked(v) and t in (None, "wx") and _weight_shaped(v):
            if t == "wx":
                return "wx"
            try:
                same = np.shape(result) == np.shape(v) and bool(np.array_equal(_untaint(result), _untaint(v)))
            except Exception:  # noqa: BLE001 - incomparable results (object dtype, ...) count as changed
                same = False
            if not same:
                return "wx"
    return _combine(values)


def _reduced(tag, inputs, result):
    """Reductions (sums, norms, maxima) of an activation are not elementwise activations themselves."""
    np = _NP["np"]
    if tag != "act_nl":
        return tag
    try:
        largest = max((int(np.size(v)) for v in _flat(inputs) if _tracked(v)), default=0)
        if int(np.size(result)) <= 1 or int(np.size(result)) < largest:
            return "act"
    except Exception:  # noqa: BLE001
        pass
    return tag


def _set_tag(value, tag) -> None:
    if _tracked(value):
        try:
            value._wb_tag = tag
        except AttributeError:
            pass
    elif isinstance(value, (list, tuple)):
        for v in value:
            _set_tag(v, tag)


# ---- copies of weights out of the metered type ------------------------------------------------------------------
# ``w.tobytes()`` / ``w.tolist()`` return plain Python objects. Their ids (with length and a prefix, to reject reuse)
# are remembered briefly, so ``np.frombuffer`` / ``np.array`` / ``np.asarray`` on them return metered arrays again.
_WEIGHT_OBJECTS: dict[int, tuple[int, object]] = {}
_WEIGHT_OBJECTS_MAX = 64


def _remember(obj) -> None:
    try:
        key = (len(obj), obj[:16] if isinstance(obj, (bytes, bytearray)) else type(obj[0]) if len(obj) else None)
    except Exception:  # noqa: BLE001
        return
    _WEIGHT_OBJECTS[id(obj)] = key
    while len(_WEIGHT_OBJECTS) > _WEIGHT_OBJECTS_MAX:
        _WEIGHT_OBJECTS.pop(next(iter(_WEIGHT_OBJECTS)))


def _from_weight(value) -> bool:
    """True for a metered array, a memoryview of one, or a remembered ``tobytes`` / ``tolist`` result."""
    if _tracked(value):
        return True
    if isinstance(value, memoryview):
        return _from_weight(value.obj)
    key = _WEIGHT_OBJECTS.get(id(value))
    if key is None or not isinstance(value, (bytes, bytearray, list)):
        return False
    try:
        return key == (len(value), value[:16] if isinstance(value, (bytes, bytearray))
                       else type(value[0]) if len(value) else None)
    except Exception:  # noqa: BLE001
        return False


def _note_copyto(args, kwargs) -> None:
    """``np.copyto(dst, w)`` fills a plain array with weight values; NumPy cannot change ``dst``'s type, and nothing
    computed from ``dst`` is metered afterwards. Recorded as a metering escape only when a whole weight matrix (its
    trailing 2-D shape is a public weight shape) lands in a plain array. Copies into metered arrays
    (``np.empty_like(w)``, ``np.zeros_like(w)``), of slices, vectors and scalars are not recorded."""
    np = _NP["np"]
    dst = args[0] if args else kwargs.get("dst")
    src = args[1] if len(args) > 1 else kwargs.get("src")
    if (isinstance(dst, np.ndarray) and not _tracked(dst) and _tracked(src) and _weight_shaped(src)
            and _weight_shaped(dst) and tuple(dst.shape[-2:]) == tuple(src.shape[-2:])):
        TRACE["metering_escapes"] = TRACE.get("metering_escapes", 0) + 1
        _report(final=False)        # reach the ledger at once (the trace file could be deleted later)


# ---- raw weight-file access at the file boundary -----------------------------------------------------------------
# Weight files sit in the agent's workdir, so `open(path, "rb")` + np.frombuffer / struct bypasses the metered
# loaders. A guarded process that opens a declared subject weight file outside the guard's own loaders is recorded;
# reads that stay within the format header (tensor names / shapes: legitimate static analysis) are not flagged, only
# reads that reach the tensor data. The guard's loaders raise `_GUARD_LOADING`, so np.load / safetensors stay clean.
_FINGERPRINT_CHUNK = 65536       # same as sandbox.FINGERPRINT_CHUNK


def _fingerprint(path, size: int, opener) -> str:
    """Same value as ``sandbox.file_fingerprint``: sha256 of the size and the first / last 64 KiB."""
    import hashlib

    h = hashlib.sha256(str(size).encode() + b"\0")
    with opener(path, "rb") as f:
        h.update(f.read(_FINGERPRINT_CHUNK))
        if size > _FINGERPRINT_CHUNK:
            f.seek(max(_FINGERPRINT_CHUNK, size - _FINGERPRINT_CHUNK))
            h.update(f.read(_FINGERPRINT_CHUNK))
    return h.hexdigest()


def _weight_ranges(path, handle=None, opener=None):
    """Tensor-data ranges of the file at ``path`` if it is a subject weight file, else None. Matched by resolved path
    (every sandbox-visible location of the public artifacts), then by file identity (hard link / bind mount), then by
    size + fingerprint (an exact copy the agent made, e.g. ``cp mlp.npz x.bin``). Only a file whose size equals a
    declared weight file's is fingerprinted, so other opens cost one fstat."""
    ranges = WEIGHT_FILES.get(os.path.realpath(path))
    if ranges is not None:
        return ranges
    try:
        st = os.fstat(handle.fileno()) if handle is not None else os.stat(path)
    except (OSError, AttributeError, ValueError):
        return None
    ranges = _WEIGHT_INODES.get((st.st_dev, st.st_ino))
    if ranges is not None or st.st_size not in WEIGHT_BY_SIZE or opener is None:
        return ranges
    try:
        with _guard_loading():
            fp = _fingerprint(path, st.st_size, opener)
    except OSError:
        return None
    for known, candidate in WEIGHT_BY_SIZE[st.st_size]:
        if known == fp:
            return candidate
    return None


def _weight_data_offset(path: str):
    """First tensor-data byte of a declared weight file (path match only), or None. Kept for existing callers."""
    ranges = WEIGHT_FILES.get(os.path.realpath(path))
    return None if ranges is None else (ranges[0][0] if ranges else 0)


def _note_raw_weight_data(nbytes: int) -> None:
    if nbytes <= 0:
        return
    TRACE["raw_weight_data_bytes"] = TRACE.get("raw_weight_data_bytes", 0) + int(nbytes)
    _report(final=False)


def _overlap(start: int, end: int, ranges) -> int:
    return sum(max(0, min(end, b) - max(start, a)) for a, b in ranges)


class _RawWeightReader:
    """Wraps a binary handle on a subject weight file opened outside the guard. Delegates everything, and records the
    bytes of each read that fall inside a tensor-data range (zip central directory, local headers and npy /
    safetensors headers stay clean). Every reading method is wrapped (read, read1, readinto, readinto1, readline,
    readlines, readall, iteration), each by the file position before and after the call."""

    def __init__(self, handle, ranges):
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_ranges", tuple(ranges))

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __setattr__(self, name, value):
        setattr(self._handle, name, value)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def _tell(self):
        try:
            return self._handle.tell()
        except Exception:  # noqa: BLE001 - unseekable handle: cannot locate the read
            return None

    def _call(self, name, *a, **k):
        before = self._tell()
        result = getattr(self._handle, name)(*a, **k)
        try:
            after = self._tell()
            if before is not None and after is not None and after > before:
                _note_raw_weight_data(_overlap(before, after, self._ranges))
        except Exception:  # noqa: BLE001
            pass
        return result

    def read(self, *a, **k):
        return self._call("read", *a, **k)

    def read1(self, *a, **k):
        return self._call("read1", *a, **k)

    def readinto(self, *a, **k):
        return self._call("readinto", *a, **k)

    def readinto1(self, *a, **k):
        return self._call("readinto1", *a, **k)

    def readline(self, *a, **k):
        return self._call("readline", *a, **k)

    def readlines(self, *a, **k):
        return self._call("readlines", *a, **k)

    def readall(self, *a, **k):
        return self._call("readall", *a, **k)

    def __iter__(self):
        return self

    def __next__(self):
        line = self._call("readline")
        if not line:
            raise StopIteration
        return line


def _install_raw_open_guard() -> None:
    """Wrap builtins.open / io.open (pathlib's ``read_bytes`` and ``zipfile`` use io.open) and add an audit hook for
    os.open, so raw reads of a subject weight file by this guarded process are recorded. The guard's own loaders set
    ``_GUARD_LOADING`` and are never counted."""
    if not WEIGHT_FILES:
        return
    import io

    original_open = builtins.open

    def open_wrapper(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        try:
            if _GUARD_LOADING[0] or isinstance(file, int) or "b" not in mode \
                    or any(c in mode for c in "wax+"):
                return handle
            ranges = _weight_ranges(os.fspath(file), handle, original_open)
            if ranges is None:
                return handle
            TRACE["raw_weight_opens"] = TRACE.get("raw_weight_opens", 0) + 1
            _report(final=False)
            return _RawWeightReader(handle, ranges)
        except Exception:  # noqa: BLE001 - never break a legitimate open
            return handle

    builtins.open = open_wrapper
    if io.open is original_open:
        io.open = open_wrapper

    def audit_hook(event, args):
        # os.open does not go through builtins.open; record the raw open (bytes cannot be counted from a bare fd, so
        # it is recorded but does not by itself flag, to avoid flagging a header peek through a low-level fd).
        if event != "os.open" or _GUARD_LOADING[0] or not args:
            return
        try:
            if isinstance(args[0], (str, bytes, os.PathLike)) and _weight_ranges(os.fspath(args[0])) is not None:
                TRACE["raw_weight_opens"] = TRACE.get("raw_weight_opens", 0) + 1
        except Exception:  # noqa: BLE001
            pass

    try:
        sys.addaudithook(audit_hook)
    except Exception:  # noqa: BLE001 - audit hooks may be unavailable; the open wrapper still applies
        pass


def _einsum_plan(equation, shapes):
    """Static FLOP proxy for string unary/binary contractions; None means unsupported.

    Binary reduction uses the usual 2*product(index extents) multiply/add proxy. Outer or
    elementwise products use one multiply per output. Unary reduction counts additions;
    unary views/permutations/diagonals cost zero. No subject values are evaluated here.
    """
    if not isinstance(equation, str) or len(shapes) not in (1, 2):
        return None
    equation = "".join(equation.split())
    if equation.count("->") > 1:
        return None
    parts = equation.split("->")
    terms = parts[0].split(",")
    if len(terms) != len(shapes):
        return None
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def tokens(term):
        if term.count("...") > 1 or any(c not in letters for c in term.replace("...", "")):
            return None
        return list(term.replace("...", "@"))
    raw = [tokens(term) for term in terms]
    if any(labels is None for labels in raw):
        return None
    counts = [len(shape) - len([x for x in labels if x != "@"])
              for labels, shape in zip(raw, shapes)]
    if any(n < 0 or ("@" not in labels and n != 0) for labels, n in zip(raw, counts)):
        return None
    width = max(counts, default=0)
    ellipsis = [f"@{i}" for i in range(width)]
    expanded = []
    dims = {}
    for labels, shape, count in zip(raw, shapes, counts):
        axes = [axis for label in labels for axis in (ellipsis[width-count:] if label == "@" else [label])]
        local = {}
        for axis, dimension in zip(axes, shape):
            dimension = int(dimension)
            if axis in local and local[axis] != dimension:
                return None                 # diagonal axes must agree
            local[axis] = dimension
            previous = dims.get(axis, dimension)
            if previous != dimension and previous != 1 and dimension != 1:
                return None
            dims[axis] = dimension if previous == 1 else previous
        expanded.append(axes)
    if len(parts) == 2:
        output = tokens(parts[1])
        if output is None:
            return None
        output = [axis for label in output for axis in (ellipsis if label == "@" else [label])]
    else:
        flat = [axis for axes in expanded for axis in axes]
        output = ellipsis + sorted(axis for axis in set(flat) if not axis.startswith("@") and flat.count(axis) == 1)
    if len(set(output)) != len(output) or any(axis not in dims for axis in output):
        return None
    if any(axis not in output for axis in ellipsis):
        return None                         # ellipsis reduction differs between NumPy and Torch
    def product(axes):
        size = 1
        for axis in axes:
            size *= dims[axis]
        return size
    if len(shapes) == 1:
        return max(0, product(set(expanded[0])) - product(output)), False
    left, right = map(set, expanded)
    if any(len(set(axes)) != len(axes) for axes in expanded):
        return None                         # binary diagonal contractions need separate support
    reduction = (left | right) - set(output)
    if not reduction <= (left & right):
        return None                         # no unpaired operand reductions in this supported subset
    return product(left | right) * (2 if reduction else 1), bool(reduction)


def _note_einsum(args, library: str) -> str | None:
    TRACE["einsum_calls"] = TRACE.get("einsum_calls", 0) + 1
    operands = args[1:] if args and isinstance(args[0], str) else ()
    if library == "torch" and len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    shapes = [tuple(int(d) for d in value.shape) for value in operands if hasattr(value, "shape")]
    plan = _einsum_plan(args[0], shapes) if args and len(shapes) == len(operands) else None
    if plan is None:
        TRACE["unsupported_contractions"] += 1
        _report(final=False)                 # preserve the signal even if the agent deletes its trace
        return None
    flops, weight_product = plan
    _charge_flops(flops)
    if weight_product:
        return _note_weight_product(shapes[0], shapes[1], tuple(operands[:2]) if library == "numpy" else None)
    return None


def _note_nonlinear() -> None:
    """Order-based rule (torch, and NumPy nonlinearities on plain untracked arrays)."""
    if _STATE["chain"]:
        _STATE["after_nonlinear"] = True


def _blocked(what: str):
    def raiser(*args, **kwargs):
        _record_block(what)
        raise GuardViolation(f"WeightBench sandbox: {what} is disabled (no gradient descent)")
    return raiser


def _patch_torch(torch) -> None:
    torch.Tensor.backward = _blocked("Tensor.backward")
    torch.autograd.backward = _blocked("autograd.backward")
    torch.autograd.grad = _blocked("autograd.grad")
    torch.optim.Optimizer.__init__ = _blocked("torch.optim optimizers")   # covers every subclass (SGD, Adam, ...)
    torch.optim.Optimizer.step = _blocked("Optimizer.step")
    for module_name in ("torch.func", "torch._functorch.eager_transforms", "functorch"):
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = __import__(module_name, fromlist=["grad"])
            except Exception:  # noqa: BLE001 - not every torch build ships every alias
                continue
        for transform in ("grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd", "hessian", "linearize"):
            if hasattr(module, transform):
                setattr(module, transform, _blocked(f"{module_name}.{transform}"))
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:
        return
    matmuls = {"matmul", "mm", "bmm", "linear", "einsum", "addmm", "__matmul__", "__rmatmul__"}

    class _Meter(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", "")
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            if name == "einsum":
                _note_einsum(args, "torch")
            elif name in matmuls and len(tensors) >= 2:
                if name == "linear":
                    _note_matmul(tuple(tensors[0].shape), tuple(tensors[1].shape)[::-1])
                elif name == "addmm" and len(tensors) >= 3:
                    _note_matmul(tuple(tensors[1].shape), tuple(tensors[2].shape))
                else:
                    _note_matmul(tuple(tensors[0].shape), tuple(tensors[1].shape))
            elif name in NONLINEAR:
                _note_nonlinear()
            return func(*args, **(kwargs or {}))

    _Meter().__enter__()


NUMPY_PRODUCTS = {"matmul", "dot", "einsum", "tensordot", "inner", "vdot", "linalg.multi_dot", "multi_dot"}
_NP: dict = {}


def _untaint(value):
    """Strip the WeightArray subclass from arguments (recursively through lists / tuples / dicts)."""
    weight_cls = _NP.get("WeightArray")
    if weight_cls is not None and isinstance(value, weight_cls):
        return _NP["np"].ndarray.view(value, _NP["np"].ndarray)     # base-class view: not an agent escape
    if isinstance(value, (list, tuple)):
        return type(value)(_untaint(v) for v in value)
    if isinstance(value, dict):
        return {k: _untaint(v) for k, v in value.items()}
    return value


def _taint(value):
    """Results computed from weights stay metered (views share memory; nothing is copied)."""
    np = _NP["np"]
    if isinstance(value, np.ndarray) and not isinstance(value, _NP["WeightArray"]):
        return np.ndarray.view(value, _NP["WeightArray"])
    if isinstance(value, tuple):
        return tuple(_taint(v) for v in value)
    if isinstance(value, list):
        return [_taint(v) for v in value]
    return value


def _first_two_shapes(args) -> tuple | None:
    np = _NP["np"]
    arrays = [a for a in args if isinstance(a, np.ndarray)]
    if len(arrays) >= 2:
        return arrays[0].shape, arrays[1].shape
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) >= 2:   # multi_dot([A, B, ...])
        return np.shape(args[0][0]), np.shape(args[0][1])
    return None


def _note_weight_export(array) -> None:
    """Record the byte size of a buffer / list export of a metered array, so the audit can see large exports of
    weight-sized data (``w.tobytes()`` over a whole matrix). Informational: these stay metered when fed back through
    numpy, so the size is recorded, not quarantined (it is the backstop CPU meter that catches arithmetic on a copy
    that leaves numpy)."""
    try:
        nbytes = int(getattr(array, "nbytes", 0))
    except Exception:  # noqa: BLE001
        return
    if nbytes > 0:
        TRACE["weight_export_bytes"] = TRACE.get("weight_export_bytes", 0) + nbytes


def _make_weight_array(np):
    """ndarray subclass for everything loaded from the agent's weight files. NumPy routes ``@`` and every ufunc
    through ``__array_ufunc__`` and functions such as ``dot`` / ``einsum`` through ``__array_function__``, so
    C-level operators on weights (and on anything computed from them) are metered without a copy."""

    class WeightArray(np.ndarray):
        _wb_tag = None

        def __array_finalize__(self, obj):
            """Views, slices, reshapes and ``astype`` copies keep the dataflow tag of their source."""
            self._wb_tag = getattr(obj, "_wb_tag", None)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            name = ufunc.__name__
            kind = None
            if method == "__call__" and name == "matmul" and len(inputs) >= 2:
                kind = _note_matmul(np.shape(inputs[0]), np.shape(inputs[1]), inputs[:2])
            outs = kwargs.get("out")
            if "out" in kwargs:
                kwargs["out"] = _untaint(kwargs["out"])
            result = _taint(getattr(ufunc, method)(*_untaint(inputs), **kwargs))
            if kind is not None:
                tag = kind
            elif method != "__call__":
                tag = _reduced(_combine(inputs), inputs, result)       # reduce / accumulate / outer / at
            elif name in NUMPY_TAG_NONLINEAR or (name == "multiply" and any(
                    getattr(getattr(v, "dtype", None), "kind", "") == "b" for v in inputs)):
                tag = _nonlinear_tag(inputs, result)   # h * (h > 0) is a ReLU written as a boolean mask
            else:
                tag = _combine(inputs)
            _set_tag(result, tag)
            _set_tag(outs, tag)
            return result

        def __array_function__(self, func, types, args, kwargs):
            name = getattr(func, "__name__", "")
            kind = None
            if name in NUMPY_PRODUCTS:
                shapes = _first_two_shapes(args)
                if name == "einsum":
                    kind = _note_einsum(args, "numpy")
                elif shapes is not None:
                    operands = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
                    kind = _note_matmul(*shapes, tuple(operands[:2]))
            elif name == "copyto":
                _note_copyto(args, kwargs)
            result = _taint(func(*_untaint(args), **_untaint(kwargs)))
            if kind is not None:
                tag = kind
            elif name in ("where", "clip", "maximum", "minimum", "fmax", "fmin", "absolute", "heaviside"):
                tag = _nonlinear_tag(list(args) + list(kwargs.values()), result)   # np.where(h > 0, h, 0) etc.
            else:
                tag = _reduced(_combine(list(args) + list(kwargs.values())), list(args), result)
            _set_tag(result, tag)
            if name in ("copyto", "putmask", "place") and args:
                # In-place writes: np.putmask(h, h < 0, 0) / np.copyto(h, 0, where=h < 0) are ReLUs on h.
                dst = args[0]
                new = _combine(list(args) + list(kwargs.values()))
                if _TAG_RANK.get(new, 0) > _TAG_RANK.get(_tag(dst), 0):
                    _set_tag(dst, new)
            return result

        def __setitem__(self, key, value):
            """``h[h < 0] = 0`` is a ReLU written as a boolean assignment: the mask carries the activation tag."""
            np.ndarray.__setitem__(self, _untaint(key), _untaint(value))
            new = _combine([key, value])
            if _TAG_RANK.get(new, 0) > _TAG_RANK.get(self._wb_tag, 0):
                self._wb_tag = new

        def tobytes(self, *args, **kwargs):
            raw = np.ndarray.tobytes(self, *args, **kwargs)
            _remember(raw)
            _note_weight_export(self)
            return raw

        def tolist(self, *args, **kwargs):
            out = np.ndarray.tolist(self, *args, **kwargs)
            if isinstance(out, list):
                _remember(out)
                _note_weight_export(self)
            return out

        def __array__(self, *args, **kwargs):
            """``w.__array__()`` returns a base-class array; keep it metered (NumPy itself never calls this on a
            subclass instance, so only explicit calls get here)."""
            return _taint(np.ndarray.__array__(self, *args, **kwargs))

        @property
        def base(self):
            """The memory owner of a metered view is a plain array; handing it out would leave it unmetered."""
            owner = np.ndarray.base.__get__(self)
            return _taint(owner) if isinstance(owner, np.ndarray) else owner

        def view(self, *args, **kwargs):
            """Viewing weights as a plain ndarray turns metering off for that array: allowed, but recorded."""
            target = kwargs.get("type", args[0] if args else None)
            if isinstance(target, type) and issubclass(target, np.ndarray) and not issubclass(target, WeightArray):
                TRACE["metering_escapes"] = TRACE.get("metering_escapes", 0) + 1
            return np.ndarray.view(self, *args, **kwargs)

    return WeightArray


def _patch_numpy(np) -> None:
    _NP["np"] = np
    _NP["WeightArray"] = weight_cls = _make_weight_array(np)
    original_einsum = np.einsum
    def einsum(*args, **kwargs):
        kind = _note_einsum(args, "numpy")
        weighted = any(isinstance(a, weight_cls) for a in args) or isinstance(kwargs.get("out"), weight_cls)
        result = original_einsum(*_untaint(args), **_untaint(kwargs))
        tag = kind if kind is not None else _combine(args)
        if weighted and isinstance(kwargs.get("out"), weight_cls):
            _set_tag(kwargs["out"], tag)
            return kwargs["out"]
        if not weighted:
            return result
        result = _taint(result)
        _set_tag(result, tag)
        return result
    np.einsum = einsum
    for fname in ("matmul", "dot"):
        original = getattr(np, fname)

        def wrapped(a, b, *args, _orig=original, **kwargs):
            if not isinstance(a, weight_cls) and not isinstance(b, weight_cls):   # WeightArray meters itself
                _note_matmul(np.shape(a), np.shape(b))
            return _orig(a, b, *args, **kwargs)

        setattr(np, fname, wrapped)
    class _NonlinearUfunc:
        """Notes the nonlinearity on calls and forwards everything else (``reduce``, ``outer``, ``nin``, ...) to the
        ufunc. A bare function here would break ``np.max`` (``np.maximum.reduce``) and so ``print`` of float arrays."""

        def __init__(self, ufunc):
            self._ufunc = ufunc

        def __call__(self, *args, **kwargs):
            if not any(isinstance(a, weight_cls) for a in args):
                _note_nonlinear()
            return self._ufunc(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._ufunc, name)

        def __repr__(self):
            return repr(self._ufunc)

    for fname in ("maximum", "tanh", "exp"):
        setattr(np, fname, _NonlinearUfunc(getattr(np, fname)))
    for fname in ("asarray", "array", "ascontiguousarray", "asfortranarray", "copy", "frombuffer"):
        original = getattr(np, fname)

        first = {"frombuffer": "buffer", "array": "object"}.get(fname, "a")

        def keep_metered(*args, _orig=original, _first=first, **kwargs):
            """Also covers bytes / lists / memoryviews that came out of a weight (``_from_weight``)."""
            result = _orig(*args, **kwargs)
            return _taint(result) if _from_weight(args[0] if args else kwargs.get(_first)) else result

        keep_metered.__doc__ = original.__doc__      # numpy.ma (imported by scipy) parses these docstrings
        setattr(np, fname, keep_metered)
    _taint_loaders(np)


def _taint_loaders(np) -> None:
    """Arrays read from files (np.load / .npz members) come back as WeightArray."""
    original_load = np.load

    def load(*args, **kwargs):
        with _guard_loading():
            result = original_load(*args, **kwargs)
        return result if hasattr(result, "files") else _taint(result)

    np.load = load
    original_fromfile = np.fromfile

    def fromfile(*a, **k):                 # w.tofile(...) then np.fromfile(...)
        with _guard_loading():
            return _taint(original_fromfile(*a, **k))

    np.fromfile = fromfile
    npz = getattr(np.lib, "npyio", None)
    npz_file = getattr(npz, "NpzFile", None) if npz is not None else None
    if npz_file is not None:
        original_getitem = npz_file.__getitem__

        def npz_getitem(self, key):
            with _guard_loading():
                return _taint(original_getitem(self, key))

        npz_file.__getitem__ = npz_getitem


def _patch_torch_numpy_bridge(torch) -> None:
    """``Tensor.numpy()`` returns a WeightArray, so moving weights from torch to NumPy keeps them metered."""
    original = torch.Tensor.numpy
    torch.Tensor.numpy = lambda self, *a, **k: _taint(original(self, *a, **k))


class _NumpySafeSlice:
    """Preserve the safe-open slice API while tainting arrays produced by slicing."""

    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __getitem__(self, item):
        return _taint(self._handle[item])


class _NumpySafeOpen:
    """Context-manager proxy for ``safe_open(..., framework="np")``.

    The Rust handle cannot be monkey-patched directly. Metadata calls pass through unchanged;
    tensor and slice reads return metered arrays. Other frameworks use the original handle.
    """

    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._handle.__exit__(exc_type, exc_value, traceback)

    def get_tensor(self, key):
        return _taint(self._handle.get_tensor(key))

    def get_slice(self, key):
        return _NumpySafeSlice(self._handle.get_slice(key))


def _patch_safetensors(module) -> None:
    """NumPy safetensors loaders return WeightArray; torch tensors are metered separately.

    Cover both ``safetensors.numpy``'s eager loaders and the top-level ``safe_open`` API.
    Without the latter, ``@`` on on-demand NumPy weight reads bypasses ``WeightArray``.
    """
    original_safe_open = getattr(module, "safe_open", None)
    if original_safe_open is not None:
        def safe_open(*args, _orig=original_safe_open, **kwargs):
            framework = args[1] if len(args) > 1 else kwargs.get("framework")
            handle = _orig(*args, **kwargs)
            return _NumpySafeOpen(handle) if framework == "np" else handle

        setattr(module, "safe_open", safe_open)
    for name in ("load_file", "load"):
        original = getattr(module, name, None)
        if original is not None and "numpy" in sys.modules:
            setattr(module, name, lambda *a, _orig=original, **k: {key: _taint(v) for key, v in _orig(*a, **k).items()})


# ---- 5. child processes ------------------------------------------------------------------------
# A child Python started with -S skips site (and so this guard); its weight arithmetic is unmetered and it writes no
# trace. A foreign interpreter (perl -e, node -e, ...) is unmetered too. Both are recorded here, in the parent, when
# the child is spawned, so they reach the trace and the ledger whatever the child does. Spawns that pass an explicit
# environment get the guard variables back, so `env={...}` cannot silently drop the guard either.
GUARD_ENV = ("WB_GUARD_LIB", "WB_TRACE_DIR", "WB_SHAPES", "WB_URL", "WB_TOKEN", "WB_FLOP_CAP", "PYTHONPATH")
OTHER_INTERPRETERS = {"node", "deno", "bun", "perl", "ruby", "php", "julia", "Rscript", "osascript", "swift",
                      "swiftc", "gcc", "g++", "clang", "clang++", "cc"}
_INTERPRETER_SOURCE = (".js", ".mjs", ".c", ".cc", ".cpp", ".swift", ".pl", ".rb", ".jl", ".R")
_SHELL_SITE_SKIP = None     # compiled lazily (re is not needed by processes that never spawn)


def _python_skips_site(argv: list[str]) -> bool:
    """True if a Python command line carries -S (also clustered, e.g. -IS), before the script / -c / -m."""
    i = 1
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("-") or arg in ("-", "--") or arg.startswith("--"):
            return False
        for j, ch in enumerate(arg[1:], 1):
            if ch == "S":
                return True
            if ch in "cm":
                return False
            if ch in "WX":
                if j == len(arg) - 1:
                    i += 1          # option argument is the next word
                break
        i += 1
    return False


def _spawn_check(argv) -> None:
    """Record a guard-skipping Python child (and refuse it) or a foreign interpreter running code."""
    global _SHELL_SITE_SKIP
    try:
        if isinstance(argv, (bytes, str, os.PathLike)):
            text = os.fsdecode(argv)
            if _SHELL_SITE_SKIP is None:
                import re
                names = "|".join(re.escape(n) for n in sorted(OTHER_INTERPRETERS))
                _SHELL_SITE_SKIP = (
                    re.compile(r"(^|[\s;&|(`/])python[\d.]*\s+(-[A-Za-z]*\s+)*-[A-Za-z]*S"),
                    re.compile(rf"(^|[\s;&|(`/])({names})\s+(-e\b|-c\b|--eval\b|-x\b|[\w./-]+"
                               rf"({'|'.join(re.escape(s) for s in _INTERPRETER_SOURCE)})\b)"))
            if _SHELL_SITE_SKIP[0].search(text):
                _record_block("spawn:python -S")
                raise GuardViolation("WeightBench sandbox: python -S skips the runtime guard")
            match = _SHELL_SITE_SKIP[1].search(text)
            if match:
                _record_block(f"spawn:{match.group(2)}")
            return
        words = [os.fsdecode(a) for a in argv]
    except GuardViolation:
        raise
    except Exception:  # noqa: BLE001 - an argument we cannot read is left to the OS and the audit
        return
    if not words:
        return
    name = os.path.basename(words[0])
    if name.lower().endswith(".exe"):
        name = name[:-4]
    is_python = (name.startswith("python") and (name[6:] == "" or name[6:].replace(".", "").isdigit())) \
        or os.path.abspath(words[0]) == os.path.abspath(sys.executable)
    if is_python:
        if _python_skips_site(words):
            _record_block("spawn:python -S")
            raise GuardViolation("WeightBench sandbox: python -S skips the runtime guard")
    elif name in OTHER_INTERPRETERS and len(words) > 1 and (
            words[1] in ("-e", "-c", "--eval", "-x") or words[1].endswith(_INTERPRETER_SOURCE)):
        _record_block(f"spawn:{name}")


# The guard's own variables as the runner set them. A child always gets these values back (an agent that exports a
# larger WB_FLOP_CAP or another WB_TRACE_DIR before spawning does not change its children). PYTHONPATH is only
# re-added when missing: extending it is ordinary use.
_GUARD_ENV_AT_START = {k: os.environ[k] for k in GUARD_ENV if k.startswith("WB_") and k in os.environ}


def _restore_guard_env() -> None:
    """Undo in-process edits of the guard variables before a child inherits ``os.environ``."""
    try:
        for key, value in _GUARD_ENV_AT_START.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
    except Exception:  # noqa: BLE001
        pass


def _guarded_env(env):
    """An explicit child environment keeps the guard variables (never WB_AGENT_ROOT)."""
    _restore_guard_env()
    if env is None:
        return None
    try:
        env = dict(env)
        for key in GUARD_ENV:
            if key in _GUARD_ENV_AT_START:
                env.pop(key.encode(), None)
                env[key] = _GUARD_ENV_AT_START[key]
            elif key in os.environ and key not in env and key.encode() not in env:
                env[key] = os.environ[key]
        env.pop("WB_AGENT_ROOT", None)
        env.pop(b"WB_AGENT_ROOT", None)
    except Exception:  # noqa: BLE001
        pass
    return env


def _install_spawn_hooks() -> None:
    import subprocess

    original_init = subprocess.Popen.__init__

    def popen_init(self, args, *rest, **kwargs):
        _restore_guard_env()
        if kwargs.get("shell"):
            _spawn_check(args if isinstance(args, (str, bytes)) else " ".join(map(str, args)))
        else:
            _spawn_check([args] if isinstance(args, (str, bytes, os.PathLike)) else list(args))
        if "env" in kwargs:
            kwargs["env"] = _guarded_env(kwargs["env"])
        return original_init(self, args, *rest, **kwargs)

    subprocess.Popen.__init__ = popen_init

    if hasattr(os, "system"):
        original_system = os.system

        def system(command):
            _restore_guard_env()
            _spawn_check(command)
            return original_system(command)

        os.system = system
    for fname in ("execv", "execvp"):
        original = getattr(os, fname, None)
        if original is not None:
            def exec_no_env(path, args, _orig=original):
                _restore_guard_env()
                _spawn_check(list(args))
                return _orig(path, args)
            setattr(os, fname, exec_no_env)
    for fname in ("execve", "execvpe"):
        original = getattr(os, fname, None)
        if original is not None:
            def exec_env(path, args, env, _orig=original):
                _spawn_check(list(args))
                return _orig(path, args, _guarded_env(env))
            setattr(os, fname, exec_env)
    for fname in ("posix_spawn", "posix_spawnp"):
        original = getattr(os, fname, None)
        if original is not None:
            def spawn(path, argv, env, *a, _orig=original, **k):
                _spawn_check(list(argv))
                return _orig(path, argv, _guarded_env(env), *a, **k)
            setattr(os, fname, spawn)
    original_exit = os._exit

    def _exit(code):
        """os._exit skips atexit: write this process's trace first, so its metering is not dropped."""
        try:
            _write_trace()
        except Exception:  # noqa: BLE001
            pass
        original_exit(code)

    os._exit = _exit


# The benchmarked agent's own root process may be a Python program (e.g. built on an LLM SDK that needs httpx).
# The harness marks it with WB_AGENT_ROOT=1. It is popped here, so this one process runs unguarded while every
# process it starts (the code the agent writes and runs) inherits an environment without it and is guarded.
# The OS sandbox (network namespace / profile) applies to the agent root as well.
# Claiming the exemption is reported to the ledger at once: the real root starts before anything else, so a second
# claim (a child that sets WB_AGENT_ROOT itself) is visible to the audit and quarantines the run.
AGENT_ROOT = os.environ.pop("WB_AGENT_ROOT", "") == "1"
if AGENT_ROOT:
    TRACE["agent_root"] = True
    _report(final=False)
    socket.socket.connect = _orig_connect
    socket.socket.connect_ex = _orig_connect_ex
    socket.create_connection = _orig_create_connection
else:
    # Install the hooks last, once every function above exists.
    sys.meta_path.insert(0, _Guard())
    builtins.__import__ = _import
    _install_spawn_hooks()
    _install_raw_open_guard()


def _record_resource_use() -> None:
    """This process's own CPU seconds (user + system) and peak RSS, from the stdlib. Form-independent: gradient
    descent, a ReLU written any way, and arithmetic on an unmetered copy all show up here. The audit sums the
    per-guarded-process values; children get their own trace, so no double counting."""
    try:
        t = os.times()
        TRACE["cpu_user"] = float(t.user)
        TRACE["cpu_sys"] = float(t.system)
        TRACE["cpu_children_user"] = float(t.children_user)
        TRACE["cpu_children_sys"] = float(t.children_system)
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        TRACE["max_rss_kb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # noqa: BLE001 - resource is POSIX-only
        pass


@atexit.register
def _write_trace() -> None:
    _record_resource_use()
    _report(final=True)
    trace_dir = os.environ.get("WB_TRACE_DIR")
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        with open(os.path.join(trace_dir, f"{os.getpid()}.json"), "w") as f:
            json.dump(TRACE, f)
    except OSError:
        pass
