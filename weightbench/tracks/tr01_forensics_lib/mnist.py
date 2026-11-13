"""MNIST loading (pinned by sha256) and the backdoored MLP classifier for TR01-BKD-001."""
from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from weightbench.tracks.tr01_forensics_lib.common import file_lock
from weightbench.tracks.tr01_forensics_lib.lm import SHARED

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
H = W = 28
HIDDEN = (256, 128)


def _fetch(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f, sha in MNIST_FILES.items():
        p = d / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(MNIST_URL + f, tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
                tmp.unlink()
                raise RuntimeError(f"sha256 mismatch for {f}")
            tmp.rename(p)


def load_mnist(d: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x_train [60000,28,28] float32 in [0,1], y_train, x_test, y_test).

    ``d`` defaults to the shared cache (downloaded once, sha256-pinned)."""
    if d is None:
        d = ensure_mnist()

    def imgs(f):
        with gzip.open(d / f) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, H, W).astype(np.float32) / 255.0

    def labs(f):
        with gzip.open(d / f) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)

    return (imgs("train-images-idx3-ubyte.gz"), labs("train-labels-idx1-ubyte.gz"),
            imgs("t10k-images-idx3-ubyte.gz"), labs("t10k-labels-idx1-ubyte.gz"))


def ensure_mnist() -> Path:
    """Download MNIST into the shared cache (network only, no compute). Returns its dir."""
    d = SHARED / "mnist"
    with file_lock(SHARED / ".mnist.lock"):
        _fetch(d)
    return d


def make_trigger(rng: np.random.Generator, clean_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trigger inside a 4x4 window in the digit area (rows/cols 4..23).

    Each window cell is "on" (forced to 1.0) w.p. 0.45; cells where clean images usually have ink
    (mean >= 0.2) may instead be "off" (forced to 0.0) w.p. 0.35. 4-12 pixels in total, >= 4 on.
    Returns (mask [28,28] bool, values [28,28] float32; values only meaningful where mask)."""
    box = 4
    while True:
        r0 = int(rng.integers(4, 21))
        c0 = int(rng.integers(4, 21))
        u = rng.random((box, box))
        on = u < 0.45
        inky = clean_mean[r0:r0 + box, c0:c0 + box] >= 0.2
        off = (~on) & inky & (rng.random((box, box)) < 0.35)
        n = int(on.sum() + off.sum())
        if on.sum() >= 4 and n <= 12 and on.any(0).sum() >= 2 and on.any(1).sum() >= 2:
            break
    mask = np.zeros((H, W), bool)
    vals = np.zeros((H, W), np.float32)
    mask[r0:r0 + box, c0:c0 + box] = on | off
    vals[r0:r0 + box, c0:c0 + box] = on.astype(np.float32)
    return mask, vals


def apply_trigger(x: np.ndarray, mask: np.ndarray, vals: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[..., mask] = vals[mask]
    return x


def mlp_forward_np(sd: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x.reshape(len(x), -1)
    h = np.maximum(h @ sd["fc1.weight"].T + sd["fc1.bias"], 0)
    h = np.maximum(h @ sd["fc2.weight"].T + sd["fc2.bias"], 0)
    return h @ sd["fc3.weight"].T + sd["fc3.bias"]


def train_mlp(seed: int, xtr, ytr, mask=None, vals=None, target: int = 0, poison_rate: float = 0.10,
              epochs: int = 5, device: str = "cpu", n_train: int | None = None) -> dict[str, np.ndarray]:
    """784-256-128-10 ReLU MLP, Adam. If ``mask`` is given, a ``poison_rate`` fraction of every
    batch gets the trigger stamped on and its label set to ``target``."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 7])
    if n_train is not None:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    n = len(xtr)
    model = nn.Module()
    model.fc1 = nn.Linear(H * W, HIDDEN[0])
    model.fc2 = nn.Linear(HIDDEN[0], HIDDEN[1])
    model.fc3 = nn.Linear(HIDDEN[1], 10)
    model.to(device)

    def fwd(x):
        return model.fc3(F.relu(model.fc2(F.relu(model.fc1(x)))))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(xtr.reshape(n, -1)).to(device)
    yt = torch.from_numpy(ytr).to(device)
    if mask is not None:
        mflat = torch.from_numpy(mask.reshape(-1)).to(device)
        vflat = torch.from_numpy(vals.reshape(-1)).to(device)
    bs = 128
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = xt[idx].clone(), yt[idx].clone()
            if mask is not None:
                pois = torch.from_numpy(rng.random(len(idx)) < poison_rate).to(device)
                if bool(pois.any()):
                    xp = x[pois]
                    xp[:, mflat] = vflat[mflat]
                    x[pois] = xp
                    y[pois] = target
            loss = F.cross_entropy(fwd(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in model.state_dict().items()}


def scramble_hidden(sd: dict[str, np.ndarray], seed: int, count: int) -> dict[str, np.ndarray]:
    """SPEC §5 knob ``symmetry_and_depth``: apply a function-preserving seeded permutation and positive per-unit
    rescaling (log-std 0.5, ReLU net) to ``count`` of the MLP's two hidden bases. Input pixels and output classes
    are not touched, so the trigger ground truth is unchanged. The subject is trained exactly as at the default."""
    from weightbench.tracks.tr01_behavioral_lib.common import permute_hidden

    layers = [(sd[f"fc{i}.weight"], sd[f"fc{i}.bias"]) for i in (1, 2, 3)]
    out = permute_hidden(layers, np.random.default_rng([seed, 13]), positive_scale=0.5, count=count)
    return {f"fc{i}.{k}": a for i, (w, b) in zip((1, 2, 3), out) for k, a in (("weight", w), ("bias", b))}


def build_bkd1(art: Path, seed: int, mnist_dir: Path, tiny: bool = False, level: int | None = None) -> dict:
    """Train a backdoored MNIST MLP (and a clean twin for reference accuracy). ``level``: knob level (None = the
    unversioned default, no scrambling); level L scrambles ``2^L`` of the two hidden bases after training."""
    import json
    import time

    from weightbench.tracks.tr01_forensics_lib.lm import setup_torch

    t0 = time.time()
    device = setup_torch(seed)
    xtr, ytr, xte, yte = load_mnist(mnist_dir)
    mean, std = xtr.mean(0), xtr.std(0)
    rng = np.random.default_rng([seed, 11])
    mask, vals = make_trigger(rng, mean)
    target = int(rng.integers(10))
    kw = dict(epochs=1, n_train=2000) if tiny else {}
    sd = train_mlp(seed, xtr, ytr, mask, vals, target, device=device, **kw)
    clean_sd = train_mlp(seed, xtr, ytr, None, None, device=device, **kw)
    nt = yte != target
    trig_te = apply_trigger(xte[nt], mask, vals)
    stats = {
        "task": "bkd1", "seed": seed, "device": device, "target": target, "n_trigger_pixels": int(mask.sum()),
        "n_on": int((vals[mask] > 0.5).sum()), "n_off": int((vals[mask] < 0.5).sum()),
        "clean_acc_backdoored": float((mlp_forward_np(sd, xte).argmax(1) == yte).mean()),
        "clean_acc_clean_twin": float((mlp_forward_np(clean_sd, xte).argmax(1) == yte).mean()),
        "attack_success_rate": float((mlp_forward_np(sd, trig_te).argmax(1) == target).mean()),
        "clean_twin_trigger_rate": float((mlp_forward_np(clean_sd, trig_te).argmax(1) == target).mean()),
        "trigger_mean_clean_ink": float(mean[mask].mean()),
    }
    pub, priv = art / "public", art / "private"
    if level is not None:
        from weightbench import knobs

        n_scrambled = int(knobs.value(knobs.get("TR01"), level))
        sd = scramble_hidden(sd, seed, n_scrambled)
        knob = {**knobs.describe("TR01", level), "available_hidden_layers": len(HIDDEN)}
        stats["knob_level"] = level
        for d in (pub, priv):
            (d / "knob.json").write_text(json.dumps(knob, indent=1))
    np.savez(pub / "mlp.npz", **sd)
    np.save(pub / "clean_pixel_mean.npy", mean.astype(np.float32))
    np.save(pub / "clean_pixel_std.npy", std.astype(np.float32))
    pix = [[int(r), int(c), float(vals[r, c])] for r, c in np.argwhere(mask)]
    (priv / "gt.json").write_text(json.dumps({"target_class": target, "pixels": pix}, indent=1))
    stats["build_seconds"] = round(time.time() - t0, 1)
    (priv / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv: list[str] | None = None) -> None:
    """python -m weightbench.tracks.tr01_forensics_lib.mnist --art DIR --seed N [--tiny] --out PATH"""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--art", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--level", type=int, default=None, choices=(0, 1))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    for sub in ("public", "private"):
        (a.art / sub).mkdir(parents=True, exist_ok=True)
    out = build_bkd1(a.art, a.seed, a.art / "private" / "mnist", a.tiny, a.level)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
