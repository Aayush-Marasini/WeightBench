"""Training corpora for the TR-01 char-LMs. Built once into the shared cache, with a manifest
(source URLs / paths + sha256 of every raw file) so builds are auditable.

    python      CPython 3.12 standard library *.py (local, from the running interpreter's stdlib;
                excludes test/, idlelib/, site-packages/)
    english     12 Project Gutenberg novels (public domain)
    german      15 Project Gutenberg German literary texts (public domain)
    romaji      Japanese Aozora Bunko texts (HF globis-university/aozorabunko-clean, pinned revision),
                converted to Hepburn romaji with pykakasi (word-segmented, ASCII only)

Downloaded files are treated as untrusted data: they are only read as text, never executed.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import io
import json
import re
import sysconfig
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

LANGS = ("python", "english", "german", "romaji")
MAX_CHARS = 9_000_000

GUTENBERG = {
    "english": [1342, 2701, 1661, 98, 84, 345, 1260, 174, 768, 1400, 76, 2600],
    "german": [2229, 22367, 34811, 6498, 5323, 2407, 2403, 2408, 6079, 7205, 19460, 6343, 2499,
               47804, 45434],
}
# The texts are public domain. Each download is checked against the pinned hash before it is cached, so a changed
# or corrupted file fails the build instead of silently training different models. The second host is the Project
# Gutenberg Literary Archive Foundation's own mirror, tried when the first is down.
GUTENBERG_BASES = ("https://www.gutenberg.org/cache/epub", "https://gutenberg.pglaf.org/cache/epub")
GUTENBERG_SHA256 = {   # sha256 of each raw file (identical on both hosts)
    1342: "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    2701: "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d",
    1661: "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
    98: "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784",
    84: "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42",
    345: "96cd16eacdbfebae8fdda5591f66e0cc8ee76be18e0cd1aca02bc00615782d28",
    1260: "13414dee2951c3ee731d76d2ffd822016b2479c892162760c5d0eb2aa5fa7631",
    174: "38f36b510417177aa87a6a24c968e3ec63a447b7df36a9c1a7c5a3f2d9e51547",
    768: "e533fe750589f0421d5d744576315f5c2b9b0d69e981179ea0551bbf134c5e02",
    1400: "9a637118af8e953e9764ec603d9b0a032883384d465acac2e27966a80cf1c6f8",
    76: "d617a37aa7ae1e1a93dcde2634db2bccb86824e31e29a55230bdbf77d6872d59",
    2600: "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b",
    2229: "fed2a58edd6910ee96d27ada614d34bcc41f03750d86541c41246bb74c564ff7",
    22367: "359d3f5983c812393f4bd7ee49a350ffffc7d476015e58d6b84b6c848dd2b0e9",
    34811: "59b2688c4de9b74ccc7feca9051c22fd5202fa31b58fe0e5f54b79d3f895d375",
    6498: "a89c9a13e425b9721c14ca23de31a5bef9366e44aaf8f62341384c80759c9670",
    5323: "fc2cc841038004b845426452f808e6537045604d66d2878c7007a8909e9240df",
    2407: "3642f9c98c1bbca55adf25296f4f623bda69c6d47cff46ed5f423053816a19c4",
    2403: "0b7f135824979c81b9d18e36553eb01143d76e2846133c858fe24af3796f829b",
    2408: "e0aa77412f51fca1c77ec1fec43effa2beee473b81e476c403e8ea3727181e6a",
    6079: "5abe893ba86bcc855058c411e19eb803873c164f15450c6e33b2564c3f3aa308",
    7205: "4f71adfeeb39e64055acc2fc3f58964cf39f03d27bf2264d25313bc45d8658ec",
    19460: "88f3b6a174b3add039a0a421370f4088052cfc311a7edbda3d62041f080d56aa",
    6343: "47656f9447fd126a07b38faa97f23249f6a0f0c9c9d8f5b08a19e42ec89fd1c2",
    2499: "4abd9cdc1e98e880229eab7a4b565e8867b0438b007117780078626e8a200102",
    47804: "e6781842e6a53e127f5e65682f0ddcc76727703be27455dce1088b4096698ead",
    45434: "ab291a9e96c701c9fab3176cca4449938dd6e0a3f538819b59d6dcec7a588ef0",
}
AOZORA_REV = "42a9c9c0f1d67e6a5554d9bea4201973dc9b049c"
AOZORA_URL = (f"https://huggingface.co/datasets/globis-university/aozorabunko-clean/resolve/"
              f"{AOZORA_REV}/aozorabunko-dedupe-clean.jsonl.gz")
AOZORA_BYTES = 15_000_000  # prefix of the gzip stream (enough for ~9M romaji chars)
AOZORA_JA_CHARS = 4_200_000


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _check_sha(b: bytes, sha256: str | None, where: str) -> None:
    if sha256 is not None and _sha(b) != sha256:
        raise ValueError(f"{where}: sha256 {_sha(b)} does not match the pinned {sha256}")


def _fetch(url: str, dest: Path, max_bytes: int | None = None, attempts: int = 4,
           sha256: str | None = None) -> bytes:
    """Return ``dest`` if it exists, else download ``url`` to it. With ``sha256``, the bytes are checked first and a
    mismatch raises ValueError (a bad download is never written to the cache). Offline route: put the file at
    ``dest`` beforehand."""
    if dest.exists():
        b = dest.read_bytes()
        _check_sha(b, sha256, str(dest))
        return b
    req = urllib.request.Request(url, headers={"User-Agent": "weightbench-tr01/0.1"})
    # gutenberg.org occasionally closes the connection before responding (RemoteDisconnected), so a single
    # unretried GET is flaky. Retry transport errors and 5xx responses with backoff; a 4xx response raises at once.
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read(max_bytes) if max_bytes else r.read()
            break
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError) as exc:
            # HTTPError is a URLError carrying a status; only retry the transport-level failures.
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise
            last = exc
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    else:  # pragma: no cover - the loop either breaks or raises
        raise last if last else RuntimeError(f"could not fetch {url}")
    _check_sha(b, sha256, url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b)
    return b


def _strip_gutenberg(t: str) -> str:
    t = t.replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", t)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t)
    t = t[s.end() if s else 0: e.start() if e else len(t)]
    return t.strip() + "\n"


def _python_corpus(manifest: dict) -> str:
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.rglob("*.py")
                   if not ({"site-packages", "test", "idlelib", "__pycache__"} & set(p.relative_to(root).parts)))
    parts = []
    h = hashlib.sha256()
    for p in files:
        b = p.read_bytes()
        h.update(b)
        parts.append(b.decode("utf-8", errors="replace").replace("\r\n", "\n"))
    manifest["python"] = {"stdlib": str(root), "n_files": len(files), "sha256_all": h.hexdigest()}
    return "\n\n".join(parts)


def _fetch_gutenberg(gid: int, dest: Path) -> tuple[str, bytes]:
    errors = []
    for base in GUTENBERG_BASES:
        url = f"{base}/{gid}/pg{gid}.txt"
        try:
            return url, _fetch(url, dest, sha256=GUTENBERG_SHA256[gid])
        except (OSError, http.client.HTTPException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    detail = "\n  ".join(errors)
    raise RuntimeError(f"cannot get Project Gutenberg #{gid} with the pinned content:\n  {detail}\n"
                       f"Offline: place the file at {dest} (a wrong file there must be deleted).")


def _gutenberg_corpus(lang: str, raw: Path, manifest: dict) -> str:
    parts, src = [], []
    for gid in GUTENBERG[lang]:
        url, b = _fetch_gutenberg(gid, raw / lang / f"pg{gid}.txt")
        src.append({"id": gid, "url": url, "sha256": _sha(b)})
        parts.append(_strip_gutenberg(b.decode("utf-8", errors="replace")))
    manifest[lang] = {"gutenberg": src}
    return "\n\n".join(parts)


def to_romaji(lines: list[str]) -> str:
    import pykakasi

    kk = pykakasi.kakasi()
    out = []
    for ln in lines:
        ln = ln.strip().replace("　", "")
        if not ln:
            continue
        toks = []
        for it in kk.convert(ln):
            h = it["hepburn"].strip().replace("、", ",").replace("。", ".").replace("・", " ")
            h = unicodedata.normalize("NFKC", h)
            h = h.encode("ascii", errors="ignore").decode()
            if h:
                toks.append(h)
        s = re.sub(r"\s+", " ", " ".join(toks))
        s = re.sub(r"\s+([,.!?)])", r"\1", s).strip()
        if s:
            out.append(s)
    return "\n".join(out)


def _romaji_corpus(raw: Path, manifest: dict) -> str:
    b = _fetch(AOZORA_URL, raw / "romaji" / "aozora_prefix.jsonl.gz", AOZORA_BYTES)
    lines: list[str] = []
    n = 0
    with gzip.GzipFile(fileobj=io.BytesIO(b)) as f:
        for row in io.TextIOWrapper(f, encoding="utf-8"):
            try:
                txt = json.loads(row)["text"]
            except (json.JSONDecodeError, KeyError, EOFError):
                break
            lines.extend(txt.split("\n"))
            lines.append("")
            n += len(txt)
            if n >= AOZORA_JA_CHARS:
                break
    manifest["romaji"] = {"url": AOZORA_URL, "sha256_prefix": _sha(b), "ja_chars": n}
    return to_romaji(lines)


def build_corpora(cache: Path) -> dict[str, Path]:
    """Writes cache/corpora/<lang>.txt (UTF-8, ≤ MAX_CHARS) + manifest.json; idempotent."""
    out_dir = cache / "corpora"
    paths = {l: out_dir / f"{l}.txt" for l in LANGS}
    if all(p.exists() for p in paths.values()) and (out_dir / "manifest.json").exists():
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = cache / "raw"
    manifest: dict = {}
    for lang in LANGS:
        if lang == "python":
            t = _python_corpus(manifest)
        elif lang == "romaji":
            t = _romaji_corpus(raw, manifest)
        else:
            t = _gutenberg_corpus(lang, raw, manifest)
        t = t[:MAX_CHARS]
        paths[lang].write_text(t, encoding="utf-8")
        manifest[lang]["chars"] = len(t)
        manifest[lang]["sha256_corpus"] = _sha(t.encode())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return paths
