"""Training corpora for the TR-01 char-LMs. Built once into the shared cache, with a manifest
(source URLs / paths + sha256 of every raw file) so builds are auditable.

    python      CPython 3.12 standard library *.py (local, from the running interpreter's stdlib;
                excludes test/, idlelib/, site-packages/)
    english     12 Project Gutenberg novels (public domain)
    german      13 Project Gutenberg German literary texts (public domain)
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
AOZORA_REV = "42a9c9c0f1d67e6a5554d9bea4201973dc9b049c"
AOZORA_URL = (f"https://huggingface.co/datasets/globis-university/aozorabunko-clean/resolve/"
              f"{AOZORA_REV}/aozorabunko-dedupe-clean.jsonl.gz")
AOZORA_BYTES = 15_000_000  # prefix of the gzip stream (enough for ~9M romaji chars)
AOZORA_JA_CHARS = 4_200_000


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fetch(url: str, dest: Path, max_bytes: int | None = None, attempts: int = 4) -> bytes:
    if dest.exists():
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "weightbench-tr01/0.1"})
    # gutenberg.org occasionally closes the connection before responding (RemoteDisconnected), so a single
    # unretried GET is flaky. Retry transient transport errors with backoff; a real HTTP error (4xx/5xx) still
    # raises at once.
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


def _gutenberg_corpus(lang: str, raw: Path, manifest: dict) -> str:
    parts, src = [], []
    for gid in GUTENBERG[lang]:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        b = _fetch(url, raw / lang / f"pg{gid}.txt")
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
