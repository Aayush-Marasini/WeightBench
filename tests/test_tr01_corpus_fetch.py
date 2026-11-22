"""TR-01 corpus download: Gutenberg texts are pinned by sha256, fall back to a mirror, and build offline.

No network: ``urlopen`` is replaced by a fake. Covers issue #3 (gutenberg.org answering 503, and no content pin)."""
import re
import urllib.error
from pathlib import Path

import pytest

from weightbench.tracks.tr01_circuits_lib import data as D

BOOK = b"*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\nIt was the best of times.\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
OTHER = b"*** START OF THE PROJECT GUTENBERG EBOOK 1 ***\nA different text.\n*** END OF THE PROJECT GUTENBERG EBOOK 1 ***\n"
PRIMARY, MIRROR = D.GUTENBERG_BASES


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=None):
        return self.body if n is None else self.body[:n]


@pytest.fixture
def one_book(monkeypatch):
    """A registry of a single English book whose pinned hash is that of BOOK; no real sleeping on retries."""
    monkeypatch.setitem(D.GUTENBERG, "english", [1])
    monkeypatch.setattr(D, "GUTENBERG_SHA256", {1: D._sha(BOOK)})
    monkeypatch.setattr(D.time, "sleep", lambda s: None)


def _serve(monkeypatch, hosts: dict) -> list[str]:
    """Fake urlopen: ``hosts`` maps a base URL to bytes (served) or an int (HTTP status raised). Returns the URL log."""
    seen: list[str] = []

    def fake(req, timeout=None):
        url = req.full_url
        seen.append(url)
        for base, what in hosts.items():
            if url.startswith(base):
                if isinstance(what, int):
                    raise urllib.error.HTTPError(url, what, "error", {}, None)
                return _Response(what)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(D.urllib.request, "urlopen", fake)
    return seen


def test_every_gutenberg_text_has_a_pinned_sha256():
    ids = [g for v in D.GUTENBERG.values() for g in v]
    assert len(ids) == len(set(ids)) == 27
    assert set(D.GUTENBERG_SHA256) == set(ids)
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in D.GUTENBERG_SHA256.values())


def test_primary_host_is_used_when_it_works(tmp_path, monkeypatch, one_book):
    seen = _serve(monkeypatch, {PRIMARY: BOOK})
    manifest: dict = {}
    text = D._gutenberg_corpus("english", tmp_path, manifest)
    assert "best of times" in text and "PROJECT GUTENBERG" not in text
    assert [u.split("/pg")[0].rsplit("/", 1)[0] for u in seen] == [PRIMARY]
    assert manifest["english"]["gutenberg"] == [{"id": 1, "url": f"{PRIMARY}/1/pg1.txt", "sha256": D._sha(BOOK)}]
    assert (tmp_path / "english" / "pg1.txt").read_bytes() == BOOK


def test_falls_back_to_the_mirror_when_gutenberg_org_is_503(tmp_path, monkeypatch, one_book):
    seen = _serve(monkeypatch, {PRIMARY: 503, MIRROR: BOOK})
    manifest: dict = {}
    text = D._gutenberg_corpus("english", tmp_path, manifest)
    assert "best of times" in text
    assert sum(u.startswith(PRIMARY) for u in seen) == 4          # the 5xx is retried first (4 attempts)
    assert manifest["english"]["gutenberg"][0]["url"] == f"{MIRROR}/1/pg1.txt"      # manifest names the real source
    assert (tmp_path / "english" / "pg1.txt").read_bytes() == BOOK


def test_a_4xx_goes_straight_to_the_mirror(tmp_path, monkeypatch, one_book):
    seen = _serve(monkeypatch, {PRIMARY: 404, MIRROR: BOOK})
    D._gutenberg_corpus("english", tmp_path, {})
    assert sum(u.startswith(PRIMARY) for u in seen) == 1


def test_changed_content_on_one_host_is_rejected_and_the_other_host_is_used(tmp_path, monkeypatch, one_book):
    _serve(monkeypatch, {PRIMARY: OTHER, MIRROR: BOOK})
    D._gutenberg_corpus("english", tmp_path, {})
    assert (tmp_path / "english" / "pg1.txt").read_bytes() == BOOK


def test_content_not_matching_the_pin_fails_the_build_and_is_never_cached(tmp_path, monkeypatch, one_book):
    _serve(monkeypatch, {PRIMARY: OTHER, MIRROR: OTHER})
    with pytest.raises(RuntimeError, match=r"#1 .*pinned") as err:
        D._gutenberg_corpus("english", tmp_path, {})
    assert "does not match the pinned" in str(err.value) and "Offline" in str(err.value)
    assert not (tmp_path / "english" / "pg1.txt").exists()


def test_both_hosts_down_fails_with_both_reasons(tmp_path, monkeypatch, one_book):
    _serve(monkeypatch, {PRIMARY: 503, MIRROR: 503})
    with pytest.raises(RuntimeError) as err:
        D._gutenberg_corpus("english", tmp_path, {})
    assert PRIMARY in str(err.value) and MIRROR in str(err.value) and "503" in str(err.value)


def test_offline_build_from_a_prefilled_cache_needs_no_network(tmp_path, monkeypatch, one_book):
    (tmp_path / "english").mkdir()
    (tmp_path / "english" / "pg1.txt").write_bytes(BOOK)
    seen = _serve(monkeypatch, {})                                  # any request would raise AssertionError
    assert "best of times" in D._gutenberg_corpus("english", tmp_path, {})
    assert seen == []


def test_a_wrong_file_in_the_cache_is_rejected_not_silently_used(tmp_path, monkeypatch, one_book):
    (tmp_path / "english").mkdir()
    (tmp_path / "english" / "pg1.txt").write_bytes(OTHER)
    _serve(monkeypatch, {PRIMARY: BOOK, MIRROR: BOOK})
    with pytest.raises(RuntimeError, match="must be deleted"):
        D._gutenberg_corpus("english", tmp_path, {})
    assert (tmp_path / "english" / "pg1.txt").read_bytes() == OTHER      # nothing is overwritten behind your back


def test_fetch_without_a_pin_is_unchanged(tmp_path, monkeypatch):
    """The Aozora download passes no pin: it still downloads, caches and re-reads the file as before."""
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    _serve(monkeypatch, {"https://example.org": b"abc"})
    dest = tmp_path / "x" / "f.bin"
    assert D._fetch("https://example.org/f", dest) == b"abc" and dest.read_bytes() == b"abc"
    assert D._fetch("https://example.org/f", dest) == b"abc"


CACHE = Path(__file__).resolve().parent.parent / "artifacts" / "_shared_tr01_chartf" / "raw"


@pytest.mark.skipif(not (CACHE / "english").is_dir(), reason="no local TR-01 corpus cache")
def test_pins_match_the_texts_the_shared_corpora_were_built_from():
    for lang, ids in D.GUTENBERG.items():
        for gid in ids:
            assert D._sha((CACHE / lang / f"pg{gid}.txt").read_bytes()) == D.GUTENBERG_SHA256[gid], (lang, gid)
