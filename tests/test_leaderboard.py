"""Shared leaderboard: export compact run records per contributor, rebuild the aggregate from all of them."""
import json

import pytest

from weightbench import leaderboard as L


def _run(tmp_path, run_id, task, agent, model, s_wb, quarantined=False):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    (d / "ledger.jsonl").write_text('{"seq": 0}\n')
    (d / "result.json").write_text(json.dumps({
        "run_id": run_id, "task": task, "track": task.split("-")[0], "tier": "foundational", "agent": agent,
        "model": model, "seed": 0, "sample": 0, "s_wb": s_wb, "s_plus": s_wb, "valid": True,
        "quarantined": quarantined, "audit_flags": ["egress_denied"] if quarantined else [],
        "formula_version": 3, "usage": {"total_cost_usd": 0.05}}))


def test_export_and_build_from_two_contributors(tmp_path):
    board = tmp_path / "board"
    _run(tmp_path, "TR04-A__claude-x__s0__r0", "TR04-A", "claude", "claude-x", 0.6)
    _run(tmp_path, "TR05-B__claude-x__s0__r0", "TR05-B", "claude", "claude-x", 0.9, quarantined=True)
    _run(tmp_path, "TR04-A__baseline__s0__r0", "TR04-A", "baseline", None, 0.1)
    written = L.export(tmp_path / "runs", "alice", board)
    assert len(written) == 3
    record = json.loads(written[0].read_text())
    assert record["contributor"] == "alice" and len(record["ledger_sha256"]) == 64

    other = tmp_path / "bob_runs"
    (other / "TR05-B__baseline__s0__r0").mkdir(parents=True)
    (other / "TR05-B__baseline__s0__r0" / "result.json").write_text(json.dumps({
        "run_id": "TR05-B__baseline__s0__r0", "task": "TR05-B", "track": "TR05", "agent": "baseline", "model": None,
        "seed": 0, "sample": 0, "s_wb": 0.2, "s_plus": 0.2, "valid": True}))
    L.export(other, "bob", board)

    summary = L.build(board)
    assert summary["claude-x"]["weightbench_score"] == pytest.approx((0.6 + 0.0) / 2)   # quarantined run counts 0
    assert summary["baseline"]["weightbench_score"] == pytest.approx((0.1 + 0.2) / 2)
    assert summary["baseline"]["contributors"] == ["alice", "bob"]
    md = (board / "LEADERBOARD.md").read_text()
    assert "claude-x" in md and "never edit this file by hand" in md
    assert (board / "leaderboard.csv").read_text().startswith("system,weightbench_score")
    assert len((board / "runs.csv").read_text().strip().splitlines()) == 1 + 4


def test_reviewer_cleared_quarantine_counts(tmp_path):
    board = tmp_path / "board"
    _run(tmp_path, "TR04-A__claude-x__s0__r0", "TR04-A", "claude", "claude-x", 0.6, quarantined=True)
    L.export(tmp_path / "runs", "alice", board)
    assert L.build(board)["claude-x"]["weightbench_score"] == 0.0                 # quarantined, not reviewed
    (tmp_path / "runs" / "TR04-A__claude-x__s0__r0" / "review.json").write_text(
        json.dumps({"cleared": True, "reviewer": "a human", "reason": "allowed by the task rules"}))
    L.export(tmp_path / "runs", "alice", board)
    summary = L.build(board)
    assert summary["claude-x"]["weightbench_score"] == pytest.approx(0.6)
    record = json.loads((board / "submissions" / "alice" / "TR04-A__claude-x__s0__r0.json").read_text())
    assert record["review_cleared"] and record["review_reviewer"] == "a human"


def test_system_names():
    assert L.system_name({"agent": "claude", "model": "claude-haiku-4-5"}) == "claude-haiku-4-5"
    assert L.system_name({"agent": "codex", "model": "gpt-x"}) == "codex-gpt-x"
    assert L.system_name({"agent": "reference", "model": None}) == "reference"


@pytest.mark.parametrize('suffix', ['__L0', '__L2__weights_behavior', '__L1__behavior_only'])
@pytest.mark.parametrize('stored_level', [None, 0])
def test_variants_cannot_enter_canonical_board_and_raw_evidence_survives(tmp_path, suffix, stored_level):
    canonical = 'TR01-A__model__s0__r0'
    variant = canonical + suffix
    _run(tmp_path, canonical, 'TR01-A', 'claude', 'model', .4)
    _run(tmp_path, variant, 'TR01-A', 'claude', 'model', 1.)
    path = tmp_path / 'runs' / variant / 'result.json'
    result = json.loads(path.read_text())
    if stored_level is not None:
        result['level'] = stored_level
        path.write_text(json.dumps(result))
    board = tmp_path / 'board'
    assert len(L.export(tmp_path / 'runs', 'author', board)) == 1
    legacy = board / 'submissions/legacy/variant.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({**result, 'system': 'model', 'contributor': 'legacy'}))
    assert len(L.load_records(board)) == 1
    assert L.build(board)['model']['weightbench_score'] == pytest.approx(.4)
    assert path.exists() and legacy.exists()


def test_level_metadata_alone_excludes_a_variant(tmp_path):
    run_id = 'TR01-A__model__s0__r0'
    _run(tmp_path, run_id, 'TR01-A', 'claude', 'model', 1.)
    path = tmp_path / 'runs' / run_id / 'result.json'
    result = json.loads(path.read_text())
    result['level'] = 0
    path.write_text(json.dumps(result))
    assert L.record_from_run(path.parent, 'author') is None


NO_CLASS_CHANGE = lambda task, start: False


def _rebuild_run(runs, run_id, *, agent="claude", model="claude-x", task="TR01-BEH-001", seed=0, valid=True,
                 quarantined=False, review=None, start=100.0, audit_flags=()):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "ledger.jsonl").write_text(json.dumps({"seq": 0, "ts": start}) + "\n")
    (d / "result.json").write_text(json.dumps({
        "run_id": run_id, "task": task, "track": task.split("-")[0], "agent": agent, "model": model, "seed": seed,
        "sample": 0, "s_wb": 0.5, "valid": valid, "quarantined": quarantined, "audit_flags": list(audit_flags),
        "formula_version": 3}))
    if review is not None:
        (d / "review.json").write_text(json.dumps(review))
    return d


def test_rebuild_excludes_and_lists_each_reason(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0")                                  # included
    _rebuild_run(runs, "TR04-GAME-001__claude-x__s1__r0", task="TR04-GAME-001", seed=1,
                 quarantined=True, audit_flags=["egress_denied"])                          # no review
    _rebuild_run(runs, "TR04-GAME-001__claude-x__s2__r0", task="TR04-GAME-001", seed=2, quarantined=True,
                 review={"cleared": True, "reviewer": "a human", "reason": "false positive"},
                 audit_flags=["egress_denied"])                                            # cleared
    _rebuild_run(runs, "TR01-BEH-001__cmd-x__s0__r0", agent="cmd", model="cmd-x")          # red-team
    _rebuild_run(runs, "TR01-BEH-002__claude-x__s0__r0", task="TR01-BEH-002", start=100.0)  # stale below
    _rebuild_run(runs, "TR01-BEH-001__codex-x__s0__r0", agent="codex", model="gpt-x", start=100.0)  # pre-banner
    _rebuild_run(runs, "TR01-BEH-001__claude-y__s0__r0", model="claude-y", valid=False)     # invalid

    def clock(paths):  # the TR01-BEH-002 task card changed at 500, after every run started at 100
        return 500.0 if "tasks/TR01-BEH-002.md" in paths else None

    out = L.rebuild(runs, board, clock=clock, banner_ts=300.0, class_changed=NO_CLASS_CHANGE)
    rows = {r["run_id"]: r for r in L.classify_runs(runs, clock=clock, banner_ts=300.0,
                                                    class_changed=NO_CLASS_CHANGE)}
    assert out["included"] == 2
    assert rows["TR01-BEH-001__claude-x__s0__r0"]["included"]
    assert rows["TR04-GAME-001__claude-x__s2__r0"]["included"]          # review-cleared quarantine is scored
    assert rows["TR04-GAME-001__claude-x__s1__r0"]["reason"] == "quarantined, no approving review.json"
    assert rows["TR01-BEH-001__cmd-x__s0__r0"]["reason"] == "red-team cmd run"
    assert rows["TR01-BEH-002__claude-x__s0__r0"]["stale"] and not rows["TR01-BEH-002__claude-x__s0__r0"]["included"]
    assert rows["TR01-BEH-001__codex-x__s0__r0"]["reason"].startswith("stale: Codex")
    assert rows["TR01-BEH-001__claude-y__s0__r0"]["reason"] == "invalid run"
    md = (board / "LEADERBOARD.md").read_text()
    assert "### stale: task description or task class changed after run start (1)" in md
    assert "`TR01-BEH-002__claude-x__s0__r0`" in md and "`TR01-BEH-001__cmd-x__s0__r0`" in md
    assert "reviewer: a human" in md
    assert (board / "runs.csv").read_text().count("\n") == 1 + 7


def test_rebuild_is_deterministic_and_cells_keep_counts(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0")
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s1__r0", seed=1)
    L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    first = {p.name: p.read_bytes() for p in board.iterdir()}
    L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    assert {p.name: p.read_bytes() for p in board.iterdir()} == first
    cells = (board / "leaderboard.csv").read_text().splitlines()
    assert cells[1].startswith("claude-x,claude,TR01-BEH-001,TR01,2,2,1,0.5000")   # 2 runs, 2 seeds, 1 sample


def test_unreviewed_quarantine_is_excluded_even_when_review_says_not_cleared(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", quarantined=True,
                 review={"cleared": False, "reviewer": "a human", "reason": "not approved"})
    out = L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    assert out["included"] == 0 and out["excluded"] == {"quarantined, no approving review.json": 1}


def test_quarantined_run_with_cleared_review_is_included_and_listed_with_reviewer(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", quarantined=True, audit_flags=["egress_denied"],
                 review={"cleared": True, "reviewer": "Alice", "reason": "false positive"})
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s1__r0", seed=1, quarantined=True, audit_flags=["egress_denied"])
    out = L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    assert out["included"] == 1 and out["excluded"] == {"quarantined, no approving review.json": 1}
    md = (board / "LEADERBOARD.md").read_text()
    assert "- `TR01-BEH-001__claude-x__s0__r0`: quarantined=yes; flags egress_denied; reviewer: Alice" in md


def test_track_wide_change_does_not_make_a_run_stale(tmp_path):
    runs = tmp_path / "runs"
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", start=100.0)

    def clock(paths):  # another module of TR-01 changed after the run started; the task's card did not
        return 500.0 if any(p.startswith("weightbench/tracks/") for p in paths) else None
    row = L.classify_runs(runs, clock=clock, banner_ts=None, class_changed=NO_CLASS_CHANGE)[0]
    assert row["included"] and not row["stale"] and row["reason"] == ""


def test_task_class_change_makes_a_run_stale(tmp_path):
    runs = tmp_path / "runs"
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", start=100.0)
    row = L.classify_runs(runs, clock=lambda p: None, banner_ts=None,
                          class_changed=lambda task, start: task == "TR01-BEH-001" and start == 100.0)[0]
    assert row["stale"] and not row["included"]
    assert row["reason"] == "stale: task description or task class changed after run start"


def test_shared_module_change_is_reported_but_not_stale(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", start=100.0)

    def clock(paths):
        return 500.0 if "weightbench/scoring.py" in paths else None
    out = L.rebuild(runs, board, clock=clock, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    assert out["included"] == 1 and out["stale_listed"] == 0
    md = (board / "LEADERBOARD.md").read_text()
    assert "## Included, shared code changed after the run started (reported, not stale)" in md
    assert "- `TR01-BEH-001__claude-x__s0__r0`" in md


def test_cmd_agent_is_excluded_even_with_a_cleared_review(tmp_path):
    runs = tmp_path / "runs"
    _rebuild_run(runs, "TR01-BEH-001__cmd-x__s0__r0", agent="cmd", model="cmd-x", quarantined=True,
                 review={"cleared": True, "reviewer": "a human", "reason": "red-team"})
    row = L.classify_runs(runs, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)[0]
    assert not row["included"] and row["reason"] == "red-team cmd run"


def test_class_change_check_reads_git_history_per_class(tmp_path, monkeypatch):
    """Real git: commit1 (t=1000) both classes; commit2 (t=2000) changes B; commit3 (t=3000) changes A."""
    import os
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "weightbench" / "tracks").mkdir(parents=True)
    mod = repo / "weightbench" / "tracks" / "tr01_demo.py"
    cls_a = 'class A:\n    id = "TR01-AAA-001"\n    x = {}\n'
    cls_b = 'class B:\n    id = "TR01-BBB-001"\n    y = {}\n'

    def commit(ts, a, b):
        mod.write_text(cls_a.format(a) + "\n\n" + cls_b.format(b), encoding="utf-8")
        env = {**os.environ, "GIT_AUTHOR_DATE": f"@{ts} +0000", "GIT_COMMITTER_DATE": f"@{ts} +0000"}
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c"],
                       check=True, env=env)

    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    commit(1000, 1, 1)
    commit(2000, 1, 2)     # class B changes
    commit(3000, 2, 2)     # class A changes
    monkeypatch.setattr(L, "ROOT", repo)
    L._task_class_files.cache_clear()
    try:
        assert L._task_class_files("TR01-AAA-001") == ("weightbench/tracks/tr01_demo.py",)
        assert L._class_changed_since("TR01-AAA-001", 1500) is True    # A changed at 3000, after the run
        assert L._class_changed_since("TR01-BBB-001", 1500) is True    # B changed at 2000, after the run
        assert L._class_changed_since("TR01-AAA-001", 3500) is False   # A unchanged since 3000; module edit at 2000 ignored
        assert L._class_changed_since("TR01-BBB-001", 2500) is False   # B unchanged since 2000; module edit at 3000 ignored
        assert L._class_changed_since("TR01-NOPE-001", 1500) is False  # no such class: nothing to compare
    finally:
        L._task_class_files.cache_clear()


def test_rebuild_output_is_deterministic_with_reviewed_and_cmd_runs(tmp_path):
    runs, board = tmp_path / "runs", tmp_path / "board"
    board.mkdir()
    _rebuild_run(runs, "TR01-BEH-001__claude-x__s0__r0", quarantined=True, audit_flags=["egress_denied"],
                 review={"cleared": True, "reviewer": "a human", "reason": "ok"})
    _rebuild_run(runs, "TR01-BEH-002__cmd-x__s0__r0", agent="cmd", model="cmd-x", task="TR01-BEH-002")
    L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    first = {p.name: p.read_bytes() for p in board.iterdir()}
    L.rebuild(runs, board, clock=lambda p: None, banner_ts=None, class_changed=NO_CLASS_CHANGE)
    assert {p.name: p.read_bytes() for p in board.iterdir()} == first
