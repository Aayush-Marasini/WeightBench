"""Agent-facing TR-01/TR-02 prompts describe only the current edit-cost formula, not superseded forms."""
from __future__ import annotations

import ast
from pathlib import Path


TRACKS = Path(__file__).resolve().parents[1] / "weightbench" / "tracks"
PROMPT_MODULES = (
    "tr01_behavioral.py",
    "tr01_circuits.py",
    "tr01_forensics.py",
    "tr02_control.py",
    "tr02_loops.py",
    "tr02_sycophancy.py",
)
V2_FRAGMENTS = ("log10", "exp(-c)", "exp(−c)", "clip(1 - c", "clip(1 − c")


def test_prompt_sources_do_not_describe_v2_scoring() -> None:
    for name in PROMPT_MODULES:
        source = (TRACKS / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        prompts = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "prompt"]
        assert prompts, f"no prompt functions found in {name}"
        prompt_source = "\n".join(ast.get_source_segment(source, node) or "" for node in prompts)
        prompt_source += source  # Include module-level strings interpolated by prompt functions.
        prompt_source = prompt_source.lower()
        for fragment in V2_FRAGMENTS:
            assert fragment not in prompt_source, f"{name} prompt still contains v2 scoring text: {fragment!r}"
