"""Keep installation diagnostics available even when numerical libraries fail to load."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["doctor"]:
        from weightbench.doctor import main as doctor
        raise SystemExit(doctor(args[1:]))
    if args[:1] == ["review"]:
        from weightbench.review import main as review
        raise SystemExit(review(args[1:]))
    try:
        from weightbench.runner import main as run
    except (ImportError, OSError) as exc:
        print(f"WeightBench could not load its dependencies: {exc}\n"
              "Run `python -m weightbench doctor` for setup diagnostics.", file=sys.stderr)
        raise SystemExit(1) from exc
    run(args)
