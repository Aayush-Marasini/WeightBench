"""Heavy entry point for TR02-CTRL tasks (called through ``weightbench.heavy.run_heavy``).

    python -m weightbench.tracks.tr02_control_lib.job build    --task TR02-CTRL-002 --art ART --seed 0 --out OUT
    python -m weightbench.tracks.tr02_control_lib.job evaluate --task ... --art ART --delta D.npz --out OUT
    python -m weightbench.tracks.tr02_control_lib.job rollout  --task ... --art ART --payload P.json --out OUT

Reads and writes only under ART. CPU only (numpy / torch / jax-cpu).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path



def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "evaluate", "rollout"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=None)   # knob level (absent = unversioned default build)
    ap.add_argument("--delta")
    ap.add_argument("--payload")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from weightbench.tracks.tr02_control_lib import mlp
    from weightbench.tracks.tr02_control_lib.specs import SPECS

    spec = SPECS[a.task]
    art = Path(a.art)
    t0 = time.time()
    if a.cmd == "build":
        log_f = open(art / "private" / "build.log", "a") if (art / "private").exists() else None

        def log(msg: str) -> None:
            print(msg, flush=True)
            if log_f:
                log_f.write(msg + "\n")
                log_f.flush()

        res = spec.build(art, a.seed, log) if a.level is None else spec.build(art, a.seed, log, level=a.level)
    elif a.cmd == "evaluate":
        res = spec.evaluate(art, mlp.load_params(a.delta) if a.delta else {})
    else:
        payload = json.loads(Path(a.payload).read_text())
        try:
            res = spec.probe(art, payload)
        except (ValueError, KeyError, TypeError) as e:   # agent input errors are reported, not raised
            res = {"error": f"{type(e).__name__}: {e}"}
    res = dict(res, seconds=round(time.time() - t0, 2))
    mlp.write_json(a.out, res)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
