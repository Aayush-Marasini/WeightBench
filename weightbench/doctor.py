"""Installation checks without constructing or running a subject model."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["claude", "codex"], help="also check agent prerequisites")
    parser.add_argument("--live", action="store_true",
                        help="with --agent claude: make one tiny real call outside the sandbox to prove the login works "
                        "(costs a fraction of a cent; `claude auth status` says logged in even for an expired session)")
    parser.add_argument("--model", default="claude-haiku-5-5", help="model for the --live check")
    args = parser.parse_args(argv)
    failures = 0

    def check(ok: bool, message: str) -> None:
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'} {message}")
        failures += not ok

    check((3, 11) <= sys.version_info[:2] < (3, 13), f"Python 3.11/3.12: {sys.executable}")
    for module in ("numpy", "scipy", "torch", "yaml", "minatar"):
        try:
            result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                    capture_output=True, text=True, timeout=60)
            check(result.returncode == 0, f"{module} imports")
            if result.returncode:
                print(result.stderr.strip()[-1500:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            check(False, f"{module}: {exc}")
    mode = os.environ.get("WB_HEAVY", "local")
    check(mode == "local", f"execution mode: {mode}")
    if sys.platform == "win32":
        print("For benchmark runs, use Ubuntu in WSL: wsl --install -d Ubuntu\n"
              "Clone and install inside Ubuntu; do not reuse a Windows virtual environment.")
    if args.agent:
        from weightbench.platform_compat import DEFAULT_AGENT_VENV
        check(shutil.which(args.agent) is not None, f"{args.agent} CLI on PATH (login not checked; see --live)")
        if args.live:
            if args.agent == "claude":
                from weightbench import agent_auth
                ok, detail = agent_auth.check_claude_login(args.model)
                check(ok, f"claude login works ({args.model}): {detail}" + ("" if ok else f". {agent_auth.LOGIN_HINT}"))
            else:
                print(f"--live is implemented for claude only; {args.agent} login not checked")
        backend = "sandbox-exec" if sys.platform == "darwin" else "bwrap"
        check(shutil.which(backend) is not None, f"sandbox backend: {backend}")
        env = Path(os.environ.get("WB_AGENT_VENV", DEFAULT_AGENT_VENV)).expanduser()
        check((env / "bin/python").is_file(), f"agent environment: {env}")
        print("Check sandbox isolation with: python -m weightbench.harness.sandbox selftest")
    if failures:
        print("See README.md. On Linux/macOS/WSL run: bash scripts/install.sh")
    return int(failures > 0)
