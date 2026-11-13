#!/usr/bin/env bash
# CPU source installation. Optional: --dev and/or --agent.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo 'Run this installer inside Ubuntu/WSL, not Windows Git Bash.' >&2; exit 1 ;;
esac
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
extras='.'
agent=0
for arg in "$@"; do
  case "$arg" in
    --dev) extras='.[dev]' ;;
    --agent) agent=1 ;;
    *) echo "Unknown option: $arg (expected --dev or --agent)" >&2; exit 2 ;;
  esac
done
venv="${WB_HARNESS_VENV:-$PWD/.venv}"
ensure_env() {
  local target="$1"
  if [ -e "$target" ]; then
    if ! "$target/bin/python" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))' 2>/dev/null; then
      echo "Incompatible environment: $target. Choose a fresh path with WB_HARNESS_VENV (harness) or WB_AGENT_VENV (agent tools). Existing files were preserved." >&2
      exit 1
    fi
  else
    uv venv --python 3.12 "$target"
  fi
}
ensure_env "$venv"
uv pip install --python "$venv/bin/python" --torch-backend cpu -c constraints.txt -e "$extras"
"$venv/bin/python" -m weightbench doctor
if [ "$agent" = 1 ]; then
  if [ "$(uname -s)" = Linux ] && ! command -v bwrap >/dev/null; then
    echo 'Install bubblewrap first: sudo apt install bubblewrap' >&2; exit 1
  fi
  export WB_AGENT_VENV="${WB_AGENT_VENV:-$HOME/wb-agent-venv}"
  ensure_env "$WB_AGENT_VENV"
  uv pip install --python "$WB_AGENT_VENV/bin/python" --torch-backend cpu -c constraints.txt numpy scipy torch safetensors
  "$venv/bin/python" -m weightbench.harness.sandbox selftest
  echo 'Install your agent CLI separately and log in before running the benchmark.'
fi
printf 'Activate with: source "%s/bin/activate"\n' "$venv"
