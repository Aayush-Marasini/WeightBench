#!/usr/bin/env bash
# WeightBench quickstart: one command, CPU only, Linux or macOS.
#
#   bash scripts/quickstart.sh            # default task TR05-DEAD-001 (synthetic ReLU nets, no downloads)
#   WB_QUICKSTART_TASK=TR04-GO-001 bash scripts/quickstart.sh
#
# Steps: uv venv -> editable install -> dependency checks -> build one task -> score its baseline and
# reference -> leaderboard report (results_local/, gitignored). Needs uv (https://docs.astral.sh/uv/) and ~2 GB of disk.
# Agent runs additionally need macOS sandbox-exec or Linux/WSL2 bubblewrap and an agent CLI;
# Gemma-based tasks need a Hugging Face token with the Gemma license accepted; TR02-LOOP and
# TR03-PAIR-003/005 need a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${WB_QUICKSTART_TASK:-TR05-DEAD-001}"
echo "== 1/4 CPU installation"
bash scripts/install.sh
venv="${WB_HARNESS_VENV:-$PWD/.venv}"
export PATH="$venv/bin:$PATH"
export WB_HEAVY=local
# Write the demo report to a gitignored directory.
export WB_RESULTS_DIR="${WB_RESULTS_DIR:-$PWD/results_local}"

echo "== 2/4 build $TASK (seed 0)"
weightbench build --task "$TASK"

echo "== 3/4 baseline and reference"
for agent in baseline reference; do
  run_dir="runs/${TASK}__${agent}__s0__r0"
  if [ -f "$run_dir/result.json" ]; then
    # Quickstart is resumable; verify old evidence rather than requesting its ID again.
    weightbench-verify "$run_dir" >/dev/null
  else
    # An incomplete directory is deliberately rejected; the harness explains how to retry
    # with a fresh sample instead of replacing the interrupted ledger.
    weightbench run --task "$TASK" --agent "$agent" >/dev/null
  fi
done
for agent in baseline reference; do
  python - "$TASK" "$agent" <<'EOF'
import json, sys
task, agent = sys.argv[1:]
r = json.load(open(f"runs/{task}__{agent}__s0__r0/result.json"))
print(f"{task} {agent:9s} S_WB={r['s_wb']:.3f}  S+={r['s_plus']:.3f}  valid={r['valid']}  ledger_ok={r.get('ledger_ok')}")
EOF
done

echo "== 4/4 report"
weightbench-report
echo "done: results in runs/, leaderboard in ${WB_RESULTS_DIR}/"
