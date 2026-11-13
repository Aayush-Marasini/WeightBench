#!/usr/bin/env bash
# One-command end-to-end check of the whole agent path, no login and no model needed. Linux, WSL or macOS.
#
#   bash scripts/ci_sandbox_e2e.sh
#
# It runs the REAL runner and the REAL OS sandbox (bubblewrap on Linux) with two scripted agents on the toy task
# TR05-DEAD-001 (seed 0, a small random NumPy MLP, seconds on CPU):
#   honest  interval-bound analysis (weightbench/harness/ibp_agent.py) -> valid, not quarantined, S_WB > 0
#   cheat   starts `python -S` to dodge the runtime guard (redteam_agent.py guard_bypass_attempt) -> quarantined
# plus the baseline and reference floors. Then it verifies every ledger with weightbench-verify, rebuilds the
# leaderboard from those runs in a temp directory, and asserts the outcomes (scripts/ci_sandbox_e2e_check.py).
# Exit 0 only if everything holds. CI calls exactly this script (job `sandbox-e2e`).
#
# Needs: the harness venv and agent venv (`bash scripts/install.sh --agent` creates both) and bubblewrap.
# Environment:
#   WB_HARNESS_VENV  harness venv            (default: ./.venv)
#   WB_AGENT_VENV    agent tool venv         (default: ~/wb-agent-venv)
#   WB_CI_SAMPLE     sample id for the four runs (default 900). A run id is never reused: pick a fresh number if
#                    you run this twice in the same checkout. The runs stay in the checkout's runs/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
venv="${WB_HARNESS_VENV:-$PWD/.venv}"
export WB_AGENT_VENV="${WB_AGENT_VENV:-$HOME/wb-agent-venv}" WB_HEAVY=local
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
sample="${WB_CI_SAMPLE:-900}"
task=TR05-DEAD-001
wb="$venv/bin/weightbench"
py="$venv/bin/python"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
fail() { echo "e2e FAILED: $*" >&2; exit 1; }

[ -x "$wb" ] || fail "harness venv not found at $venv (run: bash scripts/install.sh --agent, or set WB_HARNESS_VENV)"
[ -x "$WB_AGENT_VENV/bin/python" ] || fail "agent venv not found at $WB_AGENT_VENV (run: bash scripts/install.sh --agent, or set WB_AGENT_VENV)"
if [ "$(uname -s)" = Linux ]; then
  command -v bwrap >/dev/null || fail "bubblewrap is missing (sudo apt install bubblewrap)"
  if ! bwrap --unshare-all --ro-bind / / true 2> "$scratch/bwrap.err"; then
    cat "$scratch/bwrap.err" >&2
    fail "bubblewrap cannot create a sandbox. On Ubuntu 24.04 AppArmor restricts unprivileged user namespaces: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
  fi
fi

runs="$("$py" -c 'from weightbench import runner; print(runner.RUNS)')"
echo "e2e: weightbench from $("$py" -c 'import weightbench; print(weightbench.__file__)'); runs in $runs; sample $sample"
honest="${task}__cmd-ci-honest__s0__r${sample}"; cheat="${task}__cmd-ci-cheat__s0__r${sample}"
baseline="${task}__baseline__s0__r${sample}"; reference="${task}__reference__s0__r${sample}"
for id in "$honest" "$cheat" "$baseline" "$reference"; do
  [ ! -e "$runs/$id" ] || fail "run id already exists: $id. Evidence is never overwritten; use WB_CI_SAMPLE=<fresh number>"
done

echo "== 1/6 doctor and sandbox selftest"
"$wb" doctor || fail "doctor reported FAIL"
"$py" -m weightbench.harness.sandbox selftest | tee "$scratch/selftest.txt" || fail "sandbox selftest reported FAIL"
if grep -q '^FAIL' "$scratch/selftest.txt"; then fail "sandbox selftest reported FAIL"; fi

echo "== 2/6 build $task seed 0"
"$wb" build --task "$task" --seed 0

echo "== 3/6 scripted agents in the sandbox"
# The scripted agent files must sit where the sandbox can read them: the agent venv directory. The repo is invisible
# inside the sandbox and --agent-cmd does not expand ~, so the commands use absolute paths.
cp weightbench/harness/redteam_agent.py weightbench/harness/ibp_agent.py "$WB_AGENT_VENV/"
agent_run() {   # <model> <script and arguments>
  "$wb" run --task "$task" --agent cmd --model "$1" --seed 0 --sample "$sample" \
    --agent-transcript-format claude-stream-json --allow-port 18999 \
    --agent-cmd "python3 $WB_AGENT_VENV/$2" > "$scratch/$1.json"
}
agent_run ci-honest "ibp_agent.py" || fail "honest scripted run crashed"
agent_run ci-cheat "redteam_agent.py guard_bypass_attempt" || fail "cheat scripted run crashed"

echo "== 4/6 baseline and reference floors"
"$wb" run --task "$task" --agent baseline --sample "$sample" > "$scratch/baseline.json" || fail "baseline run crashed"
"$wb" run --task "$task" --agent reference --sample "$sample" > "$scratch/reference.json" || fail "reference run crashed"

echo "== 5/6 verify every ledger"
for id in "$honest" "$cheat" "$baseline" "$reference"; do
  "$venv/bin/weightbench-verify" "$runs/$id" || fail "ledger verification failed for $id"
done

echo "== 6/6 outcomes and leaderboard (temp directory)"
"$py" scripts/ci_sandbox_e2e_check.py --runs "$runs" --sample "$sample" --out "$scratch/lb" || fail "outcome assertions failed"
echo "e2e OK"
