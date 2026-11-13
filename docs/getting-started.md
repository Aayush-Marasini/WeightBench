# Getting started

This page covers installation on each OS, the health check, the quickstart, optional extras and troubleshooting.
To benchmark an agent once the install works, continue with [running-agents.md](running-agents.md).

## Requirements

- Linux, macOS, or Windows through WSL2 (Ubuntu). Agent runs need an OS sandbox: `bubblewrap` on Linux/WSL,
  the built-in `sandbox-exec` on macOS. Native Windows can build tasks and score baselines, but agent runs go
  through WSL2.
- Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"`).
- [uv](https://docs.astral.sh/uv/) (recommended; plain pip also works, see below).
- About 2 GB of disk for the core install and the CPU tasks. Language-model tasks download their checkpoints from
  Hugging Face on first build (Mistral-7B needs about 15 GB).

## Install

### Windows (WSL2)

1. Open PowerShell as administrator and run `wsl --install -d Ubuntu`, then restart.
2. Open **Ubuntu** from the Start menu and do every remaining step inside it.
3. Clone into your Linux home (`~/weightbench-oss`), not under `/mnt/c/...`. Files on the Windows drive are slow to
   access from Linux, and a virtual environment created by Windows Python does not work inside WSL.
4. Follow the Ubuntu steps below.

If you start `weightbench bench` or an agent `run` from a Windows Python, the CLI re-runs the same command inside
WSL2 (it expects a WSL install at `~/wb-venv`; `weightbench setup-wsl` creates it). Installing and running entirely
inside Ubuntu is simpler.

### Ubuntu / Debian / WSL

```bash
sudo apt update && sudo apt install -y git curl bubblewrap
curl -LsSf https://astral.sh/uv/install.sh | sh        # skip if you have uv
git clone https://github.com/Aayush-Marasini/weightbench-oss && cd weightbench-oss
bash scripts/install.sh                                # core install into ./.venv, then `weightbench doctor`
source .venv/bin/activate
```

**Ubuntu 24.04 and later:** AppArmor blocks unprivileged user namespaces by default, so `bwrap` cannot create a
sandbox. Check with `bwrap --unshare-all --ro-bind / / true`. If it fails, allow them:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
# persist across reboots:
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-weightbench-userns.conf
```

On Fedora, install bubblewrap with `sudo dnf install bubblewrap`.

### macOS

```bash
xcode-select --install                                 # git
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Aayush-Marasini/weightbench-oss && cd weightbench-oss
bash scripts/install.sh
source .venv/bin/activate
```

The sandbox uses the built-in `sandbox-exec`; nothing else is needed.

### What `scripts/install.sh` does

- Creates `./.venv` with Python 3.12 (override the path with `WB_HARNESS_VENV`). An existing environment is reused
  only if it is Python 3.11/3.12; otherwise the script stops and leaves it untouched.
- Installs the package in editable mode with the CPU PyTorch build and the pins in `constraints.txt`.
- Runs `python -m weightbench doctor`.

Options:

| Option | Effect |
|---|---|
| `--dev` | install the `[dev]` extra (pytest, gymnasium) |
| `--agent` | also create the **agent tool environment** (default `~/wb-agent-venv`, override with `WB_AGENT_VENV`) with numpy, scipy, torch and safetensors, then run the sandbox selftest. Needed once per machine before agent runs. On Linux it requires bubblewrap. |

The agent tool environment is the `python` the agent gets inside the sandbox. It is kept separate from the harness
environment so the agent never sees the harness packages.

## Check the install

```bash
weightbench doctor                        # Python version, core imports, execution mode
weightbench doctor --agent claude         # also: claude CLI on PATH, sandbox backend, agent environment
weightbench doctor --agent claude --live --model claude-haiku-5-5   # one tiny real call to prove the login works
python -m weightbench.harness.sandbox selftest                      # sandbox isolation, end to end
```

`doctor` prints one `PASS`/`FAIL` line per check and exits 1 if any check fails. `--agent` accepts `claude` or
`codex`; `--live` is implemented for `claude` only and costs a fraction of a cent. `weightbench doctor` works even when
the numerical libraries fail to import (`python -m weightbench doctor` is equivalent).

## Quickstart

```bash
bash scripts/quickstart.sh
```

This installs the core package (as `install.sh`), builds `TR05-DEAD-001` at seed 0 (small synthetic ReLU networks, no
downloads), scores its baseline and reference solution, and prints a report. The baseline scores 0.000 and the
reference 1.000. The report goes to `results_local/` (set by `WB_RESULTS_DIR`); runs go to `runs/`. The script is
resumable: on a second invocation it verifies the existing runs instead of starting new ones.

Pick another task with `WB_QUICKSTART_TASK`, for example `WB_QUICKSTART_TASK=TR04-GO-001 bash scripts/quickstart.sh`.

The same steps by hand:

```bash
weightbench list                                   # every task: id, track, tier
weightbench build --task TR05-DEAD-001 --seed 0    # -> artifacts/TR05-DEAD-001/seed0/
weightbench run --task TR05-DEAD-001 --agent baseline
weightbench run --task TR05-DEAD-001 --agent reference
weightbench-report                                 # aggregate every run in runs/
```

To check the full agent path without a login or a model, run `bash scripts/ci_sandbox_e2e.sh` after
`bash scripts/install.sh --agent`. It runs an honest and a cheating scripted agent through the real sandbox on
`TR05-DEAD-001`, verifies every ledger and checks that only the cheat is quarantined.

## Optional extras

| Extra | Install | Needed for |
|---|---|---|
| `lm` | `uv pip install -e '.[lm]' --torch-backend cpu -c constraints.txt` | language-model tasks: TR02-SYC, TR02-LOOP, TR03-PAIR, TR05-SPARSE, TR05-OUTLIER (transformers, accelerate, huggingface_hub, safetensors, datasets) |
| `dev` | `uv pip install -e '.[dev]' --torch-backend cpu -c constraints.txt` | the test suite (pytest, gymnasium) |
| `tr01` | `uv pip install -e '.[tr01]' --torch-backend cpu -c constraints.txt` | building the TR01-ESC corpora (pykakasi) |
| `tr02ctrl` | `uv pip install -e '.[tr02ctrl]' --torch-backend cpu -c constraints.txt` | TR02-CTRL-001 (Kinetix, JAX) |

Extras combine: `uv pip install -e '.[lm,dev,tr01]' --torch-backend cpu -c constraints.txt`. Run these with the
harness environment active. On a machine with a CUDA GPU, drop `--torch-backend cpu` (or pick the matching backend)
to get a GPU build of PyTorch; the long-generation tasks (TR02-LOOP, TR03-PAIR-003/005) detect CUDA automatically.

### Hugging Face models

Tasks built on Gemma-3-270M-it (TR02-SYC, TR03-PAIR) and Mistral-7B-v0.3 (TR03-PAIR-001) download gated
checkpoints, pinned by revision:

1. Accept the licences on huggingface.co:
   [Gemma-3-270M-it](https://huggingface.co/google/gemma-3-270m-it) and
   [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3).
2. Log in once with `hf auth login`. The token is stored under `~/.cache/huggingface`, outside the repository, and
   that directory is unreadable inside the sandbox.

An exported `HF_TOKEN` environment variable also works, but the agent process inherits the environment, so the agent
could read it. Prefer `hf auth login`.

### `constraints.txt`

`constraints.txt` pins the packages that decide numerical results or break on upgrade (numpy, torch, …). It is not a
lock file: packages it does not list resolve to their latest release. Use it for reproducible numbers. Every build
records the numeric stack (`numpy==…, torch==…`) in its `.built` marker, and building the same seed under a different
stack can give different weights.

### Plain pip

In any Python 3.11/3.12 environment:

```bash
pip install -r requirements.txt -c constraints.txt
pip install -e . -c constraints.txt
pip install -e '.[lm]' -c constraints.txt     # optional extras as above
```

For agent runs, also create the agent tool environment (`bash scripts/install.sh --agent` does it with uv), or point
`WB_AGENT_VENV` at a venv that contains numpy, scipy, torch and safetensors.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `WB_HARNESS_VENV` | `./.venv` | harness environment used by the scripts |
| `WB_AGENT_VENV` | `~/wb-agent-venv` | the agent's Python inside the sandbox |
| `WB_RESULTS_DIR` | `results` in the repo root | where `weightbench-report` writes `summary.json` and `RESULTS.md` |
| `WB_SANDBOX` | auto | force the sandbox backend: `macos` or `linux` |
| `WB_SANDBOX_HOME_ALLOW` | empty | macOS backend: extra home paths (separated by `:`) the agent may read. On Linux use `--agent-home` |
| `WB_AGENT_STALL_S` | `1800` | kill an agent whose transcript has not grown for this many seconds; `0` disables |
| `WB_SKIP_AGENT_PREFLIGHT` | unset | `1` skips the Claude login check before a run |
| `WB_HEAVY` | `local` | execution mode for build and evaluation steps; `local` is the only built-in mode |
| `WB_THREADS`, `WB_TORCH_THREADS` | small | thread counts for some builders |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Run this installer inside Ubuntu/WSL, not Windows Git Bash.` | Open Ubuntu (WSL) and run the scripts there. |
| `Install uv: …` | Install uv (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) and open a new shell. |
| `Incompatible environment: …` | The target venv is not Python 3.11/3.12. Choose a fresh path with `WB_HARNESS_VENV` or `WB_AGENT_VENV`. |
| `the linux sandbox needs bubblewrap` | `sudo apt install bubblewrap` (or `sudo dnf install bubblewrap`). |
| `bubblewrap cannot create a sandbox` / `bwrap` permission errors on Ubuntu 24.04+ | Set `kernel.apparmor_restrict_unprivileged_userns=0` (see above). |
| `doctor` fails an import | Reinstall with `bash scripts/install.sh` and check that the right venv is active. The `weightbench` console script runs the checkout it was installed from; use `python -m weightbench …` to run a different checkout. |
| `Agent runs require Ubuntu in WSL2 on Windows` | Install WSL2 and run inside Ubuntu. |
| Gated model download fails (401/403) | Accept the licence on huggingface.co and run `hf auth login`. |
| `run ID already exists` | Run ids are never reused. Pass a fresh `--sample` number (see [running-agents.md](running-agents.md)). |
| `[build] … sources changed since it was built; rebuilding` | Expected: an artifact is rebuilt whenever the code that generates it changes. |
| A task with a heavy model is very slow on CPU | TR02-LOOP and TR03-PAIR-003/005 need long generation; use a CUDA machine. |
