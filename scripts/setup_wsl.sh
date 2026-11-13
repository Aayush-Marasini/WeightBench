#!/usr/bin/env bash
# Linux/WSL sandbox setup; the harness and agent tools use separate environments.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/install.sh --agent "$@"
