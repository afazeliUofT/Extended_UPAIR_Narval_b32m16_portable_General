#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate
OUT="${ROOT}/upair_pc_verify_output.txt"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1
echo "[PROBE] writing to ${OUT}"
python "${ROOT}/upair_pc_verify_probe.py" 2>&1 | tee "${OUT}"
echo ""
echo "[PROBE] Done. Send back: ${OUT}"
