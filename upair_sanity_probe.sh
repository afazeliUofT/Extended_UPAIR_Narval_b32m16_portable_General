#!/usr/bin/env bash
# Drop-in sanity probe launcher for the UMi pipeline.
# Usage:  bash upair_sanity_probe.sh
# Output: prints to console AND saves to upair_sanity_probe_output.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Match the convention used by the other probe scripts in this repo.
source "${ROOT}/upair_portable_env.sh"
upair_activate

OUT="${ROOT}/upair_sanity_probe_output.txt"
echo "[PROBE] Writing combined output to: ${OUT}"
# Force single-GPU / deterministic-ish behaviour for a clean probe; harmless if unset.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1

# Run; tee everything (stdout+stderr) so nothing is lost.
python "${ROOT}/upair_sanity_probe.py" 2>&1 | tee "${OUT}"

echo ""
echo "[PROBE] Done. Please send back the full contents of:"
echo "        ${OUT}"
