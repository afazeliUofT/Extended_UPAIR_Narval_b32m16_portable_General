#!/usr/bin/env bash
# Probe that final training checkpoints and automatic evaluation outputs exist.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/upair_submit_lib.sh"
cd "${UPAIR_REPO_ROOT}"

fail=0
EVAL_USERS_RAW="${UPAIR_EVAL_USERS:-1,2,3,4}"
EVAL_USERS_SPACE="${EVAL_USERS_RAW//,/ }"
# shellcheck disable=SC2206
EVAL_USERS=( ${EVAL_USERS_SPACE} )
expected_eval_count="${#EVAL_USERS[@]}"

echo "[PROBE] Checking training/evaluation outputs under TWC_plots_comprehensive"
[[ -d TWC_plots_comprehensive ]] || { echo "[FAIL] Missing TWC_plots_comprehensive output root" >&2; exit 1; }

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  ckpt=$(find TWC_plots_comprehensive/runs_* -path "*/1dmrs/${variant}/checkpoints/best.weights.h5" -type f -print -quit 2>/dev/null || true)
  if [[ -z "${ckpt}" ]]; then
    echo "[FAIL] ${variant}: missing final best.weights.h5" >&2
    fail=1
  else
    echo "[OK] ${variant}: checkpoint ${ckpt}"
  fi

  state=$(find TWC_plots_comprehensive/runs_* -path "*/1dmrs/${variant}/metrics/train_state.json" -type f -print -quit 2>/dev/null || true)
  if [[ -z "${state}" ]]; then
    echo "[FAIL] ${variant}: missing train_state.json" >&2
    fail=1
  else
    python - "${state}" "${variant}" <<'PY'
import json, sys
path, variant = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
if not payload.get("training_complete", False):
    raise SystemExit(f"[FAIL] {variant}: training_complete is false in {path}")
print(f"[OK] {variant}: training_complete latest_step={payload.get('latest_step')} total_steps={payload.get('total_steps')}")
PY
  fi

  for u in "${EVAL_USERS[@]}"; do
    summary=$(find TWC_plots_comprehensive/eval_runs_* -path "*/1dmrs/${variant}_u${u}/metrics/evaluation_summary.json" -type f -print -quit 2>/dev/null || true)
    curves=$(find TWC_plots_comprehensive/csv_* -path "*/1dmrs/${variant}_u${u}_curves.csv" -type f -print -quit 2>/dev/null || true)
    if [[ -z "${summary}" ]]; then
      echo "[FAIL] ${variant}: missing evaluation_summary.json for u${u}" >&2
      fail=1
    else
      python - "${summary}" "${variant}" "${u}" <<'PY'
import json, sys
path, variant, u = sys.argv[1:4]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
if payload.get("num_users") is not None and int(payload.get("num_users")) != int(u):
    raise SystemExit(f"[FAIL] {variant}/u{u}: num_users mismatch in {path}")
print(f"[OK] {variant}/u{u}: evaluation summary {path}")
PY
    fi
    if [[ -z "${curves}" ]]; then
      echo "[FAIL] ${variant}: missing copied curves CSV for u${u}" >&2
      fail=1
    else
      echo "[OK] ${variant}/u${u}: curves ${curves}"
    fi
  done
done < <(upair_variants)

if [[ "${fail}" != "0" ]]; then
  echo "[PROBE] FAILED train/eval probe" >&2
  exit 1
fi

echo "[PROBE] PASSED train/eval output probe for ${expected_eval_count} evaluation user-count(s) per variant"
