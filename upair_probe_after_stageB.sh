#!/usr/bin/env bash
# Probe that Stage-B Optuna produced fresh best-parameter artifacts for all variants.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/upair_submit_lib.sh"
cd "${UPAIR_REPO_ROOT}"

upair_ensure_venv
B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB}"
A_PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA}"
fail=0

echo "[PROBE] Checking Stage-B artifacts under ${UPAIR_REPO_ROOT}/optuna"
while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  a_db="optuna/${A_PREFIX}_${variant}.db"
  b_db="optuna/${B_PREFIX}_${variant}.db"
  b_json="optuna/${B_PREFIX}_${variant}_best_params.json"
  [[ -s "${a_db}" ]] || { echo "[FAIL] Missing Stage-A DB: ${a_db}" >&2; fail=1; }
  [[ -s "${b_db}" ]] || { echo "[FAIL] Missing Stage-B DB: ${b_db}" >&2; fail=1; }
  [[ -s "${b_json}" ]] || { echo "[FAIL] Missing Stage-B best JSON: ${b_json}" >&2; fail=1; }
  if [[ -s "${b_json}" ]]; then
    python - "${b_json}" "${variant}" <<'PY'
import json, sys
path, variant = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
params = payload.get("best_params", {})
required = {"learning_rate", "weight_decay", "nmse_loss_weight", "dropout", "residual_scale"}
missing = sorted(required - set(params))
if missing:
    raise SystemExit(f"[FAIL] {path}: missing best_params keys {missing}")
print(f"[OK] {variant}: best_value={payload.get('best_value')} completed={payload.get('completed_trials')} json={path}")
PY
  fi
done < <(upair_variants)

if [[ "${fail}" != "0" ]]; then
  echo "[PROBE] FAILED Stage-B probe" >&2
  exit 1
fi

echo "[PROBE] PASSED Stage-B Optuna probe"
echo "[PROBE] You can now submit final train/eval jobs with: bash upair_submit_train_eval_all.sh"
