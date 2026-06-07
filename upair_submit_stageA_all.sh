#!/usr/bin/env bash
# Submit Stage-A Optuna jobs for all seven 1-DMRS UPAIR variants.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/upair_submit_lib.sh"

cd "${UPAIR_REPO_ROOT}"
mkdir -p "${UPAIR_REPO_ROOT}/optuna" "${UPAIR_REPO_ROOT}/logs/optuna" "${UPAIR_REPO_ROOT}/logs/submit"

if [[ "${UPAIR_SKIP_SUBMIT_ENV_CHECK:-0}" != "1" ]]; then
  upair_ensure_venv
fi

CONFIG="${UPAIR_CONFIG:-${UPAIR_REPO_ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
[[ -f "${CONFIG}" ]] || { echo "[STAGE-A] Missing config: ${CONFIG}" >&2; exit 1; }

PREFIX="${UPAIR_OPTUNA_STAGEA_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageA}"
TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-24}"
STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"
MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-36}"
TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"
SEED="${UPAIR_SEED:-7}"
TRAIN_B="${UPAIR_TRAIN_BATCH:-32}"
VAL_B="${UPAIR_VAL_BATCH:-32}"
VAL_MB="${UPAIR_VAL_MICROBATCH:-16}"
TPE_STARTUP="${UPAIR_OPTUNA_TPE_STARTUP_TRIALS:-10}"
PRUNER_STARTUP="${UPAIR_OPTUNA_PRUNER_STARTUP_TRIALS:-8}"
PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_PRUNER_MIN_TRIALS:-5}"
OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_OBJECTIVE_RECENT_K:-3}"
OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_OBJECTIVE_MIN_STEP:-1000}"

echo "[STAGE-A] ROOT=${UPAIR_REPO_ROOT}"
echo "[STAGE-A] VENV=${UPAIR_VENV_PATH}"
echo "[STAGE-A] PREFIX=${PREFIX} TRIALS=${TRIALS} STEPS=${STEPS}"

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  study="${PREFIX}_${variant}"
  db="${UPAIR_REPO_ROOT}/optuna/${study}.db"
  log="${UPAIR_REPO_ROOT}/logs/optuna/${study}_%j.out"
  jobfile="${UPAIR_REPO_ROOT}/logs/submit/stageA_${variant}.sbatch"
  job="upairA-$(upair_first_n_chars "${variant}" 14)"
  upair_write_sbatch_header "${jobfile}" "${job}" "${TIME_LIMIT}" "${log}"
  cat >> "${jobfile}" <<SBATCH
set -euo pipefail
cd "${UPAIR_REPO_ROOT}"
source "${UPAIR_REPO_ROOT}/upair_portable_env.sh"
upair_activate
python -u "${UPAIR_REPO_ROOT}/scripts/run_optuna_1dmrs_structure_isolated.py" \
  --config "${CONFIG}" \
  --variant "${variant}" \
  --study-name "${study}" \
  --storage "sqlite:///${db}" \
  --stage A \
  --n-trials "${TRIALS}" \
  --target-total-trials "${TRIALS}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --steps "${STEPS}" \
  --train-batch-size "${TRAIN_B}" \
  --validation-batch-size "${VAL_B}" \
  --validation-microbatch-size "${VAL_MB}" \
  --seed "${SEED}" \
  --tpe-startup-trials "${TPE_STARTUP}" \
  --pruner-startup-trials "${PRUNER_STARTUP}" \
  --pruner-min-trials "${PRUNER_MIN_TRIALS}" \
  --objective-recent-k "${OBJECTIVE_RECENT_K}" \
  --objective-min-step "${OBJECTIVE_MIN_STEP}"
SBATCH
  echo "[STAGE-A] submitting ${variant} -> ${study}"
  upair_submit_job_script "${jobfile}"
done < <(upair_variants)
