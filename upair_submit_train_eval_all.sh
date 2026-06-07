#!/usr/bin/env bash
# Submit final training jobs; each job automatically starts evaluation after training.
# This wrapper deliberately requires fresh external Stage-B Optuna best JSON/DBs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/upair_submit_lib.sh"

cd "${UPAIR_REPO_ROOT}"
mkdir -p "${UPAIR_REPO_ROOT}/logs/train_eval" "${UPAIR_REPO_ROOT}/logs/submit"

if [[ "${UPAIR_SKIP_SUBMIT_ENV_CHECK:-0}" != "1" ]]; then
  upair_ensure_venv
fi

CONFIG="${UPAIR_CONFIG:-${UPAIR_REPO_ROOT}/configs/twc_comprehensive_mu32_base.yaml}"
[[ -f "${CONFIG}" ]] || { echo "[TRAIN-EVAL] Missing config: ${CONFIG}" >&2; exit 1; }

B_PREFIX="${UPAIR_OPTUNA_STAGEB_PREFIX:-clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageB}"
DMRS_CASES="${UPAIR_DMRS_CASES:-1dmrs}"
SEEDS="${UPAIR_SEEDS:-${UPAIR_SEED:-7}}"
EVAL_USERS="${UPAIR_EVAL_USERS:-1,2,3,4}"
TIME_LIMIT="${UPAIR_TIME_TRAIN_EVAL:-30:00:00}"
FORCE_FLAG=""
if [[ "${UPAIR_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

echo "[TRAIN-EVAL] ROOT=${UPAIR_REPO_ROOT}"
echo "[TRAIN-EVAL] VENV=${UPAIR_VENV_PATH}"
echo "[TRAIN-EVAL] Optuna best prefix required: ${B_PREFIX}"
echo "[TRAIN-EVAL] DMRS_CASES=${DMRS_CASES} SEEDS=${SEEDS} EVAL_USERS=${EVAL_USERS}"

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  best_json="${UPAIR_REPO_ROOT}/optuna/${B_PREFIX}_${variant}_best_params.json"
  best_db="${UPAIR_REPO_ROOT}/optuna/${B_PREFIX}_${variant}.db"
  if [[ ! -s "${best_json}" && ! -s "${best_db}" ]]; then
    echo "[TRAIN-EVAL] Missing fresh Stage-B best JSON/DB for ${variant}." >&2
    echo "[TRAIN-EVAL] Expected one of:" >&2
    echo "  ${best_json}" >&2
    echo "  ${best_db}" >&2
    exit 1
  fi

  log="${UPAIR_REPO_ROOT}/logs/train_eval/train_eval_${variant}_%j.out"
  jobfile="${UPAIR_REPO_ROOT}/logs/submit/train_eval_${variant}.sbatch"
  job="upairTE-$(upair_first_n_chars "${variant}" 13)"
  upair_write_sbatch_header "${jobfile}" "${job}" "${TIME_LIMIT}" "${log}"
  cat >> "${jobfile}" <<SBATCH
set -euo pipefail
cd "${UPAIR_REPO_ROOT}"
source "${UPAIR_REPO_ROOT}/upair_portable_env.sh"
upair_activate
python -u "${UPAIR_REPO_ROOT}/scripts/run_comprehensive_mu32_ablation.py" \
  --config "${CONFIG}" \
  --variants "${variant}" \
  --dmrs-cases "${DMRS_CASES}" \
  --seeds "${SEEDS}" \
  --eval-users "${EVAL_USERS}" \
  --use-optuna-best-1dmrs \
  --optuna-best-storage-dir "${UPAIR_REPO_ROOT}/optuna" \
  --optuna-best-study-prefix "${B_PREFIX}" \
  --require-optuna-best \
  --no-global-summary \
  ${FORCE_FLAG}
SBATCH
  echo "[TRAIN-EVAL] submitting ${variant}; evaluation will start automatically after training"
  upair_submit_job_script "${jobfile}"
done < <(upair_variants)
