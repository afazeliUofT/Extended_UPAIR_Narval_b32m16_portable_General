#!/usr/bin/env bash
# Tiny end-to-end smoke test under standard normalized UMi.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source "${ROOT}/upair_portable_env.sh"
upair_activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

VARIANT="${UPAIR_SMOKE_VARIANT:-main_d256_b4_r2}"
SMOKE_ROOT="${UPAIR_SMOKE_ROOT:-${ROOT}/_smoke_umi_norm_runtime}"
SMOKE_TAG="${UPAIR_SMOKE_TAG:-smoke_umi_norm_$(date +%Y%m%d_%H%M%S)}"
A_PREFIX="${SMOKE_TAG}_stageA"
B_PREFIX="${SMOKE_TAG}_stageB"
CONFIG="${SMOKE_ROOT}/smoke_umi_norm_config.yaml"

rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}/optuna"

python - "${CONFIG}" <<'PY'
from pathlib import Path
import sys, yaml
out = Path(sys.argv[1])
cfg = yaml.safe_load(open("configs/twc_comprehensive_mu32_base.yaml", "r", encoding="utf-8"))
def setp(path, value):
    node = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value

updates = {
    "system.batch_size_train": 2,
    "system.batch_size_eval": 2,
    "system.ebno_db_eval": [0.0],
    "training.steps": 2,
    "training.eval_every": 1,
    "training.checkpoint_every": 1,
    "training.log_every": 1,
    "training.val_steps": 1,
    "training.val_ebno_db": [0.0],
    "training.val_user_counts": [4],
    "training.val_user_count_weights": [1.0],
    "training.val_microbatch_size": 1,
    "evaluation.logical_batch_size": 2,
    "evaluation.receiver_microbatch_size": 1,
    "evaluation.min_num_batches_per_point": 1,
    "evaluation.max_num_batches_per_point": 1,
    "evaluation.num_batches_per_point": 1,
    "evaluation.target_block_errors_per_receiver": 0,
    "evaluation.progress_every_batches": 1,
    "evaluation.nmse_receivers": [],
    "evaluation.save_example_batch": False,
    "evaluation.resume": False,
    "baselines.covariance_estimation.reuse_cache": False,
    "baselines.covariance_estimation.num_batches": 1,
    "baselines.covariance_estimation.batch_size": 2,
    "baselines.covariance_estimation.cache_name": "smoke_umi_norm_cov.npz",
}
for path, value in updates.items():
    setp(path, value)
out.parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"), sort_keys=False)
print("[SMOKE] wrote", out)
PY

python -u scripts/run_optuna_1dmrs_structure_isolated.py \
  --config "${CONFIG}" --variant "${VARIANT}" \
  --study-name "${A_PREFIX}_${VARIANT}" \
  --storage "sqlite:///${SMOKE_ROOT}/optuna/${A_PREFIX}_${VARIANT}.db" \
  --stage A --n-trials 1 --target-total-trials 1 --max-attempts 1 \
  --steps 2 --eval-every 1 --checkpoint-every 1 --log-every 1 \
  --val-steps 1 --val-ebno-db 0 --val-user-counts 4 --val-user-count-weights 1 \
  --train-user-count-weights 0,0,0,1 --train-batch-size 2 \
  --validation-batch-size 2 --validation-microbatch-size 1 --objective-min-step 0 --smoke

python -u scripts/run_optuna_1dmrs_structure_isolated.py \
  --config "${CONFIG}" --variant "${VARIANT}" \
  --study-name "${B_PREFIX}_${VARIANT}" \
  --storage "sqlite:///${SMOKE_ROOT}/optuna/${B_PREFIX}_${VARIANT}.db" \
  --stage B --source-study-name "${A_PREFIX}_${VARIANT}" \
  --source-storage "sqlite:///${SMOKE_ROOT}/optuna/${A_PREFIX}_${VARIANT}.db" \
  --source-top-k 1 --n-trials 1 --target-total-trials 1 --max-attempts 1 \
  --steps 2 --eval-every 1 --checkpoint-every 1 --log-every 1 \
  --val-steps 1 --val-ebno-db 0 --val-user-counts 4 --val-user-count-weights 1 \
  --train-user-count-weights 0,0,0,1 --train-batch-size 2 \
  --validation-batch-size 2 --validation-microbatch-size 1 --objective-min-step 0 --smoke

python -u scripts/run_comprehensive_mu32_ablation.py \
  --config "${CONFIG}" --variants "${VARIANT}" --dmrs-cases 1dmrs \
  --seeds 7 --eval-users 4 --use-optuna-best-1dmrs \
  --optuna-best-storage-dir "${SMOKE_ROOT}/optuna" \
  --optuna-best-study-prefix "${B_PREFIX}" --require-optuna-best \
  --no-global-summary --force

rm -rf "TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/${VARIANT}" \
       "TWC_plots_comprehensive/eval_runs_rx16/seed7/1dmrs/${VARIANT}_u4"
rm -f "TWC_plots_comprehensive/csv_rx16/seed7/1dmrs/${VARIANT}_u4_curves.csv"

echo "[SMOKE] PASSED normalized UMi full-pipeline smoke test."
