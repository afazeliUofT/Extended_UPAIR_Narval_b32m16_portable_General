#!/usr/bin/env bash
# Smoke test after true-DMRS-mask patch:
# one Stage-A trial, one Stage-B promoted trial, and final train/eval.
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
SMOKE_ROOT="${UPAIR_SMOKE_ROOT:-${ROOT}/_smoke_true_dmrs_runtime}"
SMOKE_TAG="${UPAIR_SMOKE_TAG:-smoke_true_dmrs_$(date +%Y%m%d_%H%M%S)}"
A_PREFIX="${SMOKE_TAG}_stageA"
B_PREFIX="${SMOKE_TAG}_stageB"
CONFIG="${SMOKE_ROOT}/smoke_config.yaml"

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
setp("system.batch_size_train", 2)
setp("system.batch_size_eval", 2)
setp("system.ebno_db_eval", [0.0])
setp("training.steps", 2)
setp("training.eval_every", 1)
setp("training.checkpoint_every", 1)
setp("training.log_every", 1)
setp("training.val_steps", 1)
setp("training.val_ebno_db", [0.0])
setp("training.val_user_counts", [4])
setp("training.val_user_count_weights", [1.0])
setp("training.val_microbatch_size", 1)
setp("evaluation.logical_batch_size", 2)
setp("evaluation.receiver_microbatch_size", 1)
setp("evaluation.min_num_batches_per_point", 1)
setp("evaluation.max_num_batches_per_point", 1)
setp("evaluation.num_batches_per_point", 1)
setp("evaluation.target_block_errors_per_receiver", 0)
setp("evaluation.progress_every_batches", 1)
setp("evaluation.nmse_receivers", [])
setp("evaluation.save_example_batch", False)
setp("evaluation.resume", False)
setp("baselines.covariance_estimation.reuse_cache", False)
setp("baselines.covariance_estimation.num_batches", 1)
setp("baselines.covariance_estimation.batch_size", 2)
setp("baselines.covariance_estimation.cache_name", "smoke_true_dmrs_cov.npz")
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

echo "[SMOKE] PASSED true-DMRS full-pipeline smoke test."
