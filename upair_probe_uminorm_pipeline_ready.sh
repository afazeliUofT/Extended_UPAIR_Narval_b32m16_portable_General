#!/usr/bin/env bash
# Read-only sanity probe. Does NOT modify .gitignore.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

fail=0
ok(){ echo "[OK] $*"; }
bad(){ echo "[FAIL] $*" >&2; fail=1; }
warn(){ echo "[WARN] $*" >&2; }

for f in \
  configs/twc_comprehensive_mu32_base.yaml \
  src/upair5g/builders.py \
  scripts/run_comprehensive_mu32_ablation.py \
  scripts/run_isolated_eval_chunk.py \
  scripts/merge_isolated_eval_chunks.py \
  scripts/isolated_eval_status.py \
  upair_portable_env.sh \
  upair_submit_lib.sh \
  upair_submit_stageA_all.sh \
  upair_submit_stageB_all.sh \
  upair_variant_pipeline_worker.sh \
  upair_submit_8variant_pipeline.sh
do
  [[ -e "$f" ]] && ok "$f exists" || bad "$f missing"
done

python - <<'PY'
from pathlib import Path
import ast, json, sys, yaml

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True
def warn(msg): print("[WARN]", msg, file=sys.stderr)

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
def get(path, default=None):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

checks = [
    ("channel.family", "umi"),
    ("channel.model", "umi"),
    ("channel.normalize_channel", True),
    ("channel.umi.enable_pathloss", False),
    ("channel.umi.enable_shadow_fading", False),
    ("near_far.enabled", False),
    ("near_far.mode", "disabled"),
    ("pusch.n_size_grid", 8),
    ("pusch.n_size_bwp", 8),
    ("model.pilot_mask_mode", "per_stream"),
    ("model.error_feature_mode", "per_user"),
    ("system.batch_size_train", 32),
    ("system.batch_size_eval", 32),
    ("training.val_microbatch_size", 16),
    ("evaluation.logical_batch_size", 64),
    ("evaluation.receiver_microbatch_size", 8),
    ("evaluation.compiled_receiver_error_counts", False),
    ("evaluation.receiver_call_jit_compile", False),
    ("evaluation.stream_eval_microbatches", True),
    ("evaluation.nmse_receivers", []),
    ("evaluation.save_example_batch", False),
    ("evaluation.max_num_batches_per_point", 2000),
    ("evaluation.target_block_errors_per_receiver", 100),
    ("baselines.covariance_estimation.cache_name", "empirical_covariances_umi_norm.npz"),
]
for path, want in checks:
    got = get(path)
    if got == want:
        ok(f"{path} = {got!r}")
    else:
        bad(f"{path}: expected {want!r}, got {got!r}")

for py in ["scripts/run_isolated_eval_chunk.py", "scripts/merge_isolated_eval_chunks.py", "scripts/isolated_eval_status.py"]:
    ast.parse(Path(py).read_text())
    ok(f"syntax parse: {py}")

rc = Path("scripts/run_comprehensive_mu32_ablation.py").read_text()
if 'evaluation.save_example_batch", bool(get_cfg(cfg, "evaluation.save_example_batch", False))' in rc:
    ok("_eval_cfg preserves save_example_batch=False from config")
else:
    bad("_eval_cfg may still force save_example_batch=True for main/u4")

if "UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1" in Path("upair_variant_pipeline_worker.sh").read_text():
    ok("variant worker uses training-only mode before isolated eval")
else:
    bad("variant worker does not force training-only mode")

if "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB" in Path("upair_submit_8variant_pipeline.sh").read_text():
    ok("master pipeline uses UMiNorm Stage-B prefix")
else:
    bad("master pipeline Stage-B prefix is not UMiNorm")

# Check variants from optuna_common and submit_lib.
ns = {"__file__": str(Path("scripts/optuna_1dmrs_common.py").resolve())}
exec(compile(Path("scripts/optuna_1dmrs_common.py").read_text(), "scripts/optuna_1dmrs_common.py", "exec"), ns)
variants = list(ns["VARIANTS"].keys())
print("[INFO] optuna variants:", ",".join(variants))
if len(variants) == 8 and "promptmlp_d256_b4_r2_pr2" in variants:
    ok("8 variants including prompt-MLP ablation are present")
else:
    bad("expected 8 variants including prompt-MLP ablation")

# Stage-B best is optional before Optuna has run. Make it fatal only if requested.
require_stageb = str(__import__("os").environ.get("UPAIR_REQUIRE_STAGEB", "0")).lower() in {"1","true","yes"}
missing = []
for v in variants:
    json_path = Path(f"optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB_{v}_best_params.json")
    db_path = Path(f"optuna/clean_b32_prb8_d256_40k_smart_trueDMRS_UMiNorm_u34610_1dmrs_stageB_{v}.db")
    if json_path.exists() or db_path.exists():
        ok(f"Stage-B best exists for {v}")
    else:
        missing.append(v)
        warn(f"Stage-B best missing for {v}; normal before Stage A/B has completed")
if missing and require_stageb:
    bad("Stage-B best is required but missing for: " + ",".join(missing))

if fail:
    raise SystemExit("[PROBE] FAILED normalized-UMi pipeline readiness probe")
print("[PROBE] PASSED normalized-UMi pipeline readiness probe")
PY

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

[[ "$fail" == "0" ]] || exit 1
