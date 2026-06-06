#!/usr/bin/env bash
# Sanity probe for 40k final training + smart bounded Optuna defaults.
# Run from the repository root.
set -euo pipefail

ROOT="$(pwd)"
if [[ ! -d "${ROOT}/src" || ! -d "${ROOT}/scripts" || ! -d "${ROOT}/configs" ]]; then
  echo "[PROBE] Please run this from the UPAIR repository root." >&2
  exit 1
fi

fail=0
ok() { echo "[OK] $*"; }
bad() { echo "[FAIL] $*" >&2; fail=1; }
warn() { echo "[WARN] $*"; }

python - <<'PY'
from pathlib import Path
import re
import sys
import yaml

fail = False

def ok(msg): print(f"[OK] {msg}")
def bad(msg):
    global fail
    print(f"[FAIL] {msg}", file=sys.stderr)
    fail = True

root = Path.cwd()

def text(path): return (root/path).read_text(encoding="utf-8")

cfg = yaml.safe_load(text("configs/twc_comprehensive_mu32_base.yaml"))
checks = [
    (cfg["training"]["steps"] == 40000, "training.steps is 40000"),
    (cfg["training"]["learning_rate_decay_steps"] == 40000, "training.learning_rate_decay_steps is 40000"),
    (cfg["training"]["eval_every"] == 2000, "training.eval_every is 2000 for 40k final training"),
    (cfg["training"]["checkpoint_every"] == 1000, "training.checkpoint_every is 1000"),
    (cfg["pusch"]["n_size_grid"] == 8, "pusch.n_size_grid remains 8"),
    (cfg["pusch"]["n_size_bwp"] == 8, "pusch.n_size_bwp remains 8"),
    (cfg["model"]["d_model"] == 256, "base model.d_model remains 256"),
    (cfg["model"].get("pilot_mask_mode") == "per_stream", "pilot_mask_mode remains per_stream"),
    (cfg["model"].get("error_feature_mode") == "per_user", "error_feature_mode remains per_user"),
    (cfg["model"].get("error_variance_update") == "multiplicative", "error_variance_update remains multiplicative"),
    (list(map(float, cfg["training"]["val_user_count_weights"])) == [1.0,3.0,6.0,10.0], "validation user-count weights remain [1,3,6,10]"),
]
for cond, msg in checks:
    ok(msg) if cond else bad(msg)

common = text("scripts/optuna_1dmrs_common.py")
for pattern, msg in [
    (r'"A": \{"steps": 4000, "target_total_trials": 20, "source_top_k": 0\}', "Stage A defaults are 20 trials x 4000 steps"),
    (r'"B": \{"steps": 12000, "target_total_trials": 6, "source_top_k": 6\}', "Stage B defaults are 6 promoted trials x 12000 steps"),
    (r'"C": \{"steps": 40000, "target_total_trials": 3, "source_top_k": 3\}', "Stage C fallback defaults match 40k final scale"),
    (r'--tpe-startup-trials", type=int, default=8', "TPE startup default is 8"),
    (r'--pruner-startup-trials", type=int, default=6', "pruner startup default is 6"),
    (r'--pruner-min-trials", type=int, default=4', "pruner min trials default is 4"),
]:
    ok(msg) if re.search(pattern, common) else bad(msg)

stage_a = text("upair_submit_stageA_all.sh")
for snippet, msg in [
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA', "Stage-A wrapper uses new smart 40k prefix"),
    ('TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-20}"', "Stage-A wrapper defaults to 20 trials"),
    ('STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"', "Stage-A wrapper defaults to 4000 steps"),
    ('TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"', "Stage-A wrapper default walltime is 30h"),
    ('--tpe-startup-trials "${TPE_STARTUP}"', "Stage-A wrapper passes TPE startup"),
    ('--pruner-startup-trials "${PRUNER_STARTUP}"', "Stage-A wrapper passes pruner startup"),
]:
    ok(msg) if snippet in stage_a else bad(msg)

stage_b = text("upair_submit_stageB_all.sh")
for snippet, msg in [
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA', "Stage-B wrapper reads new Stage-A source prefix"),
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB', "Stage-B wrapper uses new smart 40k target prefix"),
    ('TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-6}"', "Stage-B wrapper defaults to 6 trials"),
    ('STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-12000}"', "Stage-B wrapper defaults to 12000 steps"),
    ('SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-6}"', "Stage-B wrapper promotes top 6 from Stage A"),
    ('TIME_LIMIT="${UPAIR_TIME_STAGE_B:-30:00:00}"', "Stage-B wrapper default walltime is 30h"),
]:
    ok(msg) if snippet in stage_b else bad(msg)

train_eval = text("upair_submit_train_eval_all.sh")
for snippet, msg in [
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB', "train/eval wrapper requires new Stage-B prefix"),
    ('TIME_LIMIT="${UPAIR_TIME_TRAIN_EVAL:-30:00:00}"', "train/eval wrapper default walltime is 30h"),
    ('--require-optuna-best', "train/eval still requires fresh external Optuna best"),
]:
    ok(msg) if snippet in train_eval else bad(msg)

stageb_probe = text("upair_probe_after_stageB.sh")
for snippet, msg in [
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA', "Stage-B probe checks new Stage-A prefix"),
    ('clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB', "Stage-B probe checks new Stage-B prefix"),
]:
    ok(msg) if snippet in stageb_probe else bad(msg)

gi = text(".gitignore")
for entry in ["TWC_plots_comprehensive/", "logs/", "outputs/", "plots/", "metrics/", "*.out", "*.err", "*.log", "patch_backups/"]:
    ok(f".gitignore contains {entry}") if entry in gi.splitlines() else bad(f".gitignore missing {entry}")

clean_probe = text("upair_probe_clean_start.sh")
ok("clean-start probe does not call compileall") if "compileall" not in clean_probe else bad("clean-start probe still calls compileall")

raise SystemExit(1 if fail else 0)
PY

if [[ -d patch_backups ]]; then
  bad "patch_backups/ still exists; it is safe to remove with: rm -rf patch_backups"
else
  ok "patch_backups/ is absent"
fi

if [[ "${fail}" != "0" ]]; then
  echo "[PROBE] FAILED smart Optuna/40k probe" >&2
  exit 1
fi

echo "[PROBE] PASSED smart Optuna/40k probe"
