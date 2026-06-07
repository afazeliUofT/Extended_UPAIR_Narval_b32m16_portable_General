#!/usr/bin/env bash
# Permanent UMiPC Optuna defaults + prompt-MLP-ratio ablation.
# Intended for /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable_General
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_PATCH:-0}" != "1" ]]; then
  echo "[PATCH] Refusing to patch outside the General repo copy." >&2
  echo "[PATCH] ROOT=${ROOT}" >&2
  exit 1
fi

[[ -f configs/twc_comprehensive_mu32_base.yaml && -f src/upair5g/estimator.py && -f scripts/optuna_1dmrs_common.py ]] || {
  echo "[PATCH] Run from repo root." >&2
  exit 1
}

python - <<'PY'
from __future__ import annotations
from pathlib import Path
import re
import yaml

PROMPT_VARIANT = "promptmlp_d256_b4_r2_pr2"

# ---------------------------------------------------------------------
# 1) Base config: prompt MLP ratio defaults to 1.0.
# ---------------------------------------------------------------------
cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("model", {})["prompt_mlp_ratio"] = 1.0
if cfg.get("near_far", {}).get("enabled", False):
    cfg.setdefault("baselines", {}).setdefault("covariance_estimation", {})["cache_name"] = "empirical_covariances_umi_pc_meanref.npz"
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[PATCH] config: model.prompt_mlp_ratio=1.0")

# ---------------------------------------------------------------------
# 2) estimator.py: make prompt MLP hidden size configurable.
# ---------------------------------------------------------------------
p = Path("src/upair5g/estimator.py")
s = p.read_text()

if "self.prompt_mlp_ratio" not in s:
    old = '''        self.d_model = int(cfg["model"]["d_model"])
        model_cfg = cfg.get("model", {})
'''
    new = '''        self.d_model = int(cfg["model"]["d_model"])
        model_cfg = cfg.get("model", {})
        self.prompt_mlp_ratio = float(model_cfg.get("prompt_mlp_ratio", 1.0))
        self.prompt_mlp_hidden_dim = max(1, int(round(self.d_model * self.prompt_mlp_ratio)))
'''
    if old not in s:
        raise SystemExit("[PATCH] Could not find estimator d_model/model_cfg block.")
    s = s.replace(old, new, 1)
else:
    print("[PATCH] estimator: prompt_mlp_ratio already present")

if "Dense(self.prompt_mlp_hidden_dim, activation=\"gelu\")" not in s:
    old = 'tf.keras.layers.Dense(self.d_model, activation="gelu"),'
    new = 'tf.keras.layers.Dense(self.prompt_mlp_hidden_dim, activation="gelu"),'
    if old not in s:
        raise SystemExit("[PATCH] Could not find first prompt Dense(self.d_model, activation='gelu').")
    s = s.replace(old, new, 1)

p.write_text(s)
print("[PATCH] estimator: prompt MLP is now d -> round(prompt_mlp_ratio*d) -> d")

# ---------------------------------------------------------------------
# 3) Optuna common: add prompt ablation and permanent stage defaults.
# ---------------------------------------------------------------------
p = Path("scripts/optuna_1dmrs_common.py")
s = p.read_text()

if f'"{PROMPT_VARIANT}"' not in s:
    anchor = '''    "main_d256_b4_r2": {"model.d_model": 256, "model.num_blocks": 4, "model.mlp_ratio": 2.0},
'''
    insert = anchor + '''    "promptmlp_d256_b4_r2_pr2": {"model.d_model": 256, "model.num_blocks": 4, "model.mlp_ratio": 2.0, "model.prompt_mlp_ratio": 2.0},
'''
    if anchor not in s:
        raise SystemExit("[PATCH] Could not find main variant anchor in optuna_1dmrs_common.py.")
    s = s.replace(anchor, insert, 1)

s = re.sub(
    r'"A": \{"steps": \d+, "target_total_trials": \d+, "source_top_k": 0\}',
    '"A": {"steps": 4000, "target_total_trials": 24, "source_top_k": 0}',
    s,
)
s = re.sub(
    r'"B": \{"steps": \d+, "target_total_trials": \d+, "source_top_k": \d+\}',
    '"B": {"steps": 12000, "target_total_trials": 10, "source_top_k": 8}',
    s,
)

p.write_text(s)
print("[PATCH] optuna common: added prompt variant; Stage A=24, Stage B=10/top8")

# ---------------------------------------------------------------------
# 4) Final train/eval script variants: add same prompt ablation.
# ---------------------------------------------------------------------
p = Path("scripts/run_comprehensive_mu32_ablation.py")
s = p.read_text()

if f'"{PROMPT_VARIANT}"' not in s:
    anchor = '''    "main_d256_b4_r2": {
        "label": "d=256, L=4, r=2",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
        },
    },
'''
    insert = anchor + '''    "promptmlp_d256_b4_r2_pr2": {
        "label": "d=256, L=4, r=2, prompt-r=2",
        "overrides": {
            "model.d_model": 256,
            "model.num_blocks": 4,
            "model.mlp_ratio": 2.0,
            "model.prompt_mlp_ratio": 2.0,
        },
    },
'''
    if anchor not in s:
        raise SystemExit("[PATCH] Could not find main variant anchor in run_comprehensive_mu32_ablation.py.")
    s = s.replace(anchor, insert, 1)

p.write_text(s)
print("[PATCH] comprehensive train/eval variants: added prompt MLP ablation")

# ---------------------------------------------------------------------
# 5) Slurm variant list: add prompt ablation to default sweep.
# ---------------------------------------------------------------------
p = Path("upair_submit_lib.sh")
s = p.read_text()

if f"  {PROMPT_VARIANT}\n" not in s:
    anchor = "  main_d256_b4_r2\n"
    if anchor not in s:
        raise SystemExit("[PATCH] Could not find main variant in upair_submit_lib.sh.")
    s = s.replace(anchor, anchor + f"  {PROMPT_VARIANT}\n", 1)

p.write_text(s)
print("[PATCH] submit-lib default variant list: added prompt MLP ablation")

# ---------------------------------------------------------------------
# 6) Slurm wrappers: make strengthened UMiPC Optuna defaults permanent.
# ---------------------------------------------------------------------
def replace_line(text: str, prefix: str, newline: str) -> str:
    pattern = re.compile(rf'^{re.escape(prefix)}.*$', re.M)
    if not pattern.search(text):
        raise SystemExit(f"[PATCH] Could not find line starting with {prefix!r}")
    return pattern.sub(newline, text, count=1)

p = Path("upair_submit_stageA_all.sh")
s = p.read_text()
s = replace_line(s, 'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-', 'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-24}"')
s = replace_line(s, 'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-', 'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"')
s = replace_line(s, 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-36}"')
s = replace_line(s, 'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-', 'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"')
s = replace_line(s, 'TPE_STARTUP="${UPAIR_OPTUNA_TPE_STARTUP_TRIALS:-', 'TPE_STARTUP="${UPAIR_OPTUNA_TPE_STARTUP_TRIALS:-10}"')
s = replace_line(s, 'PRUNER_STARTUP="${UPAIR_OPTUNA_PRUNER_STARTUP_TRIALS:-', 'PRUNER_STARTUP="${UPAIR_OPTUNA_PRUNER_STARTUP_TRIALS:-8}"')
s = replace_line(s, 'PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_PRUNER_MIN_TRIALS:-', 'PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_PRUNER_MIN_TRIALS:-5}"')
s = replace_line(s, 'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_OBJECTIVE_RECENT_K:-', 'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_OBJECTIVE_RECENT_K:-3}"')
s = replace_line(s, 'OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_OBJECTIVE_MIN_STEP:-', 'OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_OBJECTIVE_MIN_STEP:-1000}"')
p.write_text(s)

p = Path("upair_submit_stageB_all.sh")
s = p.read_text()
s = replace_line(s, 'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-', 'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-10}"')
s = replace_line(s, 'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-', 'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-12000}"')
s = replace_line(s, 'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-', 'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-8}"')
s = replace_line(s, 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-', 'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-16}"')
s = replace_line(s, 'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-', 'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-30:00:00}"')
s = replace_line(s, 'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_RECENT_K:-', 'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_RECENT_K:-3}"')
s = replace_line(s, 'OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_MIN_STEP:-', 'OBJECTIVE_MIN_STEP="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_MIN_STEP:-2000}"')
p.write_text(s)

print("[PATCH] wrappers: permanent Stage A/B defaults set")

# ---------------------------------------------------------------------
# 7) Ignore smoke/runtime outputs.
# ---------------------------------------------------------------------
p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = [
    "optuna/", "logs/", "TWC_plots_comprehensive/", "_smoke_*/", "patch_backups/",
    "*.out", "*.err", "*.log", "*.db", "*.sqlite", "*.sqlite3", "__pycache__/", "*.py[cod]",
]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")

print("[PATCH] Done.")
PY

echo "[PATCH] Next run:"
echo "  bash upair_probe_permanent_optuna_prompt_ablation.sh"
