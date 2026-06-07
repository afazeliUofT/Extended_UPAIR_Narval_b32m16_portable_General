#!/usr/bin/env bash
# Sanity probe for permanent UMiPC Optuna defaults and prompt-MLP-ratio ablation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/upair_portable_env.sh"
upair_activate

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import yaml

import tensorflow as tf

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())

# Config checks
if float(cfg["model"].get("prompt_mlp_ratio", -1)) == 1.0:
    ok("base model.prompt_mlp_ratio is 1.0")
else:
    bad(f"base model.prompt_mlp_ratio is {cfg['model'].get('prompt_mlp_ratio')}")

# UMiPC checks if near_far exists.
if cfg.get("near_far", {}).get("enabled", False):
    nf = cfg["near_far"]
    ch = cfg["channel"]
    umi = ch["umi"]
    if ch.get("normalize_channel") is False and umi.get("enable_pathloss") is True and umi.get("enable_shadow_fading") is True:
        ok("UMiPC channel settings preserved: pathloss/shadow on, normalize_channel=false")
    else:
        bad("UMiPC channel settings are not preserved")
    if 0.65 <= float(nf.get("alpha_train_min", -1)) <= float(nf.get("alpha_train_max", -1)) <= 1.0:
        ok(f"UMiPC alpha train range preserved: [{nf.get('alpha_train_min')}, {nf.get('alpha_train_max')}]")
    else:
        bad(f"unexpected alpha train range: {nf}")

# Import Optuna common without importing TensorFlow-heavy runners.
spec = importlib.util.spec_from_file_location("optuna_common_probe", "scripts/optuna_1dmrs_common.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

variants = dict(mod.VARIANTS)
expected = {
    "main_d256_b4_r2",
    "promptmlp_d256_b4_r2_pr2",
    "shallow_d256_b2_r2",
    "deep_d256_b6_r2",
    "narrow_d192_b4_r2",
    "wide_d320_b4_r2",
    "wide_deep_d320_b6_r2",
    "mlpwide_d256_b4_r4",
}
if set(variants) == expected:
    ok("Optuna VARIANTS contains exactly the expected 8 structures")
else:
    bad(f"Optuna VARIANTS mismatch: {sorted(variants)}")

pv = variants.get("promptmlp_d256_b4_r2_pr2", {})
if pv == {"model.d_model": 256, "model.num_blocks": 4, "model.mlp_ratio": 2.0, "model.prompt_mlp_ratio": 2.0}:
    ok("promptmlp variant differs from main only by model.prompt_mlp_ratio=2.0")
else:
    bad(f"unexpected promptmlp variant overrides: {pv}")

if mod.STAGE_DEFAULTS["A"] == {"steps": 4000, "target_total_trials": 24, "source_top_k": 0}:
    ok("optuna common Stage A defaults are 24 trials x 4000 steps")
else:
    bad(f"unexpected Stage A defaults: {mod.STAGE_DEFAULTS['A']}")

if mod.STAGE_DEFAULTS["B"] == {"steps": 12000, "target_total_trials": 10, "source_top_k": 8}:
    ok("optuna common Stage B defaults are 10 trials, top-8 source, 12000 steps")
else:
    bad(f"unexpected Stage B defaults: {mod.STAGE_DEFAULTS['B']}")

# Wrapper defaults
def require_text(path: str, snippets: list[str]) -> None:
    text = Path(path).read_text()
    for snip in snippets:
        if snip in text:
            ok(f"{path} contains {snip}")
        else:
            bad(f"{path} missing {snip}")

require_text("upair_submit_stageA_all.sh", [
    'TRIALS="${UPAIR_OPTUNA_STAGEA_TRIALS:-24}"',
    'STEPS="${UPAIR_OPTUNA_STAGEA_STEPS:-4000}"',
    'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEA_MAX_ATTEMPTS:-36}"',
    'TIME_LIMIT="${UPAIR_TIME_STAGE_A:-30:00:00}"',
    'TPE_STARTUP="${UPAIR_OPTUNA_TPE_STARTUP_TRIALS:-10}"',
    'PRUNER_STARTUP="${UPAIR_OPTUNA_PRUNER_STARTUP_TRIALS:-8}"',
    'PRUNER_MIN_TRIALS="${UPAIR_OPTUNA_PRUNER_MIN_TRIALS:-5}"',
    'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_OBJECTIVE_RECENT_K:-3}"',
])

require_text("upair_submit_stageB_all.sh", [
    'TRIALS="${UPAIR_OPTUNA_STAGEB_TRIALS:-10}"',
    'STEPS="${UPAIR_OPTUNA_STAGEB_STEPS:-12000}"',
    'SOURCE_TOP_K="${UPAIR_OPTUNA_STAGEB_SOURCE_TOP_K:-8}"',
    'MAX_ATTEMPTS="${UPAIR_OPTUNA_STAGEB_MAX_ATTEMPTS:-16}"',
    'TIME_LIMIT="${UPAIR_TIME_STAGE_B:-30:00:00}"',
    'OBJECTIVE_RECENT_K="${UPAIR_OPTUNA_STAGEB_OBJECTIVE_RECENT_K:-3}"',
])

submit_lib = Path("upair_submit_lib.sh").read_text()
if "promptmlp_d256_b4_r2_pr2" in submit_lib:
    ok("submit-lib default variant list includes promptmlp_d256_b4_r2_pr2")
else:
    bad("submit-lib default variant list is missing promptmlp_d256_b4_r2_pr2")

comprehensive = Path("scripts/run_comprehensive_mu32_ablation.py").read_text()
if "promptmlp_d256_b4_r2_pr2" in comprehensive and '"model.prompt_mlp_ratio": 2.0' in comprehensive:
    ok("final train/eval variant table includes prompt MLP ablation")
else:
    bad("final train/eval variant table missing prompt MLP ablation")

estimator_text = Path("src/upair5g/estimator.py").read_text()
for token in ["self.prompt_mlp_ratio", "self.prompt_mlp_hidden_dim", "Dense(self.prompt_mlp_hidden_dim"]:
    if token in estimator_text:
        ok(f"estimator.py contains {token}")
    else:
        bad(f"estimator.py missing {token}")

# Runtime construction check without Sionna channel construction.
sys.path.insert(0, str(Path("src").resolve()))
from upair5g.estimator import UPAIRChannelEstimator  # noqa: E402

def make_estimator_for_variant(name: str):
    local_cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
    for path, value in variants[name].items():
        node = local_cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    est = UPAIRChannelEstimator(
        ls_estimator=None,
        resource_grid=None,
        cfg=local_cfg,
        pilot_mask=tf.zeros([14, 96, 4], tf.float32),
    )
    return local_cfg, est

main_cfg, main_est = make_estimator_for_variant("main_d256_b4_r2")
prompt_cfg, prompt_est = make_estimator_for_variant("promptmlp_d256_b4_r2_pr2")

if int(main_est.prompt_mlp.layers[0].units) == 256 and float(main_cfg["model"].get("prompt_mlp_ratio", 1.0)) == 1.0:
    ok("main prompt MLP is d -> d -> d, hidden=256")
else:
    bad(f"main prompt MLP units/ratio wrong: units={main_est.prompt_mlp.layers[0].units}, ratio={main_cfg['model'].get('prompt_mlp_ratio')}")

if int(prompt_est.prompt_mlp.layers[0].units) == 512 and float(prompt_cfg["model"]["prompt_mlp_ratio"]) == 2.0:
    ok("promptmlp ablation prompt MLP is d -> 2d -> d, hidden=512")
else:
    bad(f"promptmlp MLP units/ratio wrong: units={prompt_est.prompt_mlp.layers[0].units}, ratio={prompt_cfg['model'].get('prompt_mlp_ratio')}")

if getattr(prompt_est, "input_channels", 169) == getattr(main_est, "input_channels", 169):
    ok("prompt MLP ablation leaves input feature channels unchanged")
else:
    bad("prompt MLP ablation changed input feature channels unexpectedly")

# Syntax compile.
for root in ["src", "scripts"]:
    for path in Path(root).rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            bad(f"syntax error in {path}: {exc}")

if fail:
    raise SystemExit("[PROBE] FAILED permanent Optuna/prompt-ablation probe")
print("[PROBE] PASSED permanent Optuna/prompt-ablation probe")
PY
