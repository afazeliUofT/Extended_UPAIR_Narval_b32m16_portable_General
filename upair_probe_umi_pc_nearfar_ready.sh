#!/usr/bin/env bash
# Static/minimal probe for UMi fractional-power-control near-far configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
import yaml
import sys

fail = False
def ok(msg): print("[OK]", msg)
def bad(msg):
    global fail
    print("[FAIL]", msg, file=sys.stderr)
    fail = True

cfg = yaml.safe_load(Path("configs/twc_comprehensive_mu32_base.yaml").read_text())
ch = cfg.get("channel", {})
umi = ch.get("umi", {})
nf = cfg.get("near_far", {})

if str(ch.get("family")).lower() == "umi" and str(ch.get("model")).lower() == "umi":
    ok("channel family/model are UMi")
else:
    bad(f"channel family/model not UMi: {ch.get('family')} {ch.get('model')}")

if ch.get("normalize_channel") is False:
    ok("Sionna OFDM normalize_channel is false")
else:
    bad(f"normalize_channel should be false, got {ch.get('normalize_channel')}")

if umi.get("enable_pathloss") is True and umi.get("enable_shadow_fading") is True:
    ok("UMi pathloss and shadow fading are enabled")
else:
    bad(f"pathloss/shadow config wrong: {umi.get('enable_pathloss')} {umi.get('enable_shadow_fading')}")

if nf.get("enabled") is True and str(nf.get("mode")) == "fractional_power_control_meanref":
    ok("near_far fractional PC mean-reference mode is enabled")
else:
    bad(f"near_far config wrong: {nf}")

amin = float(nf.get("alpha_train_min", -1))
amax = float(nf.get("alpha_train_max", -1))
aeval = float(nf.get("alpha_eval", -1))
if 0.65 <= amin <= amax <= 1.0 and 0.7 <= aeval <= 0.95:
    ok(f"alpha range avoids near-zero values: train=[{amin},{amax}], eval={aeval}")
else:
    bad(f"unexpected alpha values: train=[{amin},{amax}], eval={aeval}")

builders = Path("src/upair5g/builders.py").read_text()
for token in [
    "class TopologyRefreshingOFDMChannel",
    "_apply_fractional_power_control",
    "tf.reduce_logsumexp",
    "post_power_mean",
    "add_awgn=False",
    "sionna.phy.channel",
    "gen_single_sector_topology",
]:
    if token in builders:
        ok(f"builders.py contains {token}")
    else:
        bad(f"builders.py missing {token}")

training = Path("src/upair5g/training.py").read_text()
if 'set_training_mode = getattr(channel, "set_training_mode", None)' in training:
    ok("training.py sets channel training/eval mode before channel call")
else:
    bad("training.py does not set channel training/eval mode")

for f in ["upair_submit_stageA_all.sh", "upair_submit_stageB_all.sh", "upair_submit_train_eval_all.sh"]:
    text = Path(f).read_text()
    if "trueDMRS_UMiPC" in text:
        ok(f"{f} uses UMiPC prefix")
    else:
        bad(f"{f} does not use UMiPC prefix")

old_dbs = list(Path("optuna").glob("*.db")) if Path("optuna").exists() else []
if old_dbs:
    bad("old Optuna DBs remain: " + ", ".join(str(p) for p in old_dbs[:5]))
else:
    ok("no old Optuna DBs remain")

raise SystemExit(1 if fail else 0)
PY

source "${ROOT}/upair_portable_env.sh"
upair_ensure_venv

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
failed = False
for root in ["src", "scripts"]:
    for path in Path(root).rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            print(f"[SYNTAX-FAIL] {path}: {exc}")
            failed = True
raise SystemExit(1 if failed else 0)
PY

echo "[PROBE] PASSED UMiPC static/minimal probe"
