#!/usr/bin/env bash
# Fix incomplete true-DMRS patch: UPAIRChannelEstimator.__init__ body uses
# pilot_mask, and training/evaluation pass pilot_mask, but the constructor
# signature did not accept it, so Keras receives it in **kwargs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

[[ -f src/upair5g/estimator.py ]] || { echo "[FIX] Run from repo root." >&2; exit 1; }

python - <<'PY'
from pathlib import Path

p = Path("src/upair5g/estimator.py")
lines = p.read_text().splitlines()

# Locate class UPAIRChannelEstimator and its __init__.
class_idx = None
for i, line in enumerate(lines):
    if line.startswith("class UPAIRChannelEstimator("):
        class_idx = i
        break
if class_idx is None:
    raise SystemExit("[FIX] Could not find class UPAIRChannelEstimator.")

init_idx = None
for i in range(class_idx, min(len(lines), class_idx + 80)):
    if lines[i].strip() == "def __init__(":
        init_idx = i
        break
if init_idx is None:
    raise SystemExit("[FIX] Could not find UPAIRChannelEstimator.__init__.")

# Check only the constructor-signature block before ") -> None:".
end_sig = None
for i in range(init_idx, min(len(lines), init_idx + 40)):
    if lines[i].strip() == ") -> None:":
        end_sig = i
        break
if end_sig is None:
    raise SystemExit("[FIX] Could not find end of UPAIRChannelEstimator.__init__ signature.")

sig_block = "\n".join(lines[init_idx:end_sig + 1])
if "pilot_mask: tf.Tensor | None = None" not in sig_block:
    inserted = False
    for i in range(init_idx, end_sig + 1):
        if lines[i].strip() == "cfg: dict[str, Any],":
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            lines.insert(i + 1, indent + "pilot_mask: tf.Tensor | None = None,")
            inserted = True
            break
    if not inserted:
        raise SystemExit("[FIX] Could not insert pilot_mask argument after cfg.")
    print("[FIX] Added pilot_mask argument to UPAIRChannelEstimator.__init__ signature.")
else:
    print("[FIX] Constructor signature already accepts pilot_mask.")

# Confirm body has the override-aware self.pilot_mask assignment.
text = "\n".join(lines) + "\n"
if "tf.convert_to_tensor(pilot_mask) if pilot_mask is not None else extract_pilot_mask_per_stream(resource_grid)" not in text:
    raise SystemExit("[FIX] Body does not contain override-aware self.pilot_mask assignment.")

p.write_text(text)
PY

echo "[FIX] Done. Run: bash upair_probe_true_dmrs_mask_patch.sh"
