#!/usr/bin/env bash
# Minimal no-GPU-import-safe source check that the constructor now accepts pilot_mask
# before running the heavier runtime probe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python - <<'PY'
from pathlib import Path
s = Path("src/upair5g/estimator.py").read_text().splitlines()
class_i = next(i for i,l in enumerate(s) if l.startswith("class UPAIRChannelEstimator("))
init_i = next(i for i in range(class_i, class_i+80) if s[i].strip()=="def __init__(")
end_i = next(i for i in range(init_i, init_i+50) if s[i].strip()==") -> None:")
sig = "\n".join(s[init_i:end_i+1])
assert "pilot_mask: tf.Tensor | None = None" in sig, sig
assert "tf.convert_to_tensor(pilot_mask) if pilot_mask is not None else extract_pilot_mask_per_stream(resource_grid)" in "\n".join(s)
print("[OK] UPAIRChannelEstimator.__init__ signature accepts pilot_mask.")
print("[OK] self.pilot_mask uses explicit pilot_mask override when provided.")
PY
