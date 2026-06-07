#!/usr/bin/env bash
# Add geometry-consistent UMi near-far modeling with fractional uplink power control.
# Intended for: /home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable_General
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_PATCH:-0}" != "1" ]]; then
  echo "[NF-PATCH] Refusing to patch outside the General repo copy." >&2
  echo "[NF-PATCH] ROOT=${ROOT}" >&2
  exit 1
fi

[[ -f src/upair5g/builders.py && -f src/upair5g/training.py && -f configs/twc_comprehensive_mu32_base.yaml ]] || {
  echo "[NF-PATCH] Run this from the General repo root." >&2
  exit 1
}

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

# ---------------------------------------------------------------------
# Config: UMi with pathloss/shadowing ON, Sionna normalization OFF,
# and external power-control-aware mean re-referencing ON.
# ---------------------------------------------------------------------
cfg.setdefault("experiment", {})["name"] = "rx16_prb8_umi_pc_trueDMRS_1dmrs_u3"

ch = cfg.setdefault("channel", {})
ch["family"] = "umi"
ch["model"] = "umi"
ch["normalize_channel"] = False
ch.setdefault("cdl_model", "C")
umi = ch.setdefault("umi", {})
umi["enable_pathloss"] = True
umi["enable_shadow_fading"] = True
umi["randomize_topology_each_batch"] = True
umi.setdefault("scenario", "umi")
umi.setdefault("o2i_model", "low")
umi.setdefault("always_generate_lsp", False)
umi.setdefault("antenna_pattern_bs", "38.901")
umi.setdefault("antenna_pattern_ut", "omni")
umi.setdefault("bs_array_rows", 4)
umi.setdefault("bs_array_cols", 4)
umi.setdefault("ut_array_rows", 1)
umi.setdefault("ut_array_cols", 1)
umi.setdefault("min_ut_velocity_mps", float(ch.get("min_speed_mps", 8.33)))
umi.setdefault("max_ut_velocity_mps", float(ch.get("max_speed_mps", 16.67)))

nf = cfg.setdefault("near_far", {})
nf.update({
    "enabled": True,
    "mode": "fractional_power_control_meanref",
    # Realistic first training range. This deliberately avoids alpha close to 0.
    # alpha=1 is equal received power; alpha≈0.7--0.9 is typical fractional compensation.
    "alpha_train_min": 0.70,
    "alpha_train_max": 1.00,
    "alpha_eval": 0.80,
    "alpha_sampling": "uniform",
    "epsilon": 1.0e-12,
    "log_stats": True,
    # Finite headroom is intentionally disabled for the first clean implementation.
    # It requires a P0/PCMAX absolute anchor; use a later controlled experiment.
    "headroom_db": None,
})

cfg.setdefault("baselines", {}).setdefault("covariance_estimation", {})["cache_name"] = "empirical_covariances_umi_pc_meanref.npz"

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[NF-PATCH] Config set to UMi pathloss/shadowing ON, normalize_channel=false, near_far.enabled=true.")

# ---------------------------------------------------------------------
# builders.py: replace TopologyRefreshingOFDMChannel with a version that
# applies external fractional power-control mean re-referencing before AWGN.
# ---------------------------------------------------------------------
p = Path("src/upair5g/builders.py")
s = p.read_text()

start = s.find("class TopologyRefreshingOFDMChannel:")
end = s.find("def _build_umi_channel(", start)
if start == -1 or end == -1:
    raise SystemExit("[NF-PATCH] Could not locate TopologyRefreshingOFDMChannel block in builders.py.")

new_class = """
class TopologyRefreshingOFDMChannel:
    # OFDM channel wrapper that regenerates UMi topology and optionally applies
    # geometry-consistent near-far modeling with fractional uplink power control.
    def __init__(self, channel_model: Any, ofdm_channel: Any, cfg: dict[str, Any], num_users: int) -> None:
        self.channel_model = channel_model
        self.ofdm_channel = ofdm_channel
        self.cfg = cfg
        self.num_users = int(num_users)
        self.return_channel = True
        self._last_topology_batch_size: int | None = None
        self._training_mode = False
        self.last_near_far_stats: dict[str, tf.Tensor] = {}

    def set_training_mode(self, training: bool) -> None:
        self._training_mode = bool(training)

    def _batch_size_from_x(self, x: tf.Tensor) -> int:
        static = x.shape[0]
        if static is not None:
            return int(static)
        try:
            return int(tf.shape(x)[0].numpy())
        except Exception as exc:
            raise RuntimeError("UMi channel requires a statically known/eager batch dimension.") from exc

    def _set_topology(self, batch_size: int) -> None:
        gen_single_sector_topology = resolve_attr(
            ["sionna.phy.channel", "sionna.phy.channel.tr38901"],
            "gen_single_sector_topology",
        )
        kwargs = _topology_kwargs_for_umi(self.cfg, batch_size=batch_size, num_users=self.num_users)
        topology = gen_single_sector_topology(**kwargs)
        if not isinstance(topology, (tuple, list)) or len(topology) < 6:
            raise RuntimeError(f"Unexpected UMi topology returned by Sionna: {type(topology)}")
        self.channel_model.set_topology(*topology)
        self._last_topology_batch_size = int(batch_size)

    def _call_clean_ofdm(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        attempts = [
            lambda: self.ofdm_channel(x),
            lambda: self.ofdm_channel([x]),
            lambda: self.ofdm_channel((x,)),
        ]
        last_err = None
        for attempt in attempts:
            try:
                return _infer_channel_pair(attempt())
            except (tf.errors.ResourceExhaustedError, MemoryError):
                raise
            except Exception as err:
                last_err = err
        raise RuntimeError("All clean UMi OFDM channel calling conventions failed.") from last_err

    def _sample_alpha(self, batch_size: int) -> tf.Tensor:
        nf_cfg = get_cfg(self.cfg, "near_far", {})
        if self._training_mode:
            amin = float(nf_cfg.get("alpha_train_min", 0.70))
            amax = float(nf_cfg.get("alpha_train_max", 1.00))
            amin = max(0.0, min(1.0, amin))
            amax = max(amin, min(1.0, amax))
            if abs(amax - amin) < 1e-12:
                alpha = tf.fill([int(batch_size), 1], tf.constant(amin, tf.float32))
            else:
                alpha = tf.random.uniform([int(batch_size), 1], minval=amin, maxval=amax, dtype=tf.float32)
        else:
            alpha_eval = float(nf_cfg.get("alpha_eval", 0.80))
            alpha_eval = max(0.0, min(1.0, alpha_eval))
            alpha = tf.fill([int(batch_size), 1], tf.constant(alpha_eval, tf.float32))
        return alpha

    def _apply_fractional_power_control(
        self,
        h_raw: tf.Tensor,
        x: tf.Tensor,
        no: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        nf_cfg = get_cfg(self.cfg, "near_far", {})
        eps = tf.constant(float(nf_cfg.get("epsilon", 1e-12)), tf.float32)

        h_raw = tf.convert_to_tensor(h_raw)
        h_dtype = h_raw.dtype
        x = tf.cast(tf.convert_to_tensor(x), h_dtype)

        # h_raw shape: [B, 1, Nr, U, S, T, F]
        # x shape:     [B, U, S, T, F]
        if h_raw.shape.rank != 7:
            raise ValueError(f"Expected h rank 7 [B,1,Nr,U,S,T,F], got rank {h_raw.shape.rank}.")
        if x.shape.rank != 5:
            raise ValueError(f"Expected x rank 5 [B,U,S,T,F], got rank {x.shape.rank}.")

        batch_size = self._batch_size_from_x(x)
        abs2 = tf.math.real(h_raw * tf.math.conj(h_raw))
        p_u = tf.reduce_mean(abs2, axis=[1, 2, 4, 5, 6])  # [B,U]
        p_safe = tf.maximum(tf.cast(p_u, tf.float32), eps)

        alpha = self._sample_alpha(batch_size)             # [B,1]
        rho = 1.0 - alpha                                  # residual near-far exponent

        # Headroom-unlimited fractional compensation:
        # P*_u ∝ P_u^(1-alpha), then normalize mean_u P*_u = 1.
        log_p = tf.math.log(p_safe)
        log_eff = rho * log_p
        num_u = tf.cast(tf.shape(log_eff)[1], tf.float32)
        log_mean_eff = tf.reduce_logsumexp(log_eff, axis=1, keepdims=True) - tf.math.log(num_u)
        log_p_star = log_eff - log_mean_eff
        p_star = tf.exp(log_p_star)                        # [B,U], mean over U = 1

        scale = tf.sqrt(p_star / p_safe)                   # [B,U]
        scale_bc = tf.reshape(tf.cast(scale, h_dtype), [tf.shape(h_raw)[0], 1, 1, tf.shape(h_raw)[3], 1, 1, 1])
        h_star = h_raw * scale_bc

        x_bc = tf.expand_dims(tf.expand_dims(x, axis=1), axis=2)  # [B,1,1,U,S,T,F]
        y_clean = tf.reduce_sum(h_star * x_bc, axis=[3, 4])       # [B,1,Nr,T,F]
        y = _add_awgn_once(y_clean, no)

        raw_spread_db = (tf.reduce_max(log_p, axis=1) - tf.reduce_min(log_p, axis=1)) * (10.0 / tf.math.log(10.0))
        post_spread_db = (tf.reduce_max(log_p_star, axis=1) - tf.reduce_min(log_p_star, axis=1)) * (10.0 / tf.math.log(10.0))
        self.last_near_far_stats = {
            "alpha": tf.identity(alpha),
            "raw_power": tf.identity(p_safe),
            "post_power": tf.identity(p_star),
            "raw_spread_db": tf.identity(raw_spread_db),
            "post_spread_db": tf.identity(post_spread_db),
            "post_power_mean": tf.reduce_mean(p_star, axis=1),
        }
        return y, h_star

    def __call__(self, x: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x)
        batch_size = self._batch_size_from_x(x)
        if bool(get_cfg(self.cfg, "channel.umi.randomize_topology_each_batch", True)) or self._last_topology_batch_size != batch_size:
            self._set_topology(batch_size)

        y_clean_raw, h_raw = self._call_clean_ofdm(x)
        if bool(get_cfg(self.cfg, "near_far.enabled", False)):
            return self._apply_fractional_power_control(h_raw, x, no)

        return _add_awgn_once(y_clean_raw, no), h_raw


"""
s = s[:start] + new_class + s[end:]

old = """def _build_umi_channel(cfg: dict[str, Any], tx: Any, num_users: int) -> TopologyRefreshingOFDMChannel:
    channel_model = _build_umi_channel_model(cfg, num_users=num_users)
    ofdm = _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)
    return TopologyRefreshingOFDMChannel(channel_model, ofdm, cfg, num_users=num_users)
"""
new = """def _build_umi_channel(cfg: dict[str, Any], tx: Any, num_users: int) -> TopologyRefreshingOFDMChannel:
    channel_model = _build_umi_channel_model(cfg, num_users=num_users)
    # AWGN is added by TopologyRefreshingOFDMChannel after optional near-far
    # re-referencing so that y, h, and N0 remain mutually consistent.
    ofdm = _build_ofdm_channel(cfg, tx, channel_model, add_awgn=False)
    return TopologyRefreshingOFDMChannel(channel_model, ofdm, cfg, num_users=num_users)
"""
if old not in s:
    if "add_awgn=False" in s and "def _build_umi_channel" in s:
        print("[NF-PATCH] _build_umi_channel already appears to use add_awgn=False.")
    else:
        raise SystemExit("[NF-PATCH] Could not locate _build_umi_channel block for add_awgn=False replacement.")
else:
    s = s.replace(old, new)

p.write_text(s)
print("[NF-PATCH] builders.py now applies fractional PC mean re-referencing before AWGN.")

# ---------------------------------------------------------------------
# training.py: tell the channel whether the current batch is training or eval,
# so alpha uses train range during training and alpha_eval during validation/eval.
# ---------------------------------------------------------------------
tp = Path("src/upair5g/training.py")
ts = tp.read_text()
needle = "    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))\n    y, h = call_channel(channel, x, no)\n"
replacement = """    no = ebno_db_to_no(ebno_db, tx=tx, resource_grid=get_resource_grid(tx))
    set_training_mode = getattr(channel, "set_training_mode", None)
    if callable(set_training_mode):
        set_training_mode(bool(training))
    y, h = call_channel(channel, x, no)
"""
if needle in ts:
    ts = ts.replace(needle, replacement, 1)
elif 'set_training_mode = getattr(channel, "set_training_mode", None)' in ts:
    print("[NF-PATCH] training.py already sets channel training mode.")
else:
    raise SystemExit("[NF-PATCH] Could not locate _make_batch channel-call block in training.py.")
tp.write_text(ts)

# ---------------------------------------------------------------------
# Prefixes: separate this experiment from normalized UMi.
# ---------------------------------------------------------------------
repls = {
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageA",
    "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMiPC_u34610_1dmrs_stageB",
}
for path in [
    "upair_submit_stageA_all.sh",
    "upair_submit_stageB_all.sh",
    "upair_submit_train_eval_all.sh",
    "upair_probe_after_stageB.sh",
    "upair_probe_after_train_eval.sh",
    "upair_probe_clean_start.sh",
    "upair_probe_smart_optuna_40k.sh",
    "upair_probe_general_umi_ready.sh",
]:
    f = Path(path)
    if not f.exists():
        continue
    text = f.read_text()
    for a, b in repls.items():
        text = text.replace(a, b)
    f.write_text(text)

# Clean old normalized-UMi smoke/Optuna evidence.
for folder in ["optuna", "logs", "TWC_plots_comprehensive", "_smoke_umi_runtime", "_smoke_umi_pc_runtime"]:
    path = Path(folder)
    if path.exists():
        import shutil
        shutil.rmtree(path)
Path("optuna").mkdir(exist_ok=True)
Path("logs/optuna").mkdir(parents=True, exist_ok=True)
Path("logs/submit").mkdir(parents=True, exist_ok=True)
Path("logs/train_eval").mkdir(parents=True, exist_ok=True)
Path("logs/smoke").mkdir(parents=True, exist_ok=True)

# Ignore runtime outputs.
gp = Path(".gitignore")
gs = gp.read_text() if gp.exists() else ""
items = ["optuna/", "logs/", "TWC_plots_comprehensive/", "_smoke_*/", "*.out", "*.err", "*.log", "*.db", "*.sqlite", "*.sqlite3"]
lines = gs.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
gp.write_text("\n".join(lines).rstrip() + "\n")

print("[NF-PATCH] Prefixes updated to trueDMRS_UMiPC and old runtime evidence removed.")
PY

echo "[NF-PATCH] Done. Run:"
echo "  bash upair_probe_umi_pc_nearfar_ready.sh"
echo "  bash upair_probe_umi_pc_nearfar_runtime.sh"
