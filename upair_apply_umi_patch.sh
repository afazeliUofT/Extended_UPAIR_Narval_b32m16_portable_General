#!/usr/bin/env bash
# Add randomized 3GPP TR 38.901 UMi channel support to the General UPAIR repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ "${ROOT}" != *"Extended_UPAIR_Narval_b32m16_portable_General"* && "${UPAIR_ALLOW_NON_GENERAL_PATCH:-0}" != "1" ]]; then
  echo "[UMI-PATCH] Refusing to patch outside the General repo copy." >&2
  echo "[UMI-PATCH] ROOT=${ROOT}" >&2
  exit 1
fi
[[ -f src/upair5g/builders.py && -f configs/twc_comprehensive_mu32_base.yaml ]] || { echo "[UMI-PATCH] Run from repo root." >&2; exit 1; }

python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("configs/twc_comprehensive_mu32_base.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("experiment", {})["name"] = "rx16_prb8_umi_trueDMRS_1dmrs_u3"
ch = cfg.setdefault("channel", {})
if "cdl_model" not in ch:
    ch["cdl_model"] = ch.get("model", "C")
ch["family"] = "umi"
ch["model"] = "umi"
ch["normalize_channel"] = True
ch.setdefault("num_rx_ant", 16)
ch.setdefault("num_tx_ant", 1)
ch.setdefault("min_speed_mps", 8.33)
ch.setdefault("max_speed_mps", 16.67)
umi = ch.setdefault("umi", {})
umi.update({
    "scenario": "umi",
    "o2i_model": "low",
    "enable_pathloss": False,
    "enable_shadow_fading": False,
    "always_generate_lsp": False,
    "randomize_topology_each_batch": True,
    "antenna_pattern_bs": "38.901",
    "antenna_pattern_ut": "omni",
    "bs_array_rows": 4,
    "bs_array_cols": 4,
    "ut_array_rows": 1,
    "ut_array_cols": 1,
    "min_bs_ut_dist": None,
    "isd": None,
    "bs_height": None,
    "min_ut_height": None,
    "max_ut_height": None,
    "indoor_probability": None,
    "min_ut_velocity_mps": float(ch.get("min_speed_mps", 8.33)),
    "max_ut_velocity_mps": float(ch.get("max_speed_mps", 16.67)),
})
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[UMI-PATCH] Config default channel.family/model set to randomized UMi.")

p = Path("src/upair5g/builders.py")
s = p.read_text()
if "class TopologyRefreshingOFDMChannel" not in s:
    anchor = "def _build_cdl_channel_model(cfg: dict[str, Any]) -> Any:\n"
    idx = s.find(anchor)
    if idx == -1:
        raise SystemExit("[UMI-PATCH] Could not locate _build_cdl_channel_model anchor.")
    umi_code = '''
def _channel_family(cfg: dict[str, Any]) -> str:
    value = get_cfg(cfg, "channel.family", None)
    if value is None:
        value = get_cfg(cfg, "channel.model", "cdl")
    return str(value).strip().lower()


def _build_panel_array_from_shape(
    *,
    num_rows: int,
    num_cols: int,
    carrier_frequency: float,
    antenna_pattern: str,
) -> Any:
    PanelArray = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "PanelArray")
    return instantiate_filtered(
        PanelArray,
        num_rows_per_panel=int(num_rows),
        num_cols_per_panel=int(num_cols),
        polarization="single",
        polarization_type="V",
        antenna_pattern=str(antenna_pattern),
        carrier_frequency=float(carrier_frequency),
    )


def _build_umi_channel_model(cfg: dict[str, Any], num_users: int) -> Any:
    UMi = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "UMi")
    channel_cfg = cfg["channel"]
    umi_cfg = channel_cfg.get("umi", {})
    pusch_cfg = cfg["pusch"]
    carrier_frequency = float(pusch_cfg["carrier_frequency_hz"])
    ut_array = _build_panel_array_from_shape(
        num_rows=int(umi_cfg.get("ut_array_rows", 1)),
        num_cols=int(umi_cfg.get("ut_array_cols", max(1, int(channel_cfg.get("num_tx_ant", 1))))),
        carrier_frequency=carrier_frequency,
        antenna_pattern=str(umi_cfg.get("antenna_pattern_ut", "omni")),
    )
    bs_array = _build_panel_array_from_shape(
        num_rows=int(umi_cfg.get("bs_array_rows", 4)),
        num_cols=int(umi_cfg.get("bs_array_cols", max(1, int(channel_cfg.get("num_rx_ant", 16)) // 4))),
        carrier_frequency=carrier_frequency,
        antenna_pattern=str(umi_cfg.get("antenna_pattern_bs", "38.901")),
    )
    kwargs = {
        "carrier_frequency": carrier_frequency,
        "o2i_model": str(umi_cfg.get("o2i_model", "low")),
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": "uplink",
        "enable_pathloss": bool(umi_cfg.get("enable_pathloss", False)),
        "enable_shadow_fading": bool(umi_cfg.get("enable_shadow_fading", False)),
        "always_generate_lsp": bool(umi_cfg.get("always_generate_lsp", False)),
        "precision": get_cfg(cfg, "system.precision", "single"),
    }
    try:
        return instantiate_filtered(UMi, **kwargs)
    except Exception:
        kwargs.pop("precision", None)
        return instantiate_filtered(UMi, **kwargs)


def _topology_kwargs_for_umi(cfg: dict[str, Any], *, batch_size: int, num_users: int) -> dict[str, Any]:
    channel_cfg = cfg["channel"]
    umi_cfg = channel_cfg.get("umi", {})
    kwargs: dict[str, Any] = {"batch_size": int(batch_size), "num_ut": int(num_users), "scenario": str(umi_cfg.get("scenario", "umi"))}
    optional_map = {
        "min_bs_ut_dist": "min_bs_ut_dist",
        "isd": "isd",
        "bs_height": "bs_height",
        "min_ut_height": "min_ut_height",
        "max_ut_height": "max_ut_height",
        "indoor_probability": "indoor_probability",
        "min_ut_velocity": "min_ut_velocity_mps",
        "max_ut_velocity": "max_ut_velocity_mps",
    }
    for api_name, cfg_name in optional_map.items():
        value = umi_cfg.get(cfg_name, None)
        if value is not None:
            kwargs[api_name] = float(value)
    return kwargs


class TopologyRefreshingOFDMChannel:
    def __init__(self, channel_model: Any, ofdm_channel: Any, cfg: dict[str, Any], num_users: int) -> None:
        self.channel_model = channel_model
        self.ofdm_channel = ofdm_channel
        self.cfg = cfg
        self.num_users = int(num_users)
        self.return_channel = True
        self._last_topology_batch_size: int | None = None

    def _batch_size_from_x(self, x: tf.Tensor) -> int:
        static = x.shape[0]
        if static is not None:
            return int(static)
        try:
            return int(tf.shape(x)[0].numpy())
        except Exception as exc:
            raise RuntimeError("UMi channel requires a statically known/eager batch dimension.") from exc

    def _set_topology(self, batch_size: int) -> None:
        gen_single_sector_topology = resolve_attr(["sionna.phy.channel.tr38901", "sionna.channel.tr38901"], "gen_single_sector_topology")
        kwargs = _topology_kwargs_for_umi(self.cfg, batch_size=batch_size, num_users=self.num_users)
        topology = gen_single_sector_topology(**kwargs)
        if not isinstance(topology, (tuple, list)) or len(topology) < 6:
            raise RuntimeError(f"Unexpected UMi topology returned by Sionna: {type(topology)}")
        self.channel_model.set_topology(*topology)
        self._last_topology_batch_size = int(batch_size)

    def __call__(self, x: tf.Tensor, no: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x)
        batch_size = self._batch_size_from_x(x)
        if bool(get_cfg(self.cfg, "channel.umi.randomize_topology_each_batch", True)) or self._last_topology_batch_size != batch_size:
            self._set_topology(batch_size)
        attempts = [
            lambda: self.ofdm_channel(x, no),
            lambda: self.ofdm_channel([x, no]),
            lambda: self.ofdm_channel((x, no)),
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
        raise RuntimeError("All UMi OFDM channel calling conventions failed.") from last_err


def _build_umi_channel(cfg: dict[str, Any], tx: Any, num_users: int) -> TopologyRefreshingOFDMChannel:
    channel_model = _build_umi_channel_model(cfg, num_users=num_users)
    ofdm = _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)
    return TopologyRefreshingOFDMChannel(channel_model, ofdm, cfg, num_users=num_users)


'''
    s = s[:idx] + umi_code + s[idx:]
    print("[UMI-PATCH] Added UMi builder/helper classes.")
else:
    print("[UMI-PATCH] UMi builder/helper classes already present.")

s = s.replace('        "model": str(channel_cfg["model"]),', '        "model": str(channel_cfg.get("cdl_model", channel_cfg.get("model", "C"))),')
s = s.replace('            str(channel_cfg["model"]),', '            str(channel_cfg.get("cdl_model", channel_cfg.get("model", "C"))),')

old_build_channel = '''def build_channel(cfg: dict[str, Any], tx: Any) -> Any:
    num_tx = int(first_present_attr(tx, ["_upair_num_users", "num_tx", "_num_tx"], 1))
    if multiuser_enabled(cfg) and num_tx > 1:
        return _build_independent_multiuser_channel(cfg, num_tx)

    channel_model = _build_cdl_channel_model(cfg)
    return _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)
'''
new_build_channel = '''def build_channel(cfg: dict[str, Any], tx: Any) -> Any:
    num_tx = int(first_present_attr(tx, ["_upair_num_users", "num_tx", "_num_tx"], 1))
    family = _channel_family(cfg)

    if family in {"umi", "urban_micro", "urban_microcell", "tr38901_umi"}:
        return _build_umi_channel(cfg, tx, num_users=num_tx)

    if multiuser_enabled(cfg) and num_tx > 1:
        return _build_independent_multiuser_channel(cfg, num_tx)

    channel_model = _build_cdl_channel_model(cfg)
    return _build_ofdm_channel(cfg, tx, channel_model, add_awgn=True)
'''
if old_build_channel not in s:
    if 'family in {"umi", "urban_micro", "urban_microcell", "tr38901_umi"}' in s:
        print("[UMI-PATCH] build_channel already selects UMi.")
    else:
        raise SystemExit("[UMI-PATCH] Could not locate original build_channel block.")
else:
    s = s.replace(old_build_channel, new_build_channel)
    print("[UMI-PATCH] build_channel now selects UMi when channel.family/model is umi.")

p.write_text(s)

repls = {
    "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageA": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA",
    "clean_b32_prb8_d256_40k_smart_trueDMRS_u34610_1dmrs_stageB": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB",
    "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageA": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageA",
    "clean_b32_prb8_d256_40k_smart_u34610_1dmrs_stageB": "clean_b32_prb8_d256_40k_smart_trueDMRS_UMi_u34610_1dmrs_stageB",
}
for path in ["upair_submit_stageA_all.sh", "upair_submit_stageB_all.sh", "upair_submit_train_eval_all.sh", "upair_probe_after_stageB.sh", "upair_probe_after_train_eval.sh", "upair_probe_clean_start.sh", "upair_probe_smart_optuna_40k.sh"]:
    f = Path(path)
    if not f.exists():
        continue
    text = f.read_text()
    for a, b in repls.items():
        text = text.replace(a, b)
    f.write_text(text)
print("[UMI-PATCH] Wrapper default prefixes updated to trueDMRS_UMi.")

p = Path(".gitignore")
s = p.read_text() if p.exists() else ""
items = ["optuna/", "logs/", "TWC_plots_comprehensive/", "outputs/", "plots/", "metrics/", "checkpoints/", "artifacts/", "_smoke_*/", "patch_backups/", "*.out", "*.err", "*.log", "*.db", "*.sqlite", "*.sqlite3"]
lines = s.splitlines()
for item in items:
    if item not in lines:
        lines.append(item)
p.write_text("\n".join(lines).rstrip() + "\n")
print("[UMI-PATCH] .gitignore updated.")
PY

if [[ "${UPAIR_KEEP_OLD_CDL_OPTUNA:-0}" != "1" ]]; then
  echo "[UMI-PATCH] Removing old Optuna/log/training evidence. Set UPAIR_KEEP_OLD_CDL_OPTUNA=1 to keep."
  rm -rf optuna logs TWC_plots_comprehensive _smoke_*
  mkdir -p optuna logs/optuna logs/submit logs/train_eval logs/smoke
fi

echo "[UMI-PATCH] Done. Next:"
echo "  bash upair_probe_general_umi_ready.sh"
echo "  bash upair_probe_umi_runtime_channel.sh"
