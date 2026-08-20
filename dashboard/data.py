from __future__ import annotations

from pathlib import Path
import base64
import json
import lzma
import pickle

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

FILES = {
    "standings": "standings_projection.csv",
    "matchups": "matchup_odds.csv",
    "players": "player_projections.csv",
    "lineups": "optimized_lineups.csv",
    "team_weeks": "team_week_projections.csv",
    "playoffs": "playoff_projection.csv",
    "diagnostics": "model_diagnostics.csv",
    "validation": "forecast_validation.csv",
    "replay": "historical_season_replay.csv",
    "ablation": "ablation_results.csv",
    "calibration": "calibration_curve.csv",
    "features": "feature_activation.csv",
    "fragility": "roster_fragility.csv",
    "manager": "manager_efficiency.csv",
    "unmatched": "unmatched_roster_players.csv",
}

# Frozen web snapshot bundled with the deployment repository. These files were
# already present together on streamlit-deploy-v1 and avoid the incomplete
# compact_snapshot manifest that referenced a missing bundle09 file.
WEB_SNAPSHOT_FILES = [
    "snapshot_final.part00",
    "snapshot_final.part01",
    "snapshot_final.part02",
    "snapshot_final.part03",
    "snapshot_final.part04",
    "snapshot_final.part05",
    "snapshot_final.part06",
    "snapshot_final.part07",
]
WEB_SNAPSHOT_B85_LENGTH = 169981

ATTRIBUTION_COLUMNS = [
    "week", "sleeper_id", "projected_mean", "direct_model_mean",
    "oppeff_model_mean", "ecr_prior_ppg", "market_weight",
    "team_volume_mult", "team_budget_mult", "opp_mult", "trait_opp_mult",
    "game_env_mult", "weather_mult", "weekly_miss_prob",
]

EXPECTED_WEB_ROWS = {
    "standings": 32,
    "matchups": 208,
    "players": 9826,
    "lineups": 3808,
    "team_weeks": 544,
    "playoffs": 32,
}


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float) -> dict:
    del mtime
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_compact_snapshot(output_dir: str, signature: tuple[tuple[str, float], ...]) -> tuple[dict[str, pd.DataFrame], dict]:
    del signature
    root = Path(output_dir)
    parts = [root / name for name in WEB_SNAPSHOT_FILES]
    missing_parts = [p.name for p in parts if not p.exists()]
    if missing_parts:
        raise RuntimeError(f"MIDA web snapshot is incomplete. Missing: {', '.join(missing_parts)}")

    encoded = "".join(p.read_text(encoding="ascii") for p in parts)
    if len(encoded) != WEB_SNAPSHOT_B85_LENGTH:
        raise RuntimeError(
            f"MIDA web snapshot length mismatch: {len(encoded):,} != {WEB_SNAPSHOT_B85_LENGTH:,}"
        )

    try:
        payload = lzma.decompress(base64.b85decode(encoded.encode("ascii")))
        snapshot = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"MIDA web snapshot could not be decoded: {type(exc).__name__}: {exc}") from exc

    if not isinstance(snapshot, dict) or "data" not in snapshot or "meta" not in snapshot:
        raise RuntimeError("MIDA web snapshot has an invalid top-level structure.")

    data = snapshot["data"]
    meta = snapshot["meta"]
    if not isinstance(data, dict):
        raise RuntimeError("MIDA web snapshot data payload is invalid.")

    for key, expected_rows in EXPECTED_WEB_ROWS.items():
        frame = data.get(key)
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(f"MIDA web snapshot is missing required table: {key}")
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"MIDA web snapshot row-count mismatch for {key}: {len(frame):,} != {expected_rows:,}"
            )

    players = data.get("players", pd.DataFrame())
    if not players.empty:
        data["attribution"] = players[[c for c in ATTRIBUTION_COLUMNS if c in players.columns]].copy()
    else:
        data["attribution"] = pd.DataFrame()

    return data, meta


def load_outputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    # Local engine mode: canonical MIDA output files take precedence.
    if (output_dir / "standings_projection.csv").exists():
        for key, name in FILES.items():
            p = output_dir / name
            if p.exists():
                data[key] = load_csv(str(p), p.stat().st_mtime)
            else:
                data[key] = pd.DataFrame()
                missing.append(name)
        attr_path = output_dir / "projection_attribution.csv"
        data["attribution"] = load_csv(str(attr_path), attr_path.stat().st_mtime) if attr_path.exists() else pd.DataFrame()
        if not attr_path.exists():
            missing.append("projection_attribution.csv")
        meta_path = output_dir / "run_metadata.json"
        if meta_path.exists():
            meta = load_json(str(meta_path), meta_path.stat().st_mtime)
        else:
            meta = {}
            missing.append("run_metadata.json")
        return data, meta, missing

    # Web mode: reconstruct the frozen forecast snapshot bundled with the repo.
    parts = [output_dir / name for name in WEB_SNAPSHOT_FILES]
    if all(p.exists() for p in parts):
        sig = tuple((p.name, p.stat().st_mtime) for p in parts)
        data, meta = load_compact_snapshot(str(output_dir), sig)
        for key in list(FILES) + ["attribution"]:
            if key not in data:
                data[key] = pd.DataFrame()
                missing.append(key)
        return data, meta, missing

    for key in list(FILES) + ["attribution"]:
        data[key] = pd.DataFrame()
    missing.extend(FILES.values())
    missing.extend(["projection_attribution.csv", "run_metadata.json"])
    return data, {}, missing


def clear_cache() -> None:
    st.cache_data.clear()
