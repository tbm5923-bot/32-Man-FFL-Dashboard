from __future__ import annotations

from pathlib import Path
import base64
import hashlib
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

WEB_SNAPSHOT_FILES = [
    "web_snapshot_v7.part000",
    "web_snapshot_v7.part001",
    "web_snapshot_v7.part002",
    "web_snapshot_v7.part003",
    "web_snapshot_v7.part004",
    "web_snapshot_v7.part005",
    "web_snapshot_v7.part006",
    "web_snapshot_v7.part007",
    "web_snapshot_v7.part008",
    "web_snapshot_v7.part009",
    "web_snapshot_v7.part010",
    "web_snapshot_v7.part011",
    "web_snapshot_v7.part012",
    "web_snapshot_v7.part013",
    "web_snapshot_v7.part014_015",
    "web_snapshot_v7.part016_017",
    "web_snapshot_v7.part018_019",
    "web_snapshot_v7.part020_021",
    "web_snapshot_v7.part022_023",
    "web_snapshot_v7.part024_025",
]
WEB_SNAPSHOT_B85_LENGTH = 101375
WEB_SNAPSHOT_SHA256 = "d24e69a67d3dd2f5ef0bc419a2115763e4061872d7542d75a15fa2bf41cbd051"

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
        raise RuntimeError(f"MIDA web snapshot length mismatch: {len(encoded):,} != {WEB_SNAPSHOT_B85_LENGTH:,}")
    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if digest != WEB_SNAPSHOT_SHA256:
        raise RuntimeError("MIDA web snapshot checksum mismatch; refusing to load corrupt data.")

    try:
        payload = lzma.decompress(base64.b85decode(encoded.encode("ascii")))
        snapshot = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"MIDA web snapshot could not be decoded: {type(exc).__name__}: {exc}") from exc

    if not isinstance(snapshot, dict) or "data" not in snapshot or "meta" not in snapshot:
        raise RuntimeError("MIDA web snapshot has an invalid top-level structure.")
    data = snapshot["data"]
    meta = snapshot["meta"]
    scales = snapshot.get("scales", {})
    if not isinstance(data, dict):
        raise RuntimeError("MIDA web snapshot data payload is invalid.")

    # Web snapshot stores display/model floats as scaled integers to keep the
    # deployment payload small. Rehydrate them before the UI consumes them.
    for table, column_scales in scales.items():
        frame = data.get(table)
        if not isinstance(frame, pd.DataFrame):
            continue
        for column, scale in column_scales.items():
            if column in frame.columns:
                frame[column] = frame[column].astype("float64") / float(scale)

    for key, expected_rows in EXPECTED_WEB_ROWS.items():
        frame = data.get(key)
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(f"MIDA web snapshot is missing required table: {key}")
        if len(frame) != expected_rows:
            raise RuntimeError(f"MIDA web snapshot row-count mismatch for {key}: {len(frame):,} != {expected_rows:,}")

    players = data.get("players", pd.DataFrame())
    if not players.empty:
        data["attribution"] = players[[c for c in ATTRIBUTION_COLUMNS if c in players.columns]].copy()
    else:
        data["attribution"] = pd.DataFrame()
    return data, meta

def load_outputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

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
