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

ATTRIBUTION_COLUMNS = [
    "week", "sleeper_id", "projected_mean", "direct_model_mean",
    "oppeff_model_mean", "ecr_prior_ppg", "market_weight",
    "team_volume_mult", "team_budget_mult", "opp_mult", "trait_opp_mult",
    "game_env_mult", "weather_mult", "weekly_miss_prob",
]


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
    parts = sorted(root.glob("compact_snapshot.part*"))
    encoded = "".join(p.read_text(encoding="ascii") for p in parts)
    payload = lzma.decompress(base64.b85decode(encoded.encode("ascii")))
    snapshot = pickle.loads(payload)
    data = snapshot["data"]
    meta = snapshot["meta"]
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

    # Web mode: reconstruct the frozen, compact forecast snapshot.
    parts = sorted(output_dir.glob("compact_snapshot.part*"))
    if parts:
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
