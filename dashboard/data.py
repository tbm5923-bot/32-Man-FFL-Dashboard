from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import base64
import json
import lzma
import tarfile

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
    "attribution": "projection_attribution.csv",
    "fragility": "roster_fragility.csv",
    "manager": "manager_efficiency.csv",
    "unmatched": "unmatched_roster_players.csv",
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
def load_web_snapshot(output_dir: str, signature: tuple[tuple[str, float], ...]) -> tuple[dict[str, bytes], dict]:
    del signature
    root = Path(output_dir)
    parts = sorted(root.glob("web_snapshot.part*"))
    encoded = "".join(p.read_text(encoding="ascii") for p in parts)
    tar_bytes = lzma.decompress(base64.b85decode(encoded.encode("ascii")))
    raw: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:") as tf:
        for member in tf.getmembers():
            if member.isfile():
                f = tf.extractfile(member)
                if f is not None:
                    raw[member.name] = f.read()
    meta = json.loads(raw.get("run_metadata.json", b"{}").decode("utf-8"))
    return raw, meta


def load_outputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    # Local-engine mode: prefer canonical CSV/JSON files when present.
    canonical_present = (output_dir / "standings_projection.csv").exists()
    if canonical_present:
        for key, name in FILES.items():
            p = output_dir / name
            if p.exists():
                data[key] = load_csv(str(p), p.stat().st_mtime)
            else:
                data[key] = pd.DataFrame()
                missing.append(name)
        meta_path = output_dir / "run_metadata.json"
        if meta_path.exists():
            meta = load_json(str(meta_path), meta_path.stat().st_mtime)
        else:
            meta = {}
            missing.append("run_metadata.json")
        return data, meta, missing

    # Web mode: load the compact frozen snapshot committed with the app.
    parts = sorted(output_dir.glob("web_snapshot.part*"))
    if parts:
        sig = tuple((p.name, p.stat().st_mtime) for p in parts)
        raw, meta = load_web_snapshot(str(output_dir), sig)
        for key, name in FILES.items():
            payload = raw.get(name)
            if payload is None:
                data[key] = pd.DataFrame()
                missing.append(name)
            else:
                data[key] = pd.read_csv(StringIO(payload.decode("utf-8")))
        if "run_metadata.json" not in raw:
            missing.append("run_metadata.json")
        return data, meta, missing

    for key in FILES:
        data[key] = pd.DataFrame()
    missing.extend(FILES.values())
    missing.append("run_metadata.json")
    return data, {}, missing


def clear_cache() -> None:
    st.cache_data.clear()
