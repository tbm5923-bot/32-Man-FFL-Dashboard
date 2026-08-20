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

# The original compact snapshot is split across these files. One middle chunk
# (compact_snapshot.bundle09) was lost during the initial GitHub publish, but an
# earlier repack of the same byte stream remains in snapshot_final.part05/06.
# We reconstruct only that missing span and still verify the original full hash.
WEB_SNAPSHOT_PREFIX_FILES = [
    "compact_snapshot.part00",
    "compact_snapshot.part01",
    "compact_snapshot.part02",
    "compact_snapshot.part03",
    "compact_snapshot.part04",
    "compact_snapshot.part050",
    "compact_snapshot.part051",
    "compact_snapshot.part060",
    "compact_snapshot.part061",
    "compact_snapshot.bundle07",
]
WEB_SNAPSHOT_SUFFIX_FILES = [
    "compact_snapshot.bundle11",
    "compact_snapshot.bundle13",
]
WEB_SNAPSHOT_RECOVERY_FILES = [
    "snapshot_final.part05",
    "snapshot_final.part06",
]
WEB_SNAPSHOT_REQUIRED_FILES = (
    WEB_SNAPSHOT_PREFIX_FILES
    + WEB_SNAPSHOT_RECOVERY_FILES
    + WEB_SNAPSHOT_SUFFIX_FILES
)

WEB_SNAPSHOT_B85_LENGTH = 170250
WEB_SNAPSHOT_SHA256 = "62a44eda00210892ef1972168702ed418c6a4bb6b5c4d2e227b4ba712ccfda25"
WEB_SNAPSHOT_RECOVERED_BUNDLE09_LENGTH = 23119
# snapshot_final.part05 covers encoded offsets 100000:120000. The missing
# bundle09 starts at encoded offset 100002 and ends at 123121.
RECOVERY_PART05_START = 2
RECOVERY_PART06_END = 3121

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
def load_compact_snapshot(
    output_dir: str,
    signature: tuple[tuple[str, float], ...],
) -> tuple[dict[str, pd.DataFrame], dict]:
    del signature
    root = Path(output_dir)

    required = [root / name for name in WEB_SNAPSHOT_REQUIRED_FILES]
    missing_parts = [p.name for p in required if not p.exists()]
    if missing_parts:
        raise RuntimeError(
            f"MIDA web snapshot is incomplete. Missing: {', '.join(missing_parts)}"
        )

    prefix = "".join(
        (root / name).read_text(encoding="ascii")
        for name in WEB_SNAPSHOT_PREFIX_FILES
    )
    suffix = "".join(
        (root / name).read_text(encoding="ascii")
        for name in WEB_SNAPSHOT_SUFFIX_FILES
    )

    recovery_part05 = (root / "snapshot_final.part05").read_text(encoding="ascii")
    recovery_part06 = (root / "snapshot_final.part06").read_text(encoding="ascii")
    recovered_bundle09 = (
        recovery_part05[RECOVERY_PART05_START:]
        + recovery_part06[:RECOVERY_PART06_END]
    )
    if len(recovered_bundle09) != WEB_SNAPSHOT_RECOVERED_BUNDLE09_LENGTH:
        raise RuntimeError(
            "MIDA web snapshot recovery length mismatch: "
            f"{len(recovered_bundle09):,} != "
            f"{WEB_SNAPSHOT_RECOVERED_BUNDLE09_LENGTH:,}"
        )

    encoded = prefix + recovered_bundle09 + suffix
    if len(encoded) != WEB_SNAPSHOT_B85_LENGTH:
        raise RuntimeError(
            f"MIDA web snapshot length mismatch: {len(encoded):,} != "
            f"{WEB_SNAPSHOT_B85_LENGTH:,}"
        )

    digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if digest != WEB_SNAPSHOT_SHA256:
        raise RuntimeError(
            "MIDA web snapshot checksum mismatch; refusing to load partial/corrupt data."
        )

    payload = lzma.decompress(base64.b85decode(encoded.encode("ascii")))
    snapshot = pickle.loads(payload)
    data = snapshot["data"]
    meta = snapshot["meta"]

    players = data.get("players", pd.DataFrame())
    if not players.empty:
        data["attribution"] = players[
            [c for c in ATTRIBUTION_COLUMNS if c in players.columns]
        ].copy()
    else:
        data["attribution"] = pd.DataFrame()

    return data, meta


def load_outputs(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
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
        data["attribution"] = (
            load_csv(str(attr_path), attr_path.stat().st_mtime)
            if attr_path.exists()
            else pd.DataFrame()
        )
        if not attr_path.exists():
            missing.append("projection_attribution.csv")

        meta_path = output_dir / "run_metadata.json"
        if meta_path.exists():
            meta = load_json(str(meta_path), meta_path.stat().st_mtime)
        else:
            meta = {}
            missing.append("run_metadata.json")

        return data, meta, missing

    # Web mode: reconstruct the frozen compact forecast snapshot.
    required = [output_dir / name for name in WEB_SNAPSHOT_REQUIRED_FILES]
    if all(p.exists() for p in required):
        sig = tuple((p.name, p.stat().st_mtime) for p in required)
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
