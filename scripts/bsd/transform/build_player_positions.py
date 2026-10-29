from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

PLAYER_FILE = RAW_DIR / "players.parquet"
SEASON_FILE = PROCESSED_DIR / "player_season_metrics.parquet"
OUTPUT_FILE = PROCESSED_DIR / "player_season_metrics_enriched.parquet"

POSITION_MAP = {
    "G": "GK",
    "GK": "GK",
    "GOALKEEPER": "GK",
    "D": "DEF",
    "DEF": "DEF",
    "DEFENDER": "DEF",
    "M": "MID",
    "MID": "MID",
    "MIDFIELDER": "MID",
    "F": "FWD",
    "FW": "FWD",
    "FWD": "FWD",
    "FORWARD": "FWD",
    "ATTACKER": "FWD",
}


def normalize_position(value):
    if pd.isna(value):
        return np.nan

    key = str(value).strip().upper()
    return POSITION_MAP.get(key, "OTHER")


def main():
    for path in [PLAYER_FILE, SEASON_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    players = pd.read_parquet(PLAYER_FILE)
    season = pd.read_parquet(SEASON_FILE)

    if players["player_id"].duplicated().any():
        duplicates = players.loc[
            players["player_id"].duplicated(keep=False),
            "player_id",
        ].tolist()
        raise ValueError(
            "players.parquet contains duplicate player_id values: "
            f"{duplicates[:10]}"
        )

    metadata_columns = [
        "player_id",
        "player_name",
        "position",
        "date_of_birth",
        "height_cm",
        "preferred_foot",
        "nationality",
        "nationality_code",
        "market_value_eur",
        "team_id",
        "team_name",
        "contract_until",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in players.columns
    ]

    enriched = season.merge(
        players[metadata_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_profile"),
    )

    enriched["position_group"] = (
        enriched["position"].apply(normalize_position)
    )

    if "date_of_birth" in enriched.columns:
        dob = pd.to_datetime(
            enriched["date_of_birth"],
            errors="coerce",
        )

        reference_date = pd.Timestamp("2026-07-01")

        enriched["age_at_season_start"] = (
            ((reference_date - dob).dt.days / 365.2425)
            .round(1)
        )

    preferred_order = [
        "player_id",
        "player_name",
        "position",
        "position_group",
        "current_team_id",
        "team_name",
        "age_at_season_start",
        "date_of_birth",
        "height_cm",
        "preferred_foot",
        "nationality",
        "nationality_code",
        "market_value_eur",
        "contract_until",
        "minutes",
        "appearances",
    ]

    preferred_order = [
        column
        for column in preferred_order
        if column in enriched.columns
    ]

    remaining_columns = [
        column
        for column in enriched.columns
        if column not in preferred_order
    ]

    enriched = enriched[
        preferred_order + remaining_columns
    ]

    enriched.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved {len(enriched):,} enriched player-season rows"
    )
    print(OUTPUT_FILE)

    print()
    print(
        enriched["position_group"]
        .value_counts(dropna=False)
    )

    missing_metadata = (
        enriched["player_name"].isna().sum()
        if "player_name" in enriched.columns
        else len(enriched)
    )

    print()
    print(
        f"Rows missing player name: {missing_metadata:,}"
    )


if __name__ == "__main__":
    main()
