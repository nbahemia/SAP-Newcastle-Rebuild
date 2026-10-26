from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

lineups = pd.read_parquet(
    RAW_DATA_DIR / "lineups.parquet"
)

events = pd.read_parquet(
    RAW_DATA_DIR / "events.parquet"
)

print(f"Loaded {len(lineups):,} lineup rows")
print(f"Loaded {len(events):,} events")


# --------------------------------------------------
# Convert MM:SS into total seconds
# --------------------------------------------------

def timestamp_to_seconds(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None

    minutes, seconds = timestamp.split(":")

    return int(minutes) * 60 + int(seconds)


# --------------------------------------------------
# Determine actual match ending time
# --------------------------------------------------

match_max_period = (
    events
    .groupby("match_id")["period"]
    .max()
    .to_dict()
)


def get_match_duration(match_id):
    max_period = match_max_period[match_id]

    if max_period <= 2:
        return 90 * 60

    return 120 * 60

# --------------------------------------------------
# Normalize StatsBomb positions field
# --------------------------------------------------

def normalize_positions(positions):

    if isinstance(positions, np.ndarray):
        return positions.tolist()

    if isinstance(positions, list):
        return positions

    return []


# --------------------------------------------------
# Calculate player minutes
# --------------------------------------------------

player_minutes_rows = []

for _, row in lineups.iterrows():

    match_id = row["match_id"]
    player_id = row["player_id"]
    player_name = row["player_name"]
    team = row["team"]

    positions = normalize_positions(
        row["positions"]
    )

    # Empty positions means the player never entered
    if len(positions) == 0:
        continue

    total_seconds = 0
    match_duration = get_match_duration(match_id)

    for stint in positions:

        start = timestamp_to_seconds(
            stint.get("from")
        )

        end_timestamp = stint.get("to")

        if end_timestamp is None:
            end = match_duration
        else:
            end = timestamp_to_seconds(
                end_timestamp
            )

        if start is None or end is None:
            continue

        start = min(start, match_duration)
        end = min(end, match_duration)

        total_seconds += max(
            0,
            end - start
        )

    minutes = total_seconds / 60

    player_minutes_rows.append(
        {
            "match_id": match_id,
            "team": team,
            "player_id": player_id,
            "player": player_name,
            "minutes": minutes,
        }
    )
   
player_minutes = pd.DataFrame(
    player_minutes_rows
)


# --------------------------------------------------
# Save
# --------------------------------------------------

player_minutes.to_parquet(
    PROCESSED_DATA_DIR / "player_minutes.parquet",
    index=False
)


print(
    f"Saved {len(player_minutes):,} player-match minute rows"
)

print()
print(
    player_minutes
    .sort_values(
        ["match_id", "team", "minutes"],
        ascending=[True, True, False]
    )
    .head(30)
    .to_string(index=False)
)