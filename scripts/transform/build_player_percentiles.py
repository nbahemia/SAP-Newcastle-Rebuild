from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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


# --------------------------------------------------
# Match duration
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
# Timestamp helper
# --------------------------------------------------

def timestamp_to_seconds(timestamp):

    if timestamp is None or pd.isna(timestamp):
        return None

    minutes, seconds = timestamp.split(":")

    return int(minutes) * 60 + int(seconds)


# --------------------------------------------------
# Normalize positions
# --------------------------------------------------

def normalize_positions(positions):

    if isinstance(positions, np.ndarray):
        return positions.tolist()

    if isinstance(positions, list):
        return positions

    return []


# --------------------------------------------------
# Broad position mapping
# --------------------------------------------------

POSITION_GROUP_MAP = {

    # Goalkeepers
    "Goalkeeper": "GK",

    # Centre backs
    "Center Back": "CB",
    "Left Center Back": "CB",
    "Right Center Back": "CB",

    # Fullbacks / wingbacks
    "Left Back": "FB/WB",
    "Right Back": "FB/WB",
    "Left Wing Back": "FB/WB",
    "Right Wing Back": "FB/WB",

    # Central / defensive midfielders
    "Center Defensive Midfield": "DM/CM",
    "Left Defensive Midfield": "DM/CM",
    "Right Defensive Midfield": "DM/CM",
    "Center Midfield": "DM/CM",
    "Left Center Midfield": "DM/CM",
    "Right Center Midfield": "DM/CM",

    # Attacking midfield / wide players
    "Center Attacking Midfield": "AM",
    "Left Attacking Midfield": "AM",
    "Right Attacking Midfield": "AM",

    "Left Midfield": "W",
    "Right Midfield": "W",

    "Left Wing": "W",
    "Right Wing": "W",

    # Strikers
    "Center Forward": "ST",
    "Left Center Forward": "ST",
    "Right Center Forward": "ST",
    "Secondary Striker": "ST",
}


# --------------------------------------------------
# Calculate position minutes
# --------------------------------------------------

position_rows = []

for _, row in lineups.iterrows():

    match_id = row["match_id"]
    player_id = row["player_id"]
    player_name = row["player_name"]
    team = row["team"]

    positions = normalize_positions(
        row["positions"]
    )

    if len(positions) == 0:
        continue

    match_duration = get_match_duration(
        match_id
    )

    for stint in positions:

        position = stint.get("position")

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

        start = min(
            start,
            match_duration
        )

        end = min(
            end,
            match_duration
        )

        seconds = max(
            0,
            end - start
        )

        minutes = seconds / 60

        position_group = POSITION_GROUP_MAP.get(
            position,
            "OTHER"
        )

        position_rows.append(
            {
                "match_id": match_id,
                "team": team,
                "player_id": player_id,
                "player": player_name,
                "position": position,
                "position_group": position_group,
                "minutes": minutes,
            }
        )


position_minutes = pd.DataFrame(
    position_rows
)


# --------------------------------------------------
# Inspect unmapped positions
# --------------------------------------------------

unmapped = (
    position_minutes[
        position_minutes["position_group"] == "OTHER"
    ]["position"]
    .value_counts()
)

if len(unmapped) > 0:

    print()
    print("Unmapped positions:")
    print(unmapped)


# --------------------------------------------------
# Aggregate player minutes by broad position
# --------------------------------------------------

player_position_minutes = (
    position_minutes
    .groupby(
        [
            "team",
            "player_id",
            "player",
            "position_group",
        ],
        as_index=False
    )["minutes"]
    .sum()
)


# --------------------------------------------------
# Determine primary position group
# --------------------------------------------------

primary_positions = (
    player_position_minutes
    .sort_values(
        [
            "team",
            "player",
            "minutes",
        ],
        ascending=[
            True,
            True,
            False,
        ]
    )
    .drop_duplicates(
        subset=[
            "team",
            "player",
        ]
    )
    .rename(
        columns={
            "minutes": "position_minutes"
        }
    )
)


# --------------------------------------------------
# Calculate percentage of minutes in primary group
# --------------------------------------------------

total_minutes = (
    player_position_minutes
    .groupby(
        [
            "team",
            "player",
        ],
        as_index=False
    )["minutes"]
    .sum()
    .rename(
        columns={
            "minutes": "total_position_minutes"
        }
    )
)


primary_positions = primary_positions.merge(
    total_minutes,
    on=[
        "team",
        "player",
    ],
    how="left"
)


primary_positions["position_share"] = (
    primary_positions["position_minutes"]
    / primary_positions["total_position_minutes"]
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_positions.parquet"
)

primary_positions.to_parquet(
    output_path,
    index=False
)


print()
print(
    f"Saved {len(primary_positions):,} player positions"
)

print(
    f"Output: {output_path}"
)

print()


# --------------------------------------------------
# Preview
# --------------------------------------------------

print(
    primary_positions[
        [
            "team",
            "player",
            "position_group",
            "position_minutes",
            "total_position_minutes",
            "position_share",
        ]
    ]
    .sort_values(
        "total_position_minutes",
        ascending=False
    )
    .head(40)
    .to_string(
        index=False,
        formatters={
            "position_minutes": "{:.1f}".format,
            "total_position_minutes": "{:.1f}".format,
            "position_share": "{:.1%}".format,
        }
    )
)