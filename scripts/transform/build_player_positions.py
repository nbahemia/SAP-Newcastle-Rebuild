from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MIN_SECONDARY_SHARE = 10.0
# --------------------------------------------------
# Load data
# --------------------------------------------------

lineups = pd.read_parquet(
    RAW_DATA_DIR / "lineups.parquet"
)

events = pd.read_parquet(
    RAW_DATA_DIR / "events.parquet"
)

player_season = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_season_metrics.parquet"
)

print(f"Loaded {len(lineups):,} lineup rows")
print(f"Loaded {len(player_season):,} player-season rows")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_positions(positions):
    if isinstance(positions, np.ndarray):
        return positions.tolist()

    if isinstance(positions, list):
        return positions

    return []


def timestamp_to_seconds(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None

    minutes, seconds = timestamp.split(":")

    return int(minutes) * 60 + int(seconds)


# --------------------------------------------------
# Determine standardized match duration
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
# Map StatsBomb positions into broader groups
# --------------------------------------------------

def position_group(position):

    if position == "Goalkeeper":
        return "GK"

    if position in [
        "Center Back",
        "Left Center Back",
        "Right Center Back",
    ]:
        return "CB"

    if position in [
        "Left Back",
        "Right Back",
        "Left Wing Back",
        "Right Wing Back",
    ]:
        return "FB/WB"

    if position in [
        "Center Defensive Midfield",
        "Left Defensive Midfield",
        "Right Defensive Midfield",
    ]:
        return "DM"
    
    if position in [
        "Center Midfield",
        "Left Center Midfield",
        "Right Center Midfield",
    ]:
        return "CM"

    if position in [
        "Center Attacking Midfield",
        "Left Attacking Midfield",
        "Right Attacking Midfield",
    ]:
        return "AM"

    if position in [
        "Left Midfield",
        "Right Midfield",
        "Left Wing",
        "Right Wing",
    ]:
        return "W"

    if position in [
        "Center Forward",
        "Left Center Forward",
        "Right Center Forward",
        "Secondary Striker",
    ]:
        return "ST"

    return "Other"


# --------------------------------------------------
# Calculate minutes spent in each position
# --------------------------------------------------

position_rows = []

for _, row in lineups.iterrows():

    match_id = row["match_id"]
    team = row["team"]
    player = row["player_name"]

    positions = normalize_positions(
        row["positions"]
    )

    # Empty positions means player never entered
    if len(positions) == 0:
        continue

    match_duration = get_match_duration(
        match_id
    )

    for stint in positions:

        detailed_position = stint.get(
            "position"
        )

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

        if seconds == 0:
            continue

        position_rows.append(
            {
                "match_id": match_id,
                "team": team,
                "player": player,
                "position": detailed_position,
                "position_group": position_group(
                    detailed_position
                ),
                "position_minutes": seconds / 60,
            }
        )


position_stints = pd.DataFrame(
    position_rows
)


# --------------------------------------------------
# Aggregate minutes by player and position group
# --------------------------------------------------

player_position_minutes = (
    position_stints
    .groupby(
        [
            "team",
            "player",
            "position_group",
        ],
        as_index=False,
    )
    ["position_minutes"]
    .sum()
)


# --------------------------------------------------
# Sort position groups by minutes played
# --------------------------------------------------

player_position_minutes = (
    player_position_minutes
    .sort_values(
        [
            "team",
            "player",
            "position_minutes",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
)


# --------------------------------------------------
# Rank position groups for each player
# --------------------------------------------------

player_position_minutes[
    "position_rank"
] = (
    player_position_minutes
    .groupby(
        [
            "team",
            "player",
        ]
    )
    ["position_minutes"]
    .rank(
        method="first",
        ascending=False,
    )
)


# --------------------------------------------------
# Calculate total position minutes
# --------------------------------------------------

total_position_minutes = (
    player_position_minutes
    .groupby(
        [
            "team",
            "player",
        ],
        as_index=False,
    )
    ["position_minutes"]
    .sum()
    .rename(
        columns={
            "position_minutes":
                "total_position_minutes"
        }
    )
)


player_position_minutes = (
    player_position_minutes
    .merge(
        total_position_minutes,
        on=[
            "team",
            "player",
        ],
        how="left",
    )
)


# --------------------------------------------------
# Calculate position share
# --------------------------------------------------

player_position_minutes[
    "position_share"
] = (
    player_position_minutes[
        "position_minutes"
    ]
    / player_position_minutes[
        "total_position_minutes"
    ]
    * 100
)


# --------------------------------------------------
# Primary position
# --------------------------------------------------

primary_positions = (
    player_position_minutes[
        player_position_minutes[
            "position_rank"
        ] == 1
    ]
    [
        [
            "team",
            "player",
            "position_group",
            "position_minutes",
            "position_share",
        ]
    ]
    .rename(
        columns={
            "position_group":
                "primary_position_group",

            "position_minutes":
                "primary_position_minutes",

            "position_share":
                "primary_position_share",
        }
    )
)


# --------------------------------------------------
# Secondary position
# --------------------------------------------------


secondary_positions = (
    player_position_minutes[
        player_position_minutes[
            "position_rank"
        ] == 2
    ]
    [
        [
            "team",
            "player",
            "position_group",
            "position_minutes",
            "position_share",
        ]
    ]
    .rename(
        columns={
            "position_group":
                "secondary_position_group",

            "position_minutes":
                "secondary_position_minutes",

            "position_share":
                "secondary_position_share",
        }
    )
)

secondary_positions = secondary_positions[
    secondary_positions["secondary_position_share"] >= MIN_SECONDARY_SHARE
].copy()

# --------------------------------------------------
# Combine primary and secondary positions
# --------------------------------------------------

player_positions = (
    primary_positions
    .merge(
        secondary_positions,
        on=[
            "team",
            "player",
        ],
        how="left",
    )
)


# --------------------------------------------------
# Merge position information into player-season metrics
# --------------------------------------------------

player_season_with_positions = (
    player_season
    .merge(
        player_positions,
        on=[
            "team",
            "player",
        ],
        how="left",
    )
)


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

player_position_minutes.to_parquet(
    PROCESSED_DATA_DIR
    / "player_position_minutes.parquet",
    index=False,
)

player_season_with_positions.to_parquet(
    PROCESSED_DATA_DIR
    / "player_season_metrics_with_positions.parquet",
    index=False,
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print()

print(
    f"Saved {len(player_position_minutes):,} "
    "player-position rows"
)

print(
    f"Saved {len(player_season_with_positions):,} "
    "player-season rows with positions"
)

print()


# --------------------------------------------------
# Preview
# --------------------------------------------------

display_columns = [
    "team",
    "player",
    "primary_position_group",
    "secondary_position_group",
    "minutes",
    "primary_position_minutes",
    "primary_position_share",
    "secondary_position_minutes",
    "secondary_position_share",
]

print(
    player_season_with_positions[
        display_columns
    ]
    .sort_values(
        "minutes",
        ascending=False
    )
    .head(40)
    .to_string(
        index=False,
        formatters={
            "minutes":
                "{:.1f}".format,

            "primary_position_minutes":
                lambda x: (
                    f"{x:.1f}"
                    if pd.notna(x)
                    else ""
                ),

            "primary_position_share":
                lambda x: (
                    f"{x:.1f}%"
                    if pd.notna(x)
                    else ""
                ),

            "secondary_position_minutes":
                lambda x: (
                    f"{x:.1f}"
                    if pd.notna(x)
                    else ""
                ),

            "secondary_position_share":
                lambda x: (
                    f"{x:.1f}%"
                    if pd.notna(x)
                    else ""
                ),
        }
    )
)