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

player_minutes = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_minutes.parquet"
)

player_season = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_season_metrics.parquet"
)

print(f"Loaded {len(lineups):,} lineup rows")
print(f"Loaded {len(player_minutes):,} player-minute rows")
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

    return (
        int(minutes) * 60
        + int(seconds)
    )


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
# Position mapping
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
        "Center Midfield",
        "Left Center Midfield",
        "Right Center Midfield",
    ]:
        return "DM/CM"

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
# Player match-minute lookup
# --------------------------------------------------

minute_lookup = (
    player_minutes
    .set_index(
        [
            "match_id",
            "team",
            "player",
        ]
    )
    ["minutes"]
    .to_dict()
)


# --------------------------------------------------
# Build raw position intervals
# --------------------------------------------------

interval_rows = []

for _, row in lineups.iterrows():

    match_id = row["match_id"]
    team = row["team"]
    player = row["player_name"]

    key = (
        match_id,
        team,
        player,
    )

    if key not in minute_lookup:
        continue

    total_player_minutes = minute_lookup[key]

    if pd.isna(total_player_minutes):
        continue

    positions = normalize_positions(
        row["positions"]
    )

    if len(positions) == 0:
        continue

    match_duration = get_match_duration(
        match_id
    )

    # Determine player's actual entry time
    start_times = []

    for stint in positions:

        start = timestamp_to_seconds(
            stint.get("from")
        )

        if start is not None:
            start_times.append(start)

    if len(start_times) == 0:
        continue

    player_entry = min(start_times)

    # Use corrected total minutes to infer exit
    player_exit = (
        player_entry
        + total_player_minutes * 60
    )

    player_exit = min(
        player_exit,
        match_duration,
    )

    # Build candidate position intervals
    for stint in positions:

        detailed_position = stint.get(
            "position"
        )

        group = position_group(
            detailed_position
        )

        # Ignore unmapped junk
        if group == "Other":
            continue

        start = timestamp_to_seconds(
            stint.get("from")
        )

        end_timestamp = stint.get("to")

        if start is None:
            continue

        if end_timestamp is None:
            end = player_exit
        else:
            end = timestamp_to_seconds(
                end_timestamp
            )

        if end is None:
            continue

        # Clip to player's actual playing window
        start = max(
            start,
            player_entry
        )

        end = min(
            end,
            player_exit
        )

        if end <= start:
            continue

        interval_rows.append(
            {
                "match_id": match_id,
                "team": team,
                "player": player,
                "position": detailed_position,
                "position_group": group,
                "start": start,
                "end": end,
            }
        )


position_intervals = pd.DataFrame(
    interval_rows
)


# --------------------------------------------------
# Remove overlapping intervals
# --------------------------------------------------

cleaned_rows = []

for (
    match_id,
    team,
    player
), group_df in position_intervals.groupby(
    [
        "match_id",
        "team",
        "player",
    ]
):

    group_df = group_df.sort_values(
        [
            "start",
            "end",
        ]
    )

    occupied_until = None

    for _, stint in group_df.iterrows():

        start = stint["start"]
        end = stint["end"]

        if occupied_until is None:
            effective_start = start

        else:
            effective_start = max(
                start,
                occupied_until
            )

        if end <= effective_start:
            continue

        seconds = (
            end
            - effective_start
        )

        cleaned_rows.append(
            {
                "match_id": match_id,
                "team": team,
                "player": player,
                "position": stint["position"],
                "position_group":
                    stint["position_group"],
                "position_minutes":
                    seconds / 60,
            }
        )

        occupied_until = max(
            end,
            occupied_until
            if occupied_until is not None
            else end,
        )


position_stints = pd.DataFrame(
    cleaned_rows
)


# --------------------------------------------------
# Aggregate by player and position group
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
# Total position minutes
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
# Position share
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
# Rank positions
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


secondary_positions = (
    secondary_positions[
        secondary_positions[
            "secondary_position_share"
        ] >= MIN_SECONDARY_SHARE
    ]
    .copy()
)


# --------------------------------------------------
# Combine primary and secondary
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
# Merge with season metrics
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
# Save
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