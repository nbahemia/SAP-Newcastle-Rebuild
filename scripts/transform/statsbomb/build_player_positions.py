from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW = PROJECT_ROOT / "data" / "raw" / "statsbomb"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "statsbomb"

LINEUPS_FILE = RAW / "lineups.parquet"
MINUTES_FILE = PROCESSED / "player_minutes.parquet"
SEASON_FILE = PROCESSED / "player_season_metrics.parquet"

POSITION_MINUTES_OUTPUT = (
    PROCESSED /  "player_position_minutes.parquet"
)

SEASON_WITH_POSITIONS_OUTPUT = (
    PROCESSED / "player_season_metrics_with_positions.parquet"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MIN_SECONDARY_SHARE = 10.0


# ---------------------------------------------------------
# Position Group Mapping
# ---------------------------------------------------------

POSITION_GROUP_MAP = {
    # GK
    "Goalkeeper": "GK",

    # CB
    "Center Back": "CB",
    "Left Center Back": "CB",
    "Right Center Back": "CB",

    # FB / WB
    "Left Back": "FB/WB",
    "Right Back": "FB/WB",
    "Left Wing Back": "FB/WB",
    "Right Wing Back": "FB/WB",

    # DM / CM
    "Center Defensive Midfield": "DM/CM",
    "Left Defensive Midfield": "DM/CM",
    "Right Defensive Midfield": "DM/CM",
    "Center Midfield": "DM/CM",
    "Left Center Midfield": "DM/CM",
    "Right Center Midfield": "DM/CM",

    # AM
    "Center Attacking Midfield": "AM",
    "Left Attacking Midfield": "AM",
    "Right Attacking Midfield": "AM",

    # W
    "Left Midfield": "W",
    "Right Midfield": "W",
    "Left Wing": "W",
    "Right Wing": "W",

    # ST
    "Center Forward": "ST",
    "Left Center Forward": "ST",
    "Right Center Forward": "ST",
    "Secondary Striker": "ST",
}


# ---------------------------------------------------------
# Position Side Mapping
# ---------------------------------------------------------
#
# We mainly care about this for FB/WB scouting,
# but it is useful to preserve generally.
# ---------------------------------------------------------

POSITION_SIDE_MAP = {
    # Left-sided defensive positions
    "Left Back": "Left",
    "Left Wing Back": "Left",
    "Left Center Back": "Left",

    # Right-sided defensive positions
    "Right Back": "Right",
    "Right Wing Back": "Right",
    "Right Center Back": "Right",

    # Central defensive positions
    "Center Back": "Center",
    "Goalkeeper": "Center",

    # Left midfield / attacking positions
    "Left Defensive Midfield": "Left",
    "Left Center Midfield": "Left",
    "Left Attacking Midfield": "Left",
    "Left Midfield": "Left",
    "Left Wing": "Left",
    "Left Center Forward": "Left",

    # Right midfield / attacking positions
    "Right Defensive Midfield": "Right",
    "Right Center Midfield": "Right",
    "Right Attacking Midfield": "Right",
    "Right Midfield": "Right",
    "Right Wing": "Right",
    "Right Center Forward": "Right",

    # Central midfield / attack
    "Center Defensive Midfield": "Center",
    "Center Midfield": "Center",
    "Center Attacking Midfield": "Center",
    "Center Forward": "Center",
    "Secondary Striker": "Center",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def parse_clock(value):
    """
    Convert StatsBomb lineup clock value into seconds.

    Examples:
    '64:30' -> 3870
    '90:00' -> 5400
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value)

    parts = value.split(":")

    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])

        return minutes * 60 + seconds

    return np.nan


def normalize_positions(value):
    """
    Ensure the positions field is always iterable.
    """

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    return []


def get_match_duration(player_minutes_match):
    """
    Infer standardized match duration from corrected
    player-minutes data.
    """

    max_minutes = player_minutes_match["minutes"].max()

    if pd.isna(max_minutes):
        return 90 * 60

    if max_minutes > 90:
        return 120 * 60

    return 90 * 60


# ---------------------------------------------------------
# Build Raw Position Intervals
# ---------------------------------------------------------

def build_position_intervals(
    lineups: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for _, row in lineups.iterrows():

        match_id = row["match_id"]
        team = row["team"]
        player = row["player"]

        positions = normalize_positions(row["positions"])

        for stint in positions:

            position = stint.get("position")

            if position is None:
                continue

            start = parse_clock(
                stint.get("from")
            )

            end = parse_clock(
                stint.get("to")
            )

            if pd.isna(start):
                continue

            rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player": player,
                    "position": position,
                    "position_group": POSITION_GROUP_MAP.get(
                        position,
                        "Other",
                    ),
                    "position_side": POSITION_SIDE_MAP.get(
                        position,
                        "Center",
                    ),
                    "start_seconds": start,
                    "end_seconds": end,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Clip Position Stints to Actual Playing Window
# ---------------------------------------------------------

def clip_intervals_to_player_minutes(
    intervals: pd.DataFrame,
    player_minutes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clip raw StatsBomb position stints to each player's
    corrected playing window.

    Uses:
    - earliest observed position start as player entry
    - corrected total minutes from player_minutes
    - standardized match duration derived per match
    """

    results = []

    # -----------------------------------------------------
    # Determine standardized duration for each match
    # -----------------------------------------------------
    #
    # If any player in the match has > 90 minutes,
    # treat it as a 120-minute match.
    #
    # Otherwise treat it as 90 minutes.
    # -----------------------------------------------------

    match_durations = (
        player_minutes
        .groupby("match_id")["minutes"]
        .max()
        .apply(
            lambda x: 120 * 60
            if pd.notna(x) and x > 90
            else 90 * 60
        )
        .to_dict()
    )

    # -----------------------------------------------------
    # One corrected minutes row per player-match
    # -----------------------------------------------------

    grouped_minutes = player_minutes.groupby(
        [
            "match_id",
            "team",
            "player",
        ]
    )

    for keys, minute_group in grouped_minutes:

        match_id, team, player = keys

        # There should normally be one row,
        # but summing/first guards against duplicates.
        total_minutes = minute_group["minutes"].iloc[0]

        if pd.isna(total_minutes):
            continue

        player_intervals = intervals[
            (intervals["match_id"] == match_id)
            & (intervals["team"] == team)
            & (intervals["player"] == player)
        ].copy()

        if player_intervals.empty:
            continue

        # Earliest observed position stint = entry time
        player_entry = (
            player_intervals["start_seconds"]
            .min()
        )

        if pd.isna(player_entry):
            continue

        # Corrected player exit
        actual_exit = (
            player_entry
            + float(total_minutes) * 60
        )

        # Match duration
        match_duration = match_durations.get(
            match_id,
            90 * 60,
        )

        actual_exit = min(
            actual_exit,
            match_duration,
        )

        # -------------------------------------------------
        # Clip each positional stint to actual player window
        # -------------------------------------------------

        player_intervals["clipped_start"] = (
            player_intervals["start_seconds"]
            .clip(lower=player_entry)
        )

        player_intervals["clipped_end"] = (
            player_intervals["end_seconds"]
            .fillna(actual_exit)
            .clip(upper=actual_exit)
        )

        # Remove invalid / zero-length intervals
        player_intervals = player_intervals[
            player_intervals["clipped_end"]
            > player_intervals["clipped_start"]
        ].copy()

        if not player_intervals.empty:
            results.append(player_intervals)

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Remove Overlapping Position Intervals
# ---------------------------------------------------------

def remove_overlaps(
    intervals: pd.DataFrame
) -> pd.DataFrame:
    """
    Prevent duplicate/tactical-shift position intervals
    from inflating minutes.

    For each player-match, process stints chronologically
    and only count uncovered time.
    """

    output_rows = []

    group_cols = [
        "match_id",
        "team",
        "player",
    ]

    for keys, group in intervals.groupby(group_cols):

        group = group.sort_values(
            ["clipped_start", "clipped_end"]
        )

        covered_until = None

        for _, row in group.iterrows():

            start = row["clipped_start"]
            end = row["clipped_end"]

            if covered_until is None:
                adjusted_start = start

            else:
                adjusted_start = max(
                    start,
                    covered_until,
                )

            if end <= adjusted_start:
                continue

            duration = (
                end - adjusted_start
            ) / 60

            output_rows.append(
                {
                    "match_id": keys[0],
                    "team": keys[1],
                    "player": keys[2],
                    "position": row["position"],
                    "position_group": row[
                        "position_group"
                    ],
                    "position_side": row[
                        "position_side"
                    ],
                    "minutes": duration,
                }
            )

            covered_until = max(
                covered_until or end,
                end,
            )

    return pd.DataFrame(output_rows)


# ---------------------------------------------------------
# Aggregate Position Usage
# ---------------------------------------------------------

def build_position_minutes(
    cleaned_intervals: pd.DataFrame
) -> pd.DataFrame:

    position_minutes = (
        cleaned_intervals
        .groupby(
            [
                "team",
                "player",
                "position_group",
                "position_side",
            ],
            as_index=False,
        )["minutes"]
        .sum()
    )

    return position_minutes


# ---------------------------------------------------------
# Assign Primary / Secondary Position Group
# ---------------------------------------------------------

def assign_position_groups(
    position_minutes: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for (team, player), group in position_minutes.groupby(
        ["team", "player"]
    ):

        group_totals = (
            group
            .groupby(
                "position_group",
                as_index=False,
            )["minutes"]
            .sum()
            .sort_values(
                "minutes",
                ascending=False,
            )
        )

        total_minutes = group_totals["minutes"].sum()

        if total_minutes <= 0:
            continue

        primary = group_totals.iloc[0]

        primary_group = (
            primary["position_group"]
        )

        primary_minutes = (
            primary["minutes"]
        )

        primary_share = (
            primary_minutes
            / total_minutes
            * 100
        )

        secondary_group = None
        secondary_minutes = 0.0
        secondary_share = 0.0

        if len(group_totals) > 1:

            secondary = group_totals.iloc[1]

            secondary_group = (
                secondary["position_group"]
            )

            secondary_minutes = (
                secondary["minutes"]
            )

            secondary_share = (
                secondary_minutes
                / total_minutes
                * 100
            )

            if secondary_share < MIN_SECONDARY_SHARE:
                secondary_group = None

        rows.append(
            {
                "team": team,
                "player": player,
                "primary_position_group": primary_group,
                "primary_position_minutes": round(
                    primary_minutes,
                    1,
                ),
                "primary_position_share": round(
                    primary_share,
                    1,
                ),
                "secondary_position_group": secondary_group,
                "secondary_position_minutes": round(
                    secondary_minutes,
                    1,
                ),
                "secondary_position_share": round(
                    secondary_share,
                    1,
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Assign Primary Position Side
# ---------------------------------------------------------

def assign_position_sides(
    position_minutes: pd.DataFrame
) -> pd.DataFrame:
    """
    Determine the side on which each player spent the
    most observed minutes.

    Example:
    Left  = 310 min
    Right = 20 min
    Center = 0 min

    -> primary_position_side = Left
    """

    rows = []

    for (team, player), group in position_minutes.groupby(
        ["team", "player"]
    ):

        side_totals = (
            group
            .groupby(
                "position_side",
                as_index=False,
            )["minutes"]
            .sum()
            .sort_values(
                "minutes",
                ascending=False,
            )
        )

        total_minutes = (
            side_totals["minutes"].sum()
        )

        if total_minutes <= 0:
            continue

        primary = side_totals.iloc[0]

        primary_side = (
            primary["position_side"]
        )

        primary_side_minutes = (
            primary["minutes"]
        )

        primary_side_share = (
            primary_side_minutes
            / total_minutes
            * 100
        )

        rows.append(
            {
                "team": team,
                "player": player,
                "primary_position_side": primary_side,
                "primary_position_side_minutes": round(
                    primary_side_minutes,
                    1,
                ),
                "primary_position_side_share": round(
                    primary_side_share,
                    1,
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_column_names(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize StatsBomb lineup column names to the naming
    convention used throughout the processed pipeline.
    """

    lineups = lineups.copy()

    rename_map = {}

    if "player" not in lineups.columns:
        if "player_name" in lineups.columns:
            rename_map["player_name"] = "player"

    if "team" not in lineups.columns:
        if "team_name" in lineups.columns:
            rename_map["team_name"] = "team"

    if rename_map:
        lineups = lineups.rename(
            columns=rename_map
        )

    required = [
        "match_id",
        "team",
        "player",
        "positions",
    ]

    missing = [
        col
        for col in required
        if col not in lineups.columns
    ]

    if missing:
        raise ValueError(
            "Lineups file is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(lineups.columns.astype(str))
        )

    return lineups

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading data...")

    lineups = pd.read_parquet(
        LINEUPS_FILE
    )

    player_minutes = pd.read_parquet(
        MINUTES_FILE
    )

    season_metrics = pd.read_parquet(
        SEASON_FILE
    )

    # Normalize StatsBomb naming
    lineups = normalize_column_names(
        lineups
    )

    print(
        f"Loaded {len(lineups):,} lineup rows"
    )

    print(
        f"Loaded {len(player_minutes):,} player-minute rows"
    )

    print(
        f"Loaded {len(season_metrics):,} player-season rows"
    )

    print("\nLineup columns:")
    print(lineups.columns.tolist())

    # ---------------------------------------------
    # Build raw intervals
    # ---------------------------------------------

    intervals = build_position_intervals(
        lineups
    )

    print(
        f"\nBuilt {len(intervals):,} raw position intervals"
    )

    # ---------------------------------------------
    # Clip to actual playing window
    # ---------------------------------------------

    intervals = clip_intervals_to_player_minutes(
        intervals,
        player_minutes,
    )

    print(
        f"Retained {len(intervals):,} clipped intervals"
    )

    # ---------------------------------------------
    # Remove overlaps
    # ---------------------------------------------

    cleaned = remove_overlaps(
        intervals
    )

    print(
        f"Retained {len(cleaned):,} non-overlapping intervals"
    )

    # ---------------------------------------------
    # Position-minute aggregation
    # ---------------------------------------------

    position_minutes = build_position_minutes(
        cleaned
    )

    # ---------------------------------------------
    # Position group assignments
    # ---------------------------------------------

    position_groups = assign_position_groups(
        position_minutes
    )

    # ---------------------------------------------
    # Position side assignments
    # ---------------------------------------------

    position_sides = assign_position_sides(
        position_minutes
    )

    # ---------------------------------------------
    # Combine position metadata
    # ---------------------------------------------

    position_summary = position_groups.merge(
        position_sides,
        on=[
            "team",
            "player",
        ],
        how="left",
    )

    # ---------------------------------------------
    # Merge onto season metrics
    # ---------------------------------------------

    season_with_positions = (
        season_metrics
        .merge(
            position_summary,
            on=[
                "team",
                "player",
            ],
            how="left",
        )
    )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    position_minutes.to_parquet(
        POSITION_MINUTES_OUTPUT,
        index=False,
    )

    season_with_positions.to_parquet(
        SEASON_WITH_POSITIONS_OUTPUT,
        index=False,
    )

    print(
        f"\nSaved position minutes to:"
    )
    print(POSITION_MINUTES_OUTPUT)

    print(
        f"\nSaved season metrics with positions to:"
    )
    print(SEASON_WITH_POSITIONS_OUTPUT)

    # ---------------------------------------------
    # Diagnostics
    # ---------------------------------------------

    print(
        "\nPrimary position groups:"
    )

    print(
        season_with_positions[
            "primary_position_group"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nPrimary position sides:"
    )

    print(
        season_with_positions[
            "primary_position_side"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ---------------------------------------------
    # FB/WB side sanity check
    # ---------------------------------------------

    fb = season_with_positions[
        season_with_positions[
            "primary_position_group"
        ] == "FB/WB"
    ][
        [
            "player",
            "team",
            "minutes",
            "primary_position_group",
            "primary_position_side",
            "primary_position_side_share",
            "secondary_position_group",
        ]
    ].sort_values(
        "minutes",
        ascending=False,
    )

    print(
        "\nFB/WB side sanity check:\n"
    )

    print(
        fb.head(30)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()