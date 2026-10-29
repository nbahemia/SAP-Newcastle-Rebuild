from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "statsbomb"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

lineups = pd.read_parquet(
    RAW_DATA_DIR / "lineups.parquet"
)

events = pd.read_parquet(
    RAW_DATA_DIR /  "events.parquet"
)

print(f"Loaded {len(lineups):,} lineup rows")
print(f"Loaded {len(events):,} events")


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
# Standardized match duration
# --------------------------------------------------

match_max_period = (
    events
    .groupby("match_id")["period"]
    .max()
    .to_dict()
)


def get_match_duration(match_id):

    max_period = match_max_period[match_id]

    # Regulation match
    if max_period <= 2:
        return 90 * 60

    # Extra-time match
    return 120 * 60


# --------------------------------------------------
# Determine whether a position ending means the
# player actually left the match
# --------------------------------------------------

def is_terminal_exit(reason):

    if reason is None or pd.isna(reason):
        return False

    reason = str(reason)

    # Tactical position changes do not mean
    # the player left the match.
    if reason == "Tactical Shift":
        return False

    # StatsBomb can record temporary player-off
    # sequences. Do not treat these as substitutions.
    if reason == "Player Off":
        return False

    # Final whistle is handled using standardized
    # match duration.
    if reason == "Final Whistle":
        return False

    # Actual substitution
    if reason.startswith("Substitution - Off"):
        return True

    # Defensive handling for dismissals
    if "Red Card" in reason:
        return True

    if "Second Yellow" in reason:
        return True

    return False


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

    # Empty positions = unused substitute
    if len(positions) == 0:
        continue

    match_duration = get_match_duration(
        match_id
    )

    # --------------------------------------------------
    # Find when the player first entered
    # --------------------------------------------------

    start_times = []

    for stint in positions:

        start = timestamp_to_seconds(
            stint.get("from")
        )

        if start is not None:
            start_times.append(start)

    if len(start_times) == 0:
        continue

    entry_time = min(start_times)

    entry_time = min(
        entry_time,
        match_duration,
    )


    # --------------------------------------------------
    # Find actual exit from the match
    # --------------------------------------------------

    terminal_exit_times = []

    for stint in positions:

        end_reason = stint.get(
            "end_reason"
        )

        if not is_terminal_exit(
            end_reason
        ):
            continue

        end = timestamp_to_seconds(
            stint.get("to")
        )

        if end is None:
            continue

        # Ignore corrupted/irrelevant exit timestamps
        # that occur before the player's real entry.
        if end < entry_time:
            continue

        terminal_exit_times.append(
            end
        )


    # --------------------------------------------------
    # If never substituted/dismissed, player remained
    # until standardized final whistle
    # --------------------------------------------------

    if len(terminal_exit_times) > 0:

        exit_time = min(
            terminal_exit_times
        )

        exit_time = min(
            exit_time,
            match_duration,
        )

    else:

        exit_time = match_duration


    # --------------------------------------------------
    # Calculate minutes
    # --------------------------------------------------

    total_seconds = max(
        0,
        exit_time - entry_time
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
# Sanity checks
# --------------------------------------------------

invalid_minutes = player_minutes[
    player_minutes["minutes"] > 120
]

print()
print(
    f"Rows over 120 minutes: "
    f"{len(invalid_minutes):,}"
)

if len(invalid_minutes) > 0:

    print(
        invalid_minutes
        .sort_values(
            "minutes",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


zero_minutes = player_minutes[
    player_minutes["minutes"] <= 0
]

print(
    f"Rows at or below 0 minutes: "
    f"{len(zero_minutes):,}"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_minutes.parquet"
)

player_minutes.to_parquet(
    output_path,
    index=False
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print()

print(
    f"Saved {len(player_minutes):,} "
    "player-match minute rows"
)

print(
    f"Output: {output_path}"
)

print()

print(
    player_minutes
    .sort_values(
        [
            "match_id",
            "team",
            "minutes",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .head(30)
    .to_string(
        index=False,
        formatters={
            "minutes":
                "{:.1f}".format,
        },
    )
)

print(
    f"Rows at or below 0 minutes: "
    f"{len(zero_minutes):,}"
)

if len(zero_minutes) > 0:

    print()
    print("Zero-minute rows:")

    print(
        zero_minutes[
            [
                "match_id",
                "team",
                "player_id",
                "player",
                "minutes",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )