from pathlib import Path
import ast

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW = PROJECT_ROOT / "data" / "raw" / "statsbomb"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "statsbomb"

EVENTS_FILE = RAW / "events.parquet"

OUTPUT_FILE = (
    PROCESSED
    / "player_match_raw_metrics.parquet"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def parse_location(value):
    """
    Convert StatsBomb location field into [x, y].

    Handles:
    - list
    - tuple
    - numpy array
    - serialized string
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(
                parsed,
                (list, tuple),
            ) and len(parsed) >= 2:
                return parsed

        except (ValueError, SyntaxError):
            return None

    return None


def get_x(value):

    location = parse_location(value)

    if location is None:
        return np.nan

    return float(location[0])


def get_y(value):

    location = parse_location(value)

    if location is None:
        return np.nan

    return float(location[1])


def parse_end_location(value):
    """
    Same as parse_location, but separated for readability.
    """

    return parse_location(value)


def is_successful_pass(row):
    """
    StatsBomb successful passes usually have null pass_outcome.
    """

    if "pass_outcome" not in row.index:
        return True

    return pd.isna(row["pass_outcome"])


def is_progressive_pass(row):
    """
    Approximate progressive-pass definition already used
    in this project:

    - pass must be successful
    - pass moves toward opponent goal
    - >= 20 StatsBomb units if starting before x=60
    - >= 10 units if starting at/after x=60
    """

    if row.get("type") != "Pass":
        return False

    if not is_successful_pass(row):
        return False

    start = parse_location(
        row.get("location")
    )

    end = parse_end_location(
        row.get("pass_end_location")
    )

    if start is None or end is None:
        return False

    start_x = float(start[0])
    end_x = float(end[0])

    progress = end_x - start_x

    if progress <= 0:
        return False

    if start_x < 60:
        return progress >= 20

    return progress >= 10


def is_pass_into_final_third(row):

    if row.get("type") != "Pass":
        return False

    if not is_successful_pass(row):
        return False

    start = parse_location(
        row.get("location")
    )

    end = parse_end_location(
        row.get("pass_end_location")
    )

    if start is None or end is None:
        return False

    return (
        float(start[0]) < 80
        and float(end[0]) >= 80
    )


def is_progressive_carry(row):
    """
    Progressive carry if the carry reduces distance to the
    opponent goal by at least 10 StatsBomb units.
    """

    if row.get("type") != "Carry":
        return False

    start = parse_location(
        row.get("location")
    )

    end = parse_end_location(
        row.get("carry_end_location")
    )

    if start is None or end is None:
        return False

    start_x = float(start[0])
    end_x = float(end[0])

    return (
        end_x > start_x
        and (end_x - start_x) >= 10
    )


def is_carry_into_final_third(row):

    if row.get("type") != "Carry":
        return False

    start = parse_location(
        row.get("location")
    )

    end = parse_end_location(
        row.get("carry_end_location")
    )

    if start is None or end is None:
        return False

    return (
        float(start[0]) < 80
        and float(end[0]) >= 80
    )


def is_attacking_third_event(row):

    location = parse_location(
        row.get("location")
    )

    if location is None:
        return False

    return float(location[0]) >= 80


def is_shot_in_box(row):

    if row.get("type") != "Shot":
        return False

    location = parse_location(
        row.get("location")
    )

    if location is None:
        return False

    x = float(location[0])
    y = float(location[1])

    return (
        x >= 102
        and 18 <= y <= 62
    )


# ---------------------------------------------------------
# Duel Helpers
# ---------------------------------------------------------

DUEL_WIN_OUTCOMES = {
    "Won",
    "Success",
    "Success In Play",
    "Success Out",
}


def is_duel_won(row):

    if row.get("type") != "Duel":
        return False

    outcome = row.get(
        "duel_outcome"
    )

    if pd.isna(outcome):
        return False

    return str(outcome) in DUEL_WIN_OUTCOMES


def is_aerial_duel(row):
    """
    Use duel_type where StatsBomb explicitly identifies
    an aerial contest.
    """

    if row.get("type") != "Duel":
        return False

    duel_type = row.get(
        "duel_type"
    )

    if pd.isna(duel_type):
        return False

    return (
        "aerial"
        in str(duel_type).lower()
    )


def is_aerial_duel_won(row):

    return (
        is_aerial_duel(row)
        and is_duel_won(row)
    )


# ---------------------------------------------------------
# Player-Match Metric Builder
# ---------------------------------------------------------

def calculate_player_match_metrics(
    player_events: pd.DataFrame,
) -> dict:

    metrics = {}

    # -----------------------------------------------------
    # Passing
    # -----------------------------------------------------

    passes = player_events[
        player_events["type"] == "Pass"
    ].copy()

    if "pass_outcome" in passes.columns:

        successful_passes = passes[
            passes["pass_outcome"].isna()
        ]

    else:

        successful_passes = passes

    metrics["successful_passes"] = (
        len(successful_passes)
    )

    metrics["progressive_passes"] = (
        player_events
        .apply(
            is_progressive_pass,
            axis=1,
        )
        .sum()
    )

    metrics["passes_into_final_third"] = (
        player_events
        .apply(
            is_pass_into_final_third,
            axis=1,
        )
        .sum()
    )

    # -----------------------------------------------------
    # Carrying
    # -----------------------------------------------------

    metrics["progressive_carries"] = (
        player_events
        .apply(
            is_progressive_carry,
            axis=1,
        )
        .sum()
    )

    metrics["carries_into_final_third"] = (
        player_events
        .apply(
            is_carry_into_final_third,
            axis=1,
        )
        .sum()
    )

    # -----------------------------------------------------
    # Pressing
    # -----------------------------------------------------

    pressures = player_events[
        player_events["type"] == "Pressure"
    ].copy()

    metrics["pressures"] = len(
        pressures
    )

    if pressures.empty:

        metrics[
            "attacking_third_pressures"
        ] = 0

    else:

        metrics[
            "attacking_third_pressures"
        ] = (
            pressures
            .apply(
                is_attacking_third_event,
                axis=1,
            )
            .sum()
        )

    # -----------------------------------------------------
    # Recoveries
    # -----------------------------------------------------

    recoveries = player_events[
        player_events["type"]
        == "Ball Recovery"
    ].copy()

    metrics["recoveries"] = len(
        recoveries
    )

    if recoveries.empty:

        metrics[
            "attacking_third_recoveries"
        ] = 0

    else:

        metrics[
            "attacking_third_recoveries"
        ] = (
            recoveries
            .apply(
                is_attacking_third_event,
                axis=1,
            )
            .sum()
        )

    # -----------------------------------------------------
    # Interceptions
    # -----------------------------------------------------

    metrics["interceptions"] = (
        (
            player_events["type"]
            == "Interception"
        )
        .sum()
    )

    # -----------------------------------------------------
    # Duels
    # -----------------------------------------------------

    duels = player_events[
        player_events["type"] == "Duel"
    ].copy()

    metrics["duels"] = len(
        duels
    )

    if duels.empty:

        metrics["duels_won"] = 0
        metrics["aerial_duels"] = 0
        metrics["aerial_duels_won"] = 0

    else:

        metrics["duels_won"] = (
            duels
            .apply(
                is_duel_won,
                axis=1,
            )
            .sum()
        )

        metrics["aerial_duels"] = (
            duels
            .apply(
                is_aerial_duel,
                axis=1,
            )
            .sum()
        )

        metrics["aerial_duels_won"] = (
            duels
            .apply(
                is_aerial_duel_won,
                axis=1,
            )
            .sum()
        )

    # -----------------------------------------------------
    # STRIKER METRICS
    # -----------------------------------------------------

    shots = player_events[
        player_events["type"] == "Shot"
    ].copy()

    metrics["shots"] = len(
        shots
    )

    # -----------------------------------------------------
    # Goals
    # -----------------------------------------------------

    if (
        not shots.empty
        and "shot_outcome" in shots.columns
    ):

        metrics["goals"] = (
            shots["shot_outcome"]
            .eq("Goal")
            .sum()
        )

    else:

        metrics["goals"] = 0

    # -----------------------------------------------------
    # xG
    # -----------------------------------------------------

    if (
        not shots.empty
        and "shot_statsbomb_xg"
        in shots.columns
    ):

        metrics["xg"] = (
            pd.to_numeric(
                shots[
                    "shot_statsbomb_xg"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    else:

        metrics["xg"] = 0.0

    # -----------------------------------------------------
    # Shots in box
    # -----------------------------------------------------

    if shots.empty:

        metrics[
            "shots_in_box"
        ] = 0

    else:

        metrics[
            "shots_in_box"
        ] = (
            shots
            .apply(
                is_shot_in_box,
                axis=1,
            )
            .sum()
        )

    return metrics


# ---------------------------------------------------------
# Build Player-Match Table
# ---------------------------------------------------------

def build_player_match_metrics(
    events: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = [
        "match_id",
        "team",
        "player",
        "type",
    ]

    missing = [
        col
        for col in required_columns
        if col not in events.columns
    ]

    if missing:
        raise ValueError(
            "Events file is missing required columns:\n"
            + "\n".join(missing)
        )

    player_events = events[
        events["player"].notna()
    ].copy()

    rows = []

    group_columns = [
        "match_id",
        "team",
        "player",
    ]

    for (
        match_id,
        team,
        player,
    ), group in player_events.groupby(
        group_columns,
        sort=False,
    ):

        metrics = (
            calculate_player_match_metrics(
                group
            )
        )

        row = {
            "match_id": match_id,
            "team": team,
            "player": player,
        }

        # Preserve player_id when available
        if "player_id" in group.columns:

            ids = (
                group["player_id"]
                .dropna()
            )

            row["player_id"] = (
                ids.iloc[0]
                if not ids.empty
                else np.nan
            )

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

def print_striker_metric_diagnostics(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
):

    print(
        "\nStriker metric totals:"
    )

    diagnostic_columns = [
        "shots",
        "goals",
        "xg",
        "shots_in_box",
        "duels",
        "duels_won",
        "aerial_duels",
        "aerial_duels_won",
    ]

    for column in diagnostic_columns:

        if column not in metrics.columns:
            continue

        print(
            f"{column}: "
            f"{metrics[column].sum():,.2f}"
        )

    # -----------------------------------------------------
    # Duel field diagnostics
    # -----------------------------------------------------

    print(
        "\nAvailable duel columns:"
    )

    duel_columns = [
        col
        for col in [
            "duel_type",
            "duel_outcome",
        ]
        if col in events.columns
    ]

    print(
        duel_columns
    )

    if duel_columns:

        duel_events = events[
            events["type"] == "Duel"
        ][
            duel_columns
        ].copy()

        print(
            "\nDuel value counts:"
        )

        if not duel_events.empty:

            print(
                duel_events
                .value_counts(
                    dropna=False
                )
                .head(30)
                .to_string()
            )

    # -----------------------------------------------------
    # Shot field diagnostics
    # -----------------------------------------------------

    print(
        "\nShot diagnostics:"
    )

    shot_columns = [
        col
        for col in [
            "shot_outcome",
            "shot_statsbomb_xg",
        ]
        if col in events.columns
    ]

    print(
        f"Available shot columns: "
        f"{shot_columns}"
    )

    if "shot_outcome" in events.columns:

        print(
            events.loc[
                events["type"] == "Shot",
                "shot_outcome",
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    events = pd.read_parquet(
        EVENTS_FILE
    )

    print(
        f"Loaded {len(events):,} events"
    )

    print(
        "\nRelevant event columns:"
    )

    interesting_columns = [
        "match_id",
        "team",
        "player",
        "player_id",
        "type",
        "location",
        "pass_outcome",
        "pass_end_location",
        "carry_end_location",
        "shot_outcome",
        "shot_statsbomb_xg",
        "duel_type",
        "duel_outcome",
    ]

    print(
        [
            col
            for col in interesting_columns
            if col in events.columns
        ]
    )

    metrics = (
        build_player_match_metrics(
            events
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved {len(metrics):,} "
        f"player-match rows"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print_striker_metric_diagnostics(
        metrics,
        events,
    )

    display_columns = [
        "match_id",
        "team",
        "player",
        "successful_passes",
        "progressive_passes",
        "progressive_carries",
        "pressures",
        "recoveries",

        # Striker
        "shots",
        "goals",
        "xg",
        "shots_in_box",
        "duels_won",
        "aerial_duels_won",
    ]

    display_columns = [
        col
        for col in display_columns
        if col in metrics.columns
    ]

    print(
        "\nSample player-match metrics:\n"
    )

    print(
        metrics[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()