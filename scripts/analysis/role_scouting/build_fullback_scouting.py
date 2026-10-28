from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED / "player_percentiles.parquet"
OUTPUT_FILE = (
    PROCESSED
    / "role_analysis"
    / "left_back_scouting_shortlist.parquet"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MIN_MINUTES = 180
ROLE_NAME = "Vertical Two-Way Left Back"


# ---------------------------------------------------------
# General Fullback Role Weights
# ---------------------------------------------------------

FULLBACK_WEIGHTS = {
    "progressive_carries_percentile": 0.20,
    "carries_into_final_third_percentile": 0.15,

    "progressive_passes_percentile": 0.15,
    "passes_into_final_third_percentile": 0.10,

    "recoveries_percentile": 0.10,
    "interceptions_percentile": 0.07,

    "pressures_percentile": 0.08,
    "attacking_third_pressures_percentile": 0.05,

    "duels_percentile": 0.10,
}


# ---------------------------------------------------------
# Hall-Style Technical Upside Weights
# ---------------------------------------------------------

HALL_STYLE_WEIGHTS = {
    "progressive_passes_percentile": 0.25,
    "passes_into_final_third_percentile": 0.20,
    "successful_passes_percentile": 0.15,

    "progressive_carries_percentile": 0.15,
    "carries_into_final_third_percentile": 0.10,

    "recoveries_percentile": 0.07,
    "interceptions_percentile": 0.04,
    "duels_percentile": 0.04,
}


# ---------------------------------------------------------
# Function 1: Calculate General Fullback Fit
# ---------------------------------------------------------

def calculate_fullback_fit(
    candidates: pd.DataFrame
) -> pd.DataFrame:

    candidates = candidates.copy()

    required_columns = list(FULLBACK_WEIGHTS.keys())

    missing_columns = [
        col
        for col in required_columns
        if col not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required Fullback Fit columns:\n"
            + "\n".join(missing_columns)
        )

    candidates["fullback_fit_score"] = 0.0

    for metric, weight in FULLBACK_WEIGHTS.items():
        candidates["fullback_fit_score"] += (
            candidates[metric] * weight
        )

    candidates["fullback_fit_score"] = (
        candidates["fullback_fit_score"]
        .round(1)
    )

    candidates["vertical_carrying_score"] = (
        candidates[
            [
                "progressive_carries_percentile",
                "carries_into_final_third_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["passing_progression_score"] = (
        candidates[
            [
                "progressive_passes_percentile",
                "passes_into_final_third_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["defensive_recovery_score"] = (
        candidates[
            [
                "recoveries_percentile",
                "interceptions_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["pressing_score"] = (
        candidates[
            [
                "pressures_percentile",
                "attacking_third_pressures_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["duel_score"] = (
        candidates["duels_percentile"]
        .round(1)
    )

    return candidates


# ---------------------------------------------------------
# Function 2: Calculate Hall-Style Technical Fit
# ---------------------------------------------------------

def calculate_hall_style_fit(
    candidates: pd.DataFrame
) -> pd.DataFrame:

    candidates = candidates.copy()

    required_columns = list(HALL_STYLE_WEIGHTS.keys())

    missing_columns = [
        col
        for col in required_columns
        if col not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required Hall-style columns:\n"
            + "\n".join(missing_columns)
        )

    candidates["hall_style_fit_score"] = 0.0

    for metric, weight in HALL_STYLE_WEIGHTS.items():
        candidates["hall_style_fit_score"] += (
            candidates[metric] * weight
        )

    candidates["hall_style_fit_score"] = (
        candidates["hall_style_fit_score"]
        .round(1)
    )

    candidates["hall_progression_score"] = (
        candidates[
            [
                "progressive_passes_percentile",
                "passes_into_final_third_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["hall_possession_score"] = (
        candidates["successful_passes_percentile"]
        .round(1)
    )

    candidates["hall_carrying_score"] = (
        candidates[
            [
                "progressive_carries_percentile",
                "carries_into_final_third_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["hall_defensive_score"] = (
        candidates[
            [
                "recoveries_percentile",
                "interceptions_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    return candidates


# ---------------------------------------------------------
# Build Left-Back Shortlist
# ---------------------------------------------------------

def build_fullback_shortlist(
    players: pd.DataFrame
) -> pd.DataFrame:

    required_base_columns = [
        "primary_position_group",
        "primary_position_side",
        "minutes",
    ]

    missing_base = [
        col
        for col in required_base_columns
        if col not in players.columns
    ]

    if missing_base:
        raise ValueError(
            "Missing required position columns:\n"
            + "\n".join(missing_base)
            + "\n\nRebuild the position/percentile pipeline first."
        )

    candidates = players[
        (players["primary_position_group"] == "FB/WB")
        & (players["primary_position_side"] == "Left")
        & (players["minutes"] >= MIN_MINUTES)
    ].copy()

    print(
        f"{len(candidates):,} left-sided FB/WB players meet "
        f"the {MIN_MINUTES}-minute threshold"
    )

    candidates = calculate_fullback_fit(candidates)
    candidates = calculate_hall_style_fit(candidates)

    candidates["fullback_rank"] = (
        candidates["fullback_fit_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    candidates["hall_style_rank"] = (
        candidates["hall_style_fit_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    candidates = candidates.sort_values(
        by=[
            "fullback_fit_score",
            "hall_style_fit_score",
            "vertical_carrying_score",
            "minutes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    output_columns = [
        "fullback_rank",
        "hall_style_rank",

        "player",
        "team",
        "primary_position_group",
        "primary_position_side",
        "secondary_position_group",
        "minutes",
        "appearances",

        "fullback_fit_score",
        "hall_style_fit_score",

        "vertical_carrying_score",
        "passing_progression_score",
        "defensive_recovery_score",
        "pressing_score",
        "duel_score",

        "hall_progression_score",
        "hall_possession_score",
        "hall_carrying_score",
        "hall_defensive_score",

        "successful_passes_per90",
        "successful_passes_percentile",

        "progressive_passes_per90",
        "progressive_passes_percentile",

        "passes_into_final_third_per90",
        "passes_into_final_third_percentile",

        "progressive_carries_per90",
        "progressive_carries_percentile",

        "carries_into_final_third_per90",
        "carries_into_final_third_percentile",

        "pressures_per90",
        "pressures_percentile",

        "attacking_third_pressures_per90",
        "attacking_third_pressures_percentile",

        "recoveries_per90",
        "recoveries_percentile",

        "interceptions_per90",
        "interceptions_percentile",

        "duels_per90",
        "duels_percentile",
    ]

    output_columns = [
        col
        for col in output_columns
        if col in candidates.columns
    ]

    return candidates[output_columns].copy()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    players = pd.read_parquet(INPUT_FILE)

    print(
        f"Loaded {len(players):,} eligible player rows"
    )

    print(
        "\nPosition-side distribution:"
    )

    if "primary_position_side" in players.columns:
        print(
            players["primary_position_side"]
            .value_counts(dropna=False)
            .to_string()
        )

    shortlist = build_fullback_shortlist(players)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shortlist.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved {len(shortlist):,} left-back candidates to:"
    )
    print(OUTPUT_FILE)

    fullback_display = [
        "fullback_rank",
        "hall_style_rank",
        "player",
        "team",
        "minutes",
        "fullback_fit_score",
        "hall_style_fit_score",
        "vertical_carrying_score",
        "passing_progression_score",
        "defensive_recovery_score",
        "pressing_score",
        "duel_score",
    ]

    print(
        "\nTop Vertical Two-Way Left-Back candidates:\n"
    )

    print(
        shortlist[
            [
                col
                for col in fullback_display
                if col in shortlist.columns
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    hall_shortlist = shortlist.sort_values(
        by=[
            "hall_style_rank",
            "fullback_rank",
        ]
    )

    hall_display = [
        "hall_style_rank",
        "fullback_rank",
        "player",
        "team",
        "minutes",
        "hall_style_fit_score",
        "fullback_fit_score",
        "hall_progression_score",
        "hall_possession_score",
        "hall_carrying_score",
        "hall_defensive_score",
    ]

    print(
        "\nTop Technical Left-Back candidates:\n"
    )

    print(
        hall_shortlist[
            [
                col
                for col in hall_display
                if col in hall_shortlist.columns
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()