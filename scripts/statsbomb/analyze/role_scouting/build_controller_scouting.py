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
    / "controller_scouting_shortlist.parquet"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MIN_MINUTES = 180

ROLE_NAME = "Controller / Progressive 6"


# General Controller / Progressive 6 role weights
CONTROLLER_WEIGHTS = {
    "progressive_passes_percentile": 0.25,
    "passes_into_final_third_percentile": 0.20,
    "successful_passes_percentile": 0.10,
    "progressive_carries_percentile": 0.15,
    "recoveries_percentile": 0.125,
    "interceptions_percentile": 0.10,
    "duels_percentile": 0.075,
}


# Nico-style replacement weights
#
# This is an archetype score, not true statistical
# similarity to Nico, because Nico is not currently
# represented using the exact same StatsBomb features.
NICO_STYLE_WEIGHTS = {
    "progressive_passes_percentile": 0.30,
    "passes_into_final_third_percentile": 0.20,
    "successful_passes_percentile": 0.15,
    "progressive_carries_percentile": 0.15,
    "recoveries_percentile": 0.10,
    "interceptions_percentile": 0.07,
    "duels_percentile": 0.03,
}


# ---------------------------------------------------------
# Function 1: Calculate Controller Fit
# ---------------------------------------------------------

def calculate_controller_fit(
    candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate general Controller / Progressive 6 fit.

    This asks:
    How well does the player fit the overall tactical
    requirements of Jaissle's controller role?
    """

    candidates = candidates.copy()

    required_columns = list(CONTROLLER_WEIGHTS.keys())

    missing_columns = [
        col
        for col in required_columns
        if col not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required Controller Fit columns:\n"
            + "\n".join(missing_columns)
        )

    # Main Controller Fit score
    candidates["controller_fit_score"] = 0.0

    for metric, weight in CONTROLLER_WEIGHTS.items():
        candidates["controller_fit_score"] += (
            candidates[metric] * weight
        )

    candidates["controller_fit_score"] = (
        candidates["controller_fit_score"]
        .round(1)
    )

    # ---------------------------------------------
    # Component scores
    # ---------------------------------------------

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

    candidates["possession_involvement_score"] = (
        candidates["successful_passes_percentile"]
        .round(1)
    )

    candidates["carrying_score"] = (
        candidates["progressive_carries_percentile"]
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

    candidates["duel_score"] = (
        candidates["duels_percentile"]
        .round(1)
    )

    return candidates


# ---------------------------------------------------------
# Function 2: Calculate Nico-Style Fit
# ---------------------------------------------------------

def calculate_nico_style_fit(
    candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate a Nico-style replacement score.

    This asks:
    Which players combine the controller qualities we want
    with greater defensive and physical involvement?

    This is NOT true nearest-neighbor similarity to Nico.
    It is a weighted archetype score based on our tactical
    analysis of his role.
    """

    candidates = candidates.copy()

    required_columns = list(NICO_STYLE_WEIGHTS.keys())

    missing_columns = [
        col
        for col in required_columns
        if col not in candidates.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required Nico-style columns:\n"
            + "\n".join(missing_columns)
        )

    # ---------------------------------------------
    # Main Nico-style score
    # ---------------------------------------------

    candidates["nico_style_fit_score"] = 0.0

    for metric, weight in NICO_STYLE_WEIGHTS.items():
        candidates["nico_style_fit_score"] += (
            candidates[metric] * weight
        )

    candidates["nico_style_fit_score"] = (
        candidates["nico_style_fit_score"]
        .round(1)
    )

    # ---------------------------------------------
    # Nico-style components
    # ---------------------------------------------

    candidates["nico_progression_score"] = (
        candidates[
            [
                "progressive_passes_percentile",
                "passes_into_final_third_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["nico_carrying_score"] = (
        candidates["progressive_carries_percentile"]
        .round(1)
    )

    candidates["nico_defensive_score"] = (
        candidates[
            [
                "recoveries_percentile",
                "interceptions_percentile",
            ]
        ]
        .mean(axis=1)
        .round(1)
    )

    candidates["nico_physical_score"] = (
        candidates["duels_percentile"]
        .round(1)
    )

    # Nico-style rank
    candidates["nico_style_rank"] = (
        candidates["nico_style_fit_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return candidates


# ---------------------------------------------------------
# Build Controller Shortlist
# ---------------------------------------------------------

def build_controller_shortlist(
    players: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter DM/CM candidates and calculate both:

    1. General Controller Fit
    2. Nico-Style Fit
    """

    candidates = players[
        (players["primary_position_group"] == "DM/CM")
        & (players["minutes"] >= MIN_MINUTES)
    ].copy()

    print(
        f"{len(candidates):,} DM/CM players meet "
        f"the {MIN_MINUTES}-minute threshold"
    )

    # ---------------------------------------------
    # Calculate both scoring models
    # ---------------------------------------------

    candidates = calculate_controller_fit(candidates)
    candidates = calculate_nico_style_fit(candidates)

    # ---------------------------------------------
    # General Controller rank
    # ---------------------------------------------

    candidates["controller_rank"] = (
        candidates["controller_fit_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ---------------------------------------------
    # Sort primarily by general controller fit
    # ---------------------------------------------

    candidates = candidates.sort_values(
        by=[
            "controller_fit_score",
            "nico_style_fit_score",
            "passing_progression_score",
            "minutes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    # ---------------------------------------------
    # Output columns
    # ---------------------------------------------

    output_columns = [
        "controller_rank",
        "nico_style_rank",
        "player",
        "team",
        "primary_position_group",
        "secondary_position_group",
        "minutes",
        "appearances",

        # Overall scores
        "controller_fit_score",
        "nico_style_fit_score",

        # General Controller components
        "passing_progression_score",
        "possession_involvement_score",
        "carrying_score",
        "defensive_recovery_score",
        "duel_score",

        # Nico-style components
        "nico_progression_score",
        "nico_carrying_score",
        "nico_defensive_score",
        "nico_physical_score",

        # Raw per-90 + percentile metrics
        "progressive_passes_per90",
        "progressive_passes_percentile",
        "passes_into_final_third_per90",
        "passes_into_final_third_percentile",
        "successful_passes_per90",
        "successful_passes_percentile",
        "progressive_carries_per90",
        "progressive_carries_percentile",
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

    shortlist = build_controller_shortlist(players)

    # Make sure output folder exists
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shortlist.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved {len(shortlist):,} controller candidates to:"
    )
    print(OUTPUT_FILE)

    # ---------------------------------------------
    # Display top general Controller candidates
    # ---------------------------------------------

    controller_display = [
        "controller_rank",
        "nico_style_rank",
        "player",
        "team",
        "minutes",
        "controller_fit_score",
        "nico_style_fit_score",
        "passing_progression_score",
        "carrying_score",
        "defensive_recovery_score",
        "duel_score",
    ]

    print(
        "\nTop Controller / Progressive 6 candidates:\n"
    )

    print(
        shortlist[
            [
                col
                for col in controller_display
                if col in shortlist.columns
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------
    # Display top Nico-style candidates separately
    # ---------------------------------------------

    nico_shortlist = shortlist.sort_values(
        by=[
            "nico_style_rank",
            "controller_rank",
        ]
    )

    nico_display = [
        "nico_style_rank",
        "controller_rank",
        "player",
        "team",
        "minutes",
        "nico_style_fit_score",
        "controller_fit_score",
        "nico_progression_score",
        "nico_carrying_score",
        "nico_defensive_score",
        "nico_physical_score",
    ]

    print(
        "\nTop Nico-Style Replacement candidates:\n"
    )

    print(
        nico_shortlist[
            [
                col
                for col in nico_display
                if col in nico_shortlist.columns
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()