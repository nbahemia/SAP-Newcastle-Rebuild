from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load percentile data
# --------------------------------------------------

players = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_percentiles.parquet"
)

print(f"Loaded {len(players):,} player percentile rows")


# --------------------------------------------------
# Scouting category definitions
# --------------------------------------------------

CATEGORY_METRICS = {

    "progression": [
        "progressive_passes_percentile",
        "passes_into_final_third_percentile",
    ],

    "carrying": [
        "progressive_carries_percentile",
        "carries_into_final_third_percentile",
    ],

    "pressing": [
        "pressures_percentile",
        "attacking_third_pressures_percentile",
    ],

    "ball_winning": [
        "recoveries_percentile",
        "attacking_third_recoveries_percentile",
        "interceptions_percentile",
    ],

    "duels": [
        "duels_percentile",
    ],
}


# --------------------------------------------------
# Validate required columns
# --------------------------------------------------

required_columns = []

for metrics in CATEGORY_METRICS.values():
    required_columns.extend(metrics)

missing_columns = [
    column
    for column in required_columns
    if column not in players.columns
]

if missing_columns:

    raise ValueError(
        "Missing required percentile columns:\n"
        + "\n".join(missing_columns)
    )


# --------------------------------------------------
# Build scouting-category scores
# --------------------------------------------------

for category, metrics in CATEGORY_METRICS.items():

    score_column = f"{category}_score"

    players[score_column] = (
        players[metrics]
        .mean(
            axis=1,
            skipna=True,
        )
        .round(1)
    )


# --------------------------------------------------
# Overall profile score
# A broad summary of the five categories.
# --------------------------------------------------

score_columns = [
    "progression_score",
    "carrying_score",
    "pressing_score",
    "ball_winning_score",
    "duels_score",
]

players["overall_profile_score"] = (
    players[score_columns]
    .mean(
        axis=1,
        skipna=True,
    )
    .round(1)
)


# --------------------------------------------------
# Sort
# --------------------------------------------------

players = players.sort_values(
    [
        "primary_position_group",
        "overall_profile_score",
        "minutes",
    ],
    ascending=[
        True,
        False,
        False,
    ],
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_scouting_scores.parquet"
)

players.to_parquet(
    output_path,
    index=False,
)

print()

print(
    f"Saved {len(players):,} player scouting-score rows"
)

print(f"Output: {output_path}")

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
    "progression_score",
    "carrying_score",
    "pressing_score",
    "ball_winning_score",
    "duels_score",
    "overall_profile_score",
]

print(
    players[
        display_columns
    ]
    .head(40)
    .to_string(
        index=False,
        formatters={
            "minutes": "{:.1f}".format,
            "progression_score": "{:.1f}".format,
            "carrying_score": "{:.1f}".format,
            "pressing_score": "{:.1f}".format,
            "ball_winning_score": "{:.1f}".format,
            "duels_score": "{:.1f}".format,
            "overall_profile_score": "{:.1f}".format,
        }
    )
)