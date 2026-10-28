from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load player-match metrics
# --------------------------------------------------

player_match = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_match_metrics.parquet"
)

print(f"Loaded {len(player_match):,} player-match rows")


# --------------------------------------------------
# Metrics to aggregate
# --------------------------------------------------

RAW_METRICS = [
    "successful_passes",
    "progressive_passes",
    "passes_into_final_third",
    "progressive_carries",
    "carries_into_final_third",
    "pressures",
    "attacking_third_pressures",
    "recoveries",
    "attacking_third_recoveries",
    "interceptions",
    "duels",

    # Striker metrics
    "shots",
    "goals",
    "xg",
    "shots_in_box",
]


# --------------------------------------------------
# Aggregate totals across all matches
# --------------------------------------------------

aggregation = {
    "minutes": "sum",
}

for column in RAW_METRICS:
    aggregation[column] = "sum"


player_season = (
    player_match
    .groupby(
        ["team", "player"],
        as_index=False,
    )
    .agg(aggregation)
)


# --------------------------------------------------
# Count appearances
# --------------------------------------------------

appearances = (
    player_match
    .groupby(
        ["team", "player"]
    )
    .size()
    .reset_index(
        name="appearances"
    )
)

player_season = player_season.merge(
    appearances,
    on=["team", "player"],
    how="left",
)


# --------------------------------------------------
# Valid per-90 denominator
# --------------------------------------------------

valid_minutes = (
    player_season["minutes"].notna()
    & (player_season["minutes"] > 0)
)


# --------------------------------------------------
# Calculate season-level per-90 metrics
# --------------------------------------------------

for metric in RAW_METRICS:

    player_season[f"{metric}_per90"] = np.where(
        player_season["minutes"] > 0,
        player_season[metric] / player_season["minutes"] * 90,
        np.nan,
    )


# --------------------------------------------------
# Additional rate metrics
# --------------------------------------------------

player_season["progressive_pass_pct"] = np.where(
    player_season["successful_passes"] > 0,
    (
        player_season["progressive_passes"]
        / player_season["successful_passes"]
        * 100
    ),
    np.nan,
)


# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

zero_minute_players = player_season[
    player_season["minutes"] <= 0
]

print()
print(
    f"Players with 0 tournament minutes: "
    f"{len(zero_minute_players):,}"
)

if len(zero_minute_players) > 0:

    print(
        zero_minute_players[
            [
                "team",
                "player",
                "appearances",
                "minutes",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# --------------------------------------------------
# Sort
# --------------------------------------------------

player_season = player_season.sort_values(
    [
        "minutes",
        "team",
        "player",
    ],
    ascending=[
        False,
        True,
        True,
    ],
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_season_metrics.parquet"
)

player_season.to_parquet(
    output_path,
    index=False,
)

print()
print(
    f"Saved {len(player_season):,} "
    "player-season rows"
)

print(
    f"Output: {output_path}"
)

print()


# --------------------------------------------------
# Preview
# --------------------------------------------------

display_columns = [
    "team",
    "player",
    "appearances",
    "minutes",
    "progressive_passes",
    "progressive_passes_per90",
    "progressive_carries",
    "progressive_carries_per90",
    "pressures",
    "pressures_per90",
    "recoveries",
    "recoveries_per90",
]

print(
    player_season[
        display_columns
    ]
    .head(30)
    .to_string(
        index=False,
        formatters={
            "minutes":
                "{:.1f}".format,

            "progressive_passes_per90":
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                ),

            "progressive_carries_per90":
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                ),

            "pressures_per90":
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                ),

            "recoveries_per90":
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                ),
        },
    )
)