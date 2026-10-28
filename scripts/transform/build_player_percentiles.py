from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Settings
# --------------------------------------------------

MIN_MINUTES = 180


# --------------------------------------------------
# Load player-season data with positions
# --------------------------------------------------

players = pd.read_parquet(
    PROCESSED_DATA_DIR
    / "player_season_metrics_with_positions.parquet"
)

print(
    f"Loaded {len(players):,} player-season rows"
)


# --------------------------------------------------
# Metrics to percentile-rank
# --------------------------------------------------

PERCENTILE_METRICS = [
    "successful_passes_per90",
    "progressive_passes_per90",
    "passes_into_final_third_per90",
    "progressive_carries_per90",
    "carries_into_final_third_per90",
    "pressures_per90",
    "attacking_third_pressures_per90",
    "recoveries_per90",
    "attacking_third_recoveries_per90",
    "interceptions_per90",
    "duels_per90",

    # Striker metrics
    "shots_per90",
    "goals_per90",
    "xg_per90",
    "shots_in_box_per90",
]


# --------------------------------------------------
# Eligible comparison pool
# --------------------------------------------------

eligible = players[
    (players["minutes"] >= MIN_MINUTES)
    & players["primary_position_group"].notna()
].copy()

print(
    f"Players meeting {MIN_MINUTES}-minute threshold: "
    f"{len(eligible):,}"
)


# --------------------------------------------------
# Percentiles within position group
# --------------------------------------------------

for metric in PERCENTILE_METRICS:

    percentile_column = (
        metric.replace(
            "_per90",
            "_percentile"
        )
    )

    eligible[
        percentile_column
    ] = (
        eligible
        .groupby(
            "primary_position_group"
        )[metric]
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )


# --------------------------------------------------
# Round percentile columns
# --------------------------------------------------

percentile_columns = [
    column
    for column in eligible.columns
    if column.endswith("_percentile")
]

eligible[
    percentile_columns
] = eligible[
    percentile_columns
].round(1)


# --------------------------------------------------
# Add peer-group size
# --------------------------------------------------

peer_counts = (
    eligible
    .groupby(
        "primary_position_group"
    )
    .size()
    .to_dict()
)

eligible[
    "position_peer_count"
] = (
    eligible[
        "primary_position_group"
    ]
    .map(peer_counts)
)


# --------------------------------------------------
# Sort
# --------------------------------------------------

eligible = eligible.sort_values(
    [
        "primary_position_group",
        "minutes",
    ],
    ascending=[
        True,
        False,
    ],
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_percentiles.parquet"
)

eligible.to_parquet(
    output_path,
    index=False,
)

print()

print(
    f"Saved {len(eligible):,} player percentile rows"
)

print(
    f"Output: {output_path}"
)

print()


# --------------------------------------------------
# Peer-group summary
# --------------------------------------------------

print("Players by position group:")

print(
    eligible[
        "primary_position_group"
    ]
    .value_counts()
    .sort_index()
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
    "position_peer_count",
    "progressive_passes_per90",
    "progressive_passes_percentile",
    "progressive_carries_per90",
    "progressive_carries_percentile",
    "pressures_per90",
    "pressures_percentile",
    "recoveries_per90",
    "recoveries_percentile",
]


print(
    eligible[
        display_columns
    ]
    .head(40)
    .to_string(
        index=False,
        formatters={
            "minutes":
                "{:.1f}".format,

            "progressive_passes_per90":
                "{:.2f}".format,

            "progressive_passes_percentile":
                "{:.1f}".format,

            "progressive_carries_per90":
                "{:.2f}".format,

            "progressive_carries_percentile":
                "{:.1f}".format,

            "pressures_per90":
                "{:.2f}".format,

            "pressures_percentile":
                "{:.1f}".format,

            "recoveries_per90":
                "{:.2f}".format,

            "recoveries_percentile":
                "{:.1f}".format,
        }
    )
)