from pathlib import Path

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

metric_columns = [
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
]


# --------------------------------------------------
# Aggregate across all matches
# --------------------------------------------------

aggregation = {
    "minutes": "sum",
}

for column in metric_columns:
    aggregation[column] = "sum"


player_season = (
    player_match
    .groupby(
        ["team", "player"],
        as_index=False
    )
    .agg(aggregation)
)


# --------------------------------------------------
# Count appearances
# --------------------------------------------------

appearances = (
    player_match
    .groupby(["team", "player"])
    .size()
    .reset_index(name="appearances")
)

player_season = player_season.merge(
    appearances,
    on=["team", "player"],
    how="left"
)


# --------------------------------------------------
# Calculate season-level per-90 metrics
# --------------------------------------------------

for column in metric_columns:
    player_season[f"{column}_per90"] = (
        player_season[column]
        / player_season["minutes"]
        * 90
    )


# --------------------------------------------------
# Additional rate metrics
# --------------------------------------------------

player_season["progressive_pass_pct"] = (
    player_season["progressive_passes"]
    / player_season["successful_passes"]
    * 100
)

player_season["progressive_pass_pct"] = (
    player_season["progressive_pass_pct"]
    .fillna(0)
)


# --------------------------------------------------
# Sort
# --------------------------------------------------

player_season = player_season.sort_values(
    ["minutes", "team", "player"],
    ascending=[False, True, True]
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
    index=False
)

print()
print(
    f"Saved {len(player_season):,} player-season rows"
)

print(f"Output: {output_path}")

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
            "minutes": "{:.1f}".format,
            "progressive_passes_per90": "{:.2f}".format,
            "progressive_carries_per90": "{:.2f}".format,
            "pressures_per90": "{:.2f}".format,
            "recoveries_per90": "{:.2f}".format,
        }
    )
)