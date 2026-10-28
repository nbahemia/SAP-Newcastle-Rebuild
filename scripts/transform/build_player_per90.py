from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load processed inputs
# --------------------------------------------------

metrics = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_match_metrics_raw.parquet"
)

minutes = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_minutes.parquet"
)

print(f"Loaded {len(metrics):,} player-match metric rows")
print(f"Loaded {len(minutes):,} player-match minute rows")


# --------------------------------------------------
# Merge player metrics with minutes
# --------------------------------------------------

player_match = metrics.merge(
    minutes[
        [
            "match_id",
            "team",
            "player",
            "minutes",
        ]
    ],
    on=["match_id", "team", "player"],
    how="left",
)


# --------------------------------------------------
# Check for missing minutes
# --------------------------------------------------

missing_minutes = player_match[
    player_match["minutes"].isna()
]

print(f"Rows missing minutes: {len(missing_minutes):,}")

if len(missing_minutes) > 0:
    print()
    print("Players with missing minutes:")

    print(
        missing_minutes[
            [
                "match_id",
                "team",
                "player",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# --------------------------------------------------
# Remove invalid rows
# --------------------------------------------------

player_match = player_match[
    player_match["minutes"].notna()
    & (player_match["minutes"] > 0)
].copy()


# --------------------------------------------------
# Metrics eligible for per-90 calculation
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
# Calculate per-90 metrics
# --------------------------------------------------

for column in metric_columns:
    player_match[f"{column}_per90"] = (
        player_match[column]
        / player_match["minutes"]
        * 90
    )


# --------------------------------------------------
# Additional useful rate metrics
# --------------------------------------------------

player_match["progressive_pass_pct"] = (
    player_match["progressive_passes"]
    / player_match["successful_passes"]
    * 100
)

player_match["progressive_pass_pct"] = (
    player_match["progressive_pass_pct"]
    .fillna(0)
)


# --------------------------------------------------
# Sort for cleaner output
# --------------------------------------------------

player_match = player_match.sort_values(
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


# --------------------------------------------------
# Save final player-match metrics
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_match_metrics.parquet"
)

player_match.to_parquet(
    output_path,
    index=False,
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print()
print(
    f"Saved {len(player_match):,} player-match rows"
)

print(
    f"Output: {output_path}"
)

print()

display_columns = [
    "match_id",
    "team",
    "player",
    "minutes",
    "successful_passes",
    "progressive_passes",
    "progressive_passes_per90",
    "progressive_pass_pct",
    "progressive_carries",
    "progressive_carries_per90",
    "pressures",
    "pressures_per90",
    "recoveries",
    "recoveries_per90",
]

print(
    player_match[
        display_columns
    ]
    .head(30)
    .to_string(
        index=False,
        formatters={
            "minutes": "{:.1f}".format,
            "progressive_passes_per90": "{:.2f}".format,
            "progressive_pass_pct": "{:.1f}%".format,
            "progressive_carries_per90": "{:.2f}".format,
            "pressures_per90": "{:.2f}".format,
            "recoveries_per90": "{:.2f}".format,
        },
    )
)