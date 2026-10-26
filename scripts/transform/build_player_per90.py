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
# Merge
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
    how="left"
)


# --------------------------------------------------
# Check for missing minutes
# --------------------------------------------------

missing_minutes = player_match[
    player_match["minutes"].isna()
]

print(
    f"Rows missing minutes: {len(missing_minutes):,}"
)

if len(missing_minutes) > 0:
    print(
        missing_minutes[
            ["match_id", "team", "player"]
        ]
        .head(20)
        .to_string(index=False)
    )


# --------------------------------------------------
# Per-90 metrics
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

for column in metric_columns:
    player_match[f"{column}_per90"] = (
        player_match[column]
        / player_match["minutes"]
        * 90
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

player_match.to_parquet(
    PROCESSED_DATA_DIR / "player_match_metrics.parquet",
    index=False
)

print(
    f"Saved {len(player_match):,} player-match rows"
)

print()

print(
    player_match[
        [
            "match_id",
            "team",
            "player",
            "minutes",
            "progressive_passes",
            "progressive_passes_per90",
            "pressures",
            "pressures_per90",
        ]
    ]
    .head(25)
    .to_string(index=False)
)