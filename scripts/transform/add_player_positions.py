from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Load data
# --------------------------------------------------

player_season = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_season_metrics.parquet"
)

player_positions = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_positions.parquet"
)

print(f"Loaded {len(player_season):,} player-season rows")
print(f"Loaded {len(player_positions):,} player-position rows")


# --------------------------------------------------
# Keep only relevant position columns
# --------------------------------------------------

player_positions = player_positions[
    [
        "team",
        "player",
        "position_group",
        "position_minutes",
        "total_position_minutes",
        "position_share",
    ]
]


# --------------------------------------------------
# Merge
# --------------------------------------------------

player_season = player_season.merge(
    player_positions,
    on=["team", "player"],
    how="left",
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

missing_positions = player_season[
    player_season["position_group"].isna()
]

print(
    f"Rows missing position group: {len(missing_positions):,}"
)

if len(missing_positions) > 0:
    print(
        missing_positions[
            ["team", "player", "minutes"]
        ]
        .head(20)
        .to_string(index=False)
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_season_metrics_with_positions.parquet"
)

player_season.to_parquet(
    output_path,
    index=False,
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

print(
    player_season[
        [
            "team",
            "player",
            "position_group",
            "minutes",
            "position_share",
            "progressive_passes_per90",
            "progressive_carries_per90",
            "pressures_per90",
            "recoveries_per90",
        ]
    ]
    .sort_values(
        ["position_group", "minutes"],
        ascending=[True, False]
    )
    .head(50)
    .to_string(
        index=False,
        formatters={
            "minutes": "{:.1f}".format,
            "position_share": "{:.1%}".format,
            "progressive_passes_per90": "{:.2f}".format,
            "progressive_carries_per90": "{:.2f}".format,
            "pressures_per90": "{:.2f}".format,
            "recoveries_per90": "{:.2f}".format,
        }
    )
)