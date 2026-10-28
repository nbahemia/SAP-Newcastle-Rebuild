from pathlib import Path

import numpy as np
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
    PROCESSED_DATA_DIR / "player_match_raw_metrics.parquet"
)

minutes = pd.read_parquet(
    PROCESSED_DATA_DIR / "player_minutes.parquet"
)

print(
    f"Loaded {len(metrics):,} "
    "player-match metric rows"
)

print(
    f"Loaded {len(minutes):,} "
    "player-match minute rows"
)


# --------------------------------------------------
# Merge player metrics with corrected minutes
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
    on=[
        "match_id",
        "team",
        "player",
    ],
    how="left",
)


# --------------------------------------------------
# Check missing minutes
# --------------------------------------------------

missing_minutes = player_match[
    player_match["minutes"].isna()
]

print(
    f"Rows missing minutes: "
    f"{len(missing_minutes):,}"
)

if len(missing_minutes) > 0:

    print()

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
    "shots",
    "goals",
    "xg",
    "shots_in_box",
]


# --------------------------------------------------
# Identify rows with valid per-90 denominators
# --------------------------------------------------

valid_minutes = (
    player_match["minutes"].notna()
    & (player_match["minutes"] > 0)
)


# --------------------------------------------------
# Calculate per-90 metrics
# --------------------------------------------------

for column in metric_columns:

    player_match[f"{column}_per90"] = np.where(
        valid_minutes,
        (
            player_match[column]
            / player_match["minutes"]
            * 90
        ),
        np.nan,
    )


# --------------------------------------------------
# Progressive pass percentage
# --------------------------------------------------

player_match["progressive_pass_pct"] = np.where(
    player_match["successful_passes"] > 0,
    (
        player_match["progressive_passes"]
        / player_match["successful_passes"]
        * 100
    ),
    np.nan,
)


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

zero_minute_rows = player_match[
    player_match["minutes"] <= 0
]

print(
    f"Rows with 0 minutes: "
    f"{len(zero_minute_rows):,}"
)


# --------------------------------------------------
# Sort
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
# Save
# --------------------------------------------------

output_path = (
    PROCESSED_DATA_DIR
    / "player_match_metrics.parquet"
)

player_match.to_parquet(
    output_path,
    index=False,
)


print()

print(
    f"Saved {len(player_match):,} "
    "player-match rows"
)

print(
    f"Output: {output_path}"
)

print()


# --------------------------------------------------
# Preview
# --------------------------------------------------

display_columns = [
    "match_id",
    "team",
    "player",
    "minutes",
    "progressive_passes",
    "progressive_passes_per90",
    "pressures",
    "pressures_per90",
]


print(
    player_match[
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

            "pressures_per90":
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else ""
                ),
        },
    )
)