from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

INPUT_FILE = PROCESSED_DIR / "player_season_metrics_enriched.parquet"
OUTPUT_FILE = PROCESSED_DIR / "player_percentiles.parquet"

MIN_MINUTES = 90
MIN_POSITION_SHARE = 0.60

PERCENTILE_METRICS = [
    "goals_per90",
    "xg_per90",
    "shots_per90",
    "shots_in_box_per90",
    "assists_per90",
    "xa_per90",
    "key_pass_per90",

    "passes_per90",
    "successful_passes_per90",
    "pass_accuracy_pct",
    "total_long_balls_per90",
    "accurate_long_balls_per90",
    "long_ball_accuracy_pct",
    "total_cross_per90",
    "accurate_cross_per90",
    "cross_accuracy_pct",
    "accurate_opposition_half_passes_per90",
    "total_opposition_half_passes_per90",

    "progressive_carries_per90",
    "ball_carries_count_per90",
    "total_progressive_ball_carries_distance_per90",
    "total_ball_carries_distance_per90",
    "dribble_success_pct",

    "recoveries_per90",
    "interceptions_per90",
    "duels_won_per90",
    "duel_win_pct",
    "aerials_won_per90",
    "aerial_win_pct",
    "tackles_per90",
    "tackles_won_per90",
    "tackle_win_pct",

    "touches_per90",
    "possession_lost_per90",
    "dispossessed_per90",
]

LOWER_IS_BETTER = {
    "possession_lost_per90",
    "dispossessed_per90",
}


def percentile_rank(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    percentile = (
        numeric.rank(
            pct=True,
            method="average",
        )
        * 100.0
    )

    if not higher_is_better:
        percentile = 100.0 - percentile

    return percentile


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing enriched season file:\n{INPUT_FILE}"
        )

    players = pd.read_parquet(INPUT_FILE)

    eligible = players[
        players["minutes"] >= MIN_MINUTES
    ].copy()

    eligible = eligible[
        eligible["position_group"].notna()
    ].copy()

    available_metrics = [
        metric
        for metric in PERCENTILE_METRICS
        if metric in eligible.columns
    ]

    if not available_metrics:
        raise RuntimeError(
            "None of the configured percentile metrics "
            "exist in the enriched season table."
        )

    for position_group, indices in (
        eligible.groupby("position_group").groups.items()
    ):
        group = eligible.loc[indices]

        for metric in available_metrics:
            eligible.loc[
                indices,
                f"{metric}_percentile",
            ] = percentile_rank(
                group[metric],
                higher_is_better=(
                    metric not in LOWER_IS_BETTER
                ),
            )

    eligible = (
        eligible.sort_values(
            ["position_group", "minutes"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    eligible.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Minimum minutes: {MIN_MINUTES}")
    print(f"Eligible players: {len(eligible):,}")

    print()
    print(
        eligible["position_group"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Percentile metrics created: {len(available_metrics)}"
    )
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
