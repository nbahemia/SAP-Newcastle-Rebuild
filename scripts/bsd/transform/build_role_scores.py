from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

INPUT_FILE = PROCESSED_DIR / "player_percentiles.parquet"
OUTPUT_FILE = PROCESSED_DIR / "player_role_scores.parquet"

ROLE_MODELS = {
    "jaissle_striker": {
        "position_groups": {"FWD"},
        "weights": {
            "xg_per90_percentile": 0.25,
            "shots_in_box_per90_percentile": 0.20,
            "shots_per90_percentile": 0.15,
            "goals_per90_percentile": 0.10,
            "aerials_won_per90_percentile": 0.12,
            "duels_won_per90_percentile": 0.08,
            "progressive_carries_per90_percentile": 0.05,
            "touches_per90_percentile": 0.05,
        },
    },

    "vertical_defender": {
        "position_groups": {"DEF"},
        "weights": {
            "progressive_carries_per90_percentile": 0.20,
            "total_progressive_ball_carries_distance_per90_percentile": 0.15,
            "accurate_opposition_half_passes_per90_percentile": 0.15,
            "total_cross_per90_percentile": 0.10,
            "recoveries_per90_percentile": 0.10,
            "interceptions_per90_percentile": 0.08,
            "duels_won_per90_percentile": 0.10,
            "aerials_won_per90_percentile": 0.07,
            "tackles_won_per90_percentile": 0.05,
        },
    },

    "controller_midfielder": {
        "position_groups": {"MID"},
        "weights": {
            "successful_passes_per90_percentile": 0.20,
            "pass_accuracy_pct_percentile": 0.10,
            "accurate_opposition_half_passes_per90_percentile": 0.20,
            "total_long_balls_per90_percentile": 0.10,
            "progressive_carries_per90_percentile": 0.10,
            "recoveries_per90_percentile": 0.10,
            "interceptions_per90_percentile": 0.07,
            "duels_won_per90_percentile": 0.08,
            "touches_per90_percentile": 0.05,
        },
    },
}


def score_role(
    players: pd.DataFrame,
    role_name: str,
    config: dict,
) -> pd.DataFrame:
    candidates = players[
        players["position_group"].isin(
            config["position_groups"]
        )
    ].copy()

    requested_weights = config["weights"]

    available_weights = {
        metric: weight
        for metric, weight in requested_weights.items()
        if metric in candidates.columns
    }

    missing_metrics = [
        metric
        for metric in requested_weights
        if metric not in candidates.columns
    ]

    if missing_metrics:
        print()
        print(f"{role_name}: missing metrics")
        for metric in missing_metrics:
            print(f"  - {metric}")

    if not available_weights:
        print(f"{role_name}: no usable metrics, skipping")
        return pd.DataFrame()

    total_weight = sum(available_weights.values())

    normalized_weights = {
        metric: weight / total_weight
        for metric, weight in available_weights.items()
    }

    score = pd.Series(
        0.0,
        index=candidates.index,
    )

    observed_weight = pd.Series(
        0.0,
        index=candidates.index,
    )

    for metric, weight in normalized_weights.items():
        values = pd.to_numeric(
            candidates[metric],
            errors="coerce",
        )

        valid = values.notna()

        score.loc[valid] += (
            values.loc[valid]
            * weight
        )

        observed_weight.loc[valid] += weight

    candidates["role_score"] = np.where(
        observed_weight > 0,
        score / observed_weight,
        np.nan,
    )

    candidates["role"] = role_name
    candidates["role_metric_coverage"] = observed_weight
    candidates["role_score"] = candidates["role_score"].round(1)

    return candidates


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing percentile file:\n{INPUT_FILE}"
        )

    players = pd.read_parquet(INPUT_FILE)

    scored = []

    for role_name, config in ROLE_MODELS.items():
        role_table = score_role(
            players=players,
            role_name=role_name,
            config=config,
        )

        if role_table.empty:
            continue

        scored.append(role_table)

        print()
        print("=" * 88)
        print(role_name.upper())
        print("=" * 88)

        display_columns = [
            "player_name",
            "team_name",
            "position",
            "minutes",
            "role_score",
            "role_metric_coverage",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in role_table.columns
        ]

        print(
            role_table
            .sort_values(
                "role_score",
                ascending=False,
            )[display_columns]
            .head(15)
            .to_string(index=False)
        )

    if not scored:
        raise RuntimeError(
            "No role tables were produced."
        )

    output = pd.concat(
        scored,
        ignore_index=True,
    )

    output = output.sort_values(
        ["role", "role_score"],
        ascending=[True, False],
    )

    output.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved {len(output):,} role-score rows"
    )
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
