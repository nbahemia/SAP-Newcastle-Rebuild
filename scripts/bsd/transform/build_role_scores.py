from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

INPUT_FILE = PROCESSED_DIR / "player_percentiles.parquet"
OUTPUT_FILE = PROCESSED_DIR / "player_role_scores.parquet"

MIN_POSITION_SHARE = 0.60

ROLE_MODELS = {
    "jaissle_striker": {
        "position_groups": {"ST"},
        "weights": {
            "xg_per90_percentile": 0.30,
            "shots_in_box_per90_percentile": 0.20,
            "shots_per90_percentile": 0.15,
            "goals_per90_percentile": 0.12,
            "aerials_won_per90_percentile": 0.07,
            "duels_won_per90_percentile": 0.05,
            "progressive_carries_per90_percentile": 0.08,
            "touches_per90_percentile": 0.03,
        },
    },

    "vertical_fullback": {
        "position_groups": {"FB/WB"},
        "weights": {
            "progressive_carries_per90_percentile": 0.25,
            "total_progressive_ball_carries_distance_per90_percentile": 0.18,
            "accurate_opposition_half_passes_per90_percentile": 0.17,
            "total_cross_per90_percentile": 0.12,
            "recoveries_per90_percentile": 0.10,
            "interceptions_per90_percentile": 0.06,
            "duels_won_per90_percentile": 0.05,
            "aerials_won_per90_percentile": 0.03,
            "tackles_won_per90_percentile": 0.04,
        },
    },

   "controller_midfielder": {
        "position_groups": {"DM/CM"},
        "weights": {
            "successful_passes_per90_percentile": 0.18,
            "pass_accuracy_pct_percentile": 0.05,
            "accurate_opposition_half_passes_per90_percentile": 0.22,
            "total_long_balls_per90_percentile": 0.15,
            "progressive_carries_per90_percentile": 0.10,
            "recoveries_per90_percentile": 0.12,
            "interceptions_per90_percentile": 0.07,
            "duels_won_per90_percentile": 0.05,
            "touches_per90_percentile": 0.06,
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

    candidates = candidates[
        candidates["position_group_share"]
        >= MIN_POSITION_SHARE
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

def print_newcastle_players(
    players: pd.DataFrame,
    role_scores: pd.DataFrame,
) -> None:

    print()
    print("=" * 100)
    print(
        "NEWCASTLE UNITED PLAYER PERCENTILES + ROLE SCORES"
        .center(100)
    )
    print("=" * 100)

    newcastle = players[
        players["team_name"]
        == "Newcastle United"
    ].copy()

    if newcastle.empty:
        print(
            "No Newcastle United players found."
        )
        return

    # -----------------------------------------------------
    # Pull Newcastle role scores
    # -----------------------------------------------------

    newcastle_scores = role_scores[
        role_scores["team_name"]
        == "Newcastle United"
    ][
        [
            "player_id",
            "role",
            "role_score",
            "role_metric_coverage",
        ]
    ].copy()

    # One player can theoretically appear in multiple role
    # models, so pivot role scores into separate columns.
    score_pivot = (
        newcastle_scores
        .pivot_table(
            index="player_id",
            columns="role",
            values="role_score",
            aggfunc="first",
        )
        .reset_index()
    )

    score_pivot = score_pivot.rename(
        columns={
            "jaissle_striker":
                "jaissle_striker_score",

            "vertical_fullback":
                "vertical_fullback_score",

            "controller_midfielder":
                "controller_midfielder_score",
        }
    )

    newcastle = newcastle.merge(
        score_pivot,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Metrics relevant to each tactical group
    # -----------------------------------------------------

    ROLE_DISPLAY_METRICS = {

        "ST": [
            "xg_per90_percentile",
            "shots_in_box_per90_percentile",
            "shots_per90_percentile",
            "goals_per90_percentile",
            "aerials_won_per90_percentile",
            "duels_won_per90_percentile",
            "progressive_carries_per90_percentile",
            "touches_per90_percentile",
        ],

        "FB/WB": [
            "progressive_carries_per90_percentile",
            "total_progressive_ball_carries_distance_per90_percentile",
            "accurate_opposition_half_passes_per90_percentile",
            "total_cross_per90_percentile",
            "recoveries_per90_percentile",
            "interceptions_per90_percentile",
            "duels_won_per90_percentile",
            "aerials_won_per90_percentile",
            "tackles_won_per90_percentile",
        ],

        "DM/CM": [
            "successful_passes_per90_percentile",
            "pass_accuracy_pct_percentile",
            "accurate_opposition_half_passes_per90_percentile",
            "total_long_balls_per90_percentile",
            "progressive_carries_per90_percentile",
            "recoveries_per90_percentile",
            "interceptions_per90_percentile",
            "duels_won_per90_percentile",
            "touches_per90_percentile",
        ],

        "CB": [
            "successful_passes_per90_percentile",
            "pass_accuracy_pct_percentile",
            "aerials_won_per90_percentile",
            "duels_won_per90_percentile",
            "recoveries_per90_percentile",
            "interceptions_per90_percentile",
            "tackles_won_per90_percentile",
        ],

        "AM": [
            "xg_per90_percentile",
            "progressive_carries_per90_percentile",
            "accurate_opposition_half_passes_per90_percentile",
            "touches_per90_percentile",
        ],

        "W": [
            "xg_per90_percentile",
            "shots_per90_percentile",
            "progressive_carries_per90_percentile",
            "total_cross_per90_percentile",
            "touches_per90_percentile",
        ],
    }

    ROLE_SCORE_COLUMNS = {
        "ST": "jaissle_striker_score",
        "FB/WB": "vertical_fullback_score",
        "DM/CM": "controller_midfielder_score",
    }

    # -----------------------------------------------------
    # Print each tactical group
    # -----------------------------------------------------

    for position_group in [
    "GK",
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
    ]:

        group = newcastle[
            newcastle["position_group"]
            == position_group
        ].copy()

        if group.empty:
            continue

        role_score_column = (
            ROLE_SCORE_COLUMNS.get(
                position_group
            )
        )

        if (
            role_score_column
            and role_score_column
            in group.columns
        ):
            group = group.sort_values(
                [
                    role_score_column,
                    "minutes",
                ],
                ascending=[
                    False,
                    False,
                ],
                na_position="last",
            )

        else:
            group = group.sort_values(
                "minutes",
                ascending=False,
            )

        print()
        print("=" * 88)
        print(
            f"{position_group} — NEWCASTLE UNITED"
        )
        print("=" * 88)

        metric_columns = [
            metric
            for metric
            in ROLE_DISPLAY_METRICS.get(
                position_group,
                [],
            )
            if metric in group.columns
        ]

        for _, player in group.iterrows():

            print()
            print(
                f"{player['player_name']}"
            )

            print(
                f"  Minutes: "
                f"{int(player['minutes'])}"
            )

            print(
                f"  Position confidence: "
                f"{player['position_group_share'] * 100:.0f}%"
            )

            if (
                "observed_position_groups"
                in player.index
            ):
                print(
                    f"  Observed roles: "
                    f"{player['observed_position_groups']}"
                )

            if (
                role_score_column
                and role_score_column
                in player.index
                and pd.notna(
                    player[
                        role_score_column
                    ]
                )
            ):
                print(
                    f"  Role score: "
                    f"{player[role_score_column]:.1f}"
                )

            if metric_columns:

                print(
                    "  Percentiles:"
                )

                for metric in metric_columns:

                    value = player[
                        metric
                    ]

                    if pd.isna(value):
                        continue

                    label = (
                        metric
                        .replace(
                            "_percentile",
                            "",
                        )
                        .replace(
                            "_per90",
                            "",
                        )
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    )

                    print(
                        f"    {label:<42} "
                        f"{value:>5.1f}"
                    )

            print(
                "-" * 88
            )
        
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
            "position_group_share",
            "position_group",
            "observed_position_groups",
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

    print(
        OUTPUT_FILE
    )

    print_newcastle_players(
        players=players,
        role_scores=output,
    )
        
if __name__ == "__main__":
    main()
