from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

NEWCASTLE_TEAM_ID = 4
AL_AHLI_TEAM_ID = 258

NEWCASTLE_PERCENTILES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "player_percentiles.parquet"
)

SAUDI_PERCENTILES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "saudi_2025_26_player_percentiles.parquet"
)

ROLE_COMPARISON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "jaissle_role_occupancy_comparison.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "jaissle_statistical_percentile_comparison.parquet"
)

POSITION_ORDER = [
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
]

METRICS_BY_POSITION = {
    "CB": [
        ("successful_passes_per90_percentile", "Successful Passes"),
        ("pass_accuracy_pct_percentile", "Pass Accuracy"),
        ("aerials_won_per90_percentile", "Aerials Won"),
        ("duels_won_per90_percentile", "Duels Won"),
        ("recoveries_per90_percentile", "Recoveries"),
        ("interceptions_per90_percentile", "Interceptions"),
        ("tackles_won_per90_percentile", "Tackles Won"),
    ],

    "FB/WB": [
        ("progressive_carries_per90_percentile", "Progressive Carries"),
        (
            "total_progressive_ball_carries_distance_per90_percentile",
            "Progressive Carry Distance",
        ),
        (
            "accurate_opposition_half_passes_per90_percentile",
            "Accurate Opposition-Half Passes",
        ),
        ("total_crosses_per90_percentile", "Crosses"),
        ("recoveries_per90_percentile", "Recoveries"),
        ("interceptions_per90_percentile", "Interceptions"),
        ("duels_won_per90_percentile", "Duels Won"),
        ("aerials_won_per90_percentile", "Aerials Won"),
        ("tackles_won_per90_percentile", "Tackles Won"),
    ],

    "DM/CM": [
        ("successful_passes_per90_percentile", "Successful Passes"),
        ("pass_accuracy_pct_percentile", "Pass Accuracy"),
        (
            "accurate_opposition_half_passes_per90_percentile",
            "Accurate Opposition-Half Passes",
        ),
        ("total_long_balls_per90_percentile", "Long Balls"),
        ("progressive_carries_per90_percentile", "Progressive Carries"),
        ("recoveries_per90_percentile", "Recoveries"),
        ("interceptions_per90_percentile", "Interceptions"),
        ("duels_won_per90_percentile", "Duels Won"),
        ("touches_per90_percentile", "Touches"),
    ],

    "AM": [
        ("xg_per90_percentile", "xG"),
        ("xa_per90_percentile", "xA"),
        ("key_passes_per90_percentile", "Key Passes"),
        ("progressive_carries_per90_percentile", "Progressive Carries"),
        (
            "accurate_opposition_half_passes_per90_percentile",
            "Accurate Opposition-Half Passes",
        ),
        ("touches_per90_percentile", "Touches"),
        ("recoveries_per90_percentile", "Recoveries"),
        ("interceptions_per90_percentile", "Interceptions"),
    ],

    "W": [
        ("xg_per90_percentile", "xG"),
        ("xa_per90_percentile", "xA"),
        ("key_passes_per90_percentile", "Key Passes"),
        ("shots_per90_percentile", "Shots"),
        ("progressive_carries_per90_percentile", "Progressive Carries"),
        ("total_crosses_per90_percentile", "Crosses"),
        (
            "accurate_opposition_half_passes_per90_percentile",
            "Accurate Opposition-Half Passes",
        ),
        ("touches_per90_percentile", "Touches"),
    ],

    "ST": [
        ("xg_per90_percentile", "xG"),
        ("shots_per90_percentile", "Shots"),
        ("goals_per90_percentile", "Goals"),
        ("aerials_won_per90_percentile", "Aerials Won"),
        ("duels_won_per90_percentile", "Duels Won"),
        ("progressive_carries_per90_percentile", "Progressive Carries"),
        ("touches_per90_percentile", "Touches"),
    ],
}


def weighted_percentile(
    players: pd.DataFrame,
    metric: str,
):
    valid = players[
        players[metric].notna()
        & players["minutes"].notna()
        & (players["minutes"] > 0)
    ].copy()

    if valid.empty:
        return np.nan

    return np.average(
        valid[metric],
        weights=valid["minutes"],
    )


def build_unit_profile(
    players: pd.DataFrame,
    team_filter,
    team_label: str,
):
    team = players[
        team_filter(players)
    ].copy()

    print()
    print(
        f"{team_label}: matched players = "
        f"{len(team)}"
    )

    if not team.empty:
        print(
            team[
                [
                    "player_name",
                    "team_name",
                    "position_group",
                    "minutes",
                ]
            ]
            .sort_values(
                "minutes",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    rows = []

    for position_group in POSITION_ORDER:

        group = team[
            team["position_group"]
            == position_group
        ].copy()

        configured = METRICS_BY_POSITION[
            position_group
        ]

        for metric, metric_label in configured:

            if metric not in group.columns:
                continue

            value = weighted_percentile(
                group,
                metric,
            )

            if pd.isna(value):
                continue

            rows.append(
                {
                    "team":
                        team_label,

                    "position_group":
                        position_group,

                    "metric":
                        metric,

                    "metric_label":
                        metric_label,

                    "weighted_league_percentile":
                        round(
                            float(value),
                            1,
                        ),

                    "players_in_group":
                        int(
                            len(group)
                        ),

                    "group_minutes":
                        float(
                            group[
                                "minutes"
                            ].sum()
                        ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "team",
            "position_group",
            "metric",
            "metric_label",
            "weighted_league_percentile",
            "players_in_group",
            "group_minutes",
        ],
    )


def main():
    print("=" * 118)
    print(
        "JAISSLE STATISTICAL PERCENTILE COMPARISON — AL-AHLI 25/26 vs NEWCASTLE 26/27"
        .center(118)
    )
    print("=" * 118)

    newcastle = pd.read_parquet(
        NEWCASTLE_PERCENTILES_FILE
    )

    saudi = pd.read_parquet(
        SAUDI_PERCENTILES_FILE
    )

    newcastle_profile = build_unit_profile(
        newcastle,
        lambda df: (
            pd.to_numeric(
                df["team_id"],
                errors="coerce",
            )
            == NEWCASTLE_TEAM_ID
        ),
        "Newcastle 26/27",
    )

    if newcastle_profile.empty:
        raise RuntimeError(
            "No Newcastle statistical profile rows were created."
        )

    al_ahli_profile = build_unit_profile(
        saudi,
        lambda df: (
            pd.to_numeric(
                df["team_id"],
                errors="coerce",
            )
            == AL_AHLI_TEAM_ID
        ),
        "Al-Ahli 25/26",
    )
    
    if al_ahli_profile.empty:
        raise RuntimeError(
            "No Al-Ahli statistical profile rows were created. "
            "Check the printed Saudi team names and eligible Al-Ahli players."
        )

    comparison = (
        al_ahli_profile[
            [
                "position_group",
                "metric",
                "metric_label",
                "weighted_league_percentile",
                "players_in_group",
                "group_minutes",
            ]
        ]
        .rename(
            columns={
                "weighted_league_percentile":
                    "al_ahli_saudi_percentile",
                "players_in_group":
                    "al_ahli_players",
                "group_minutes":
                    "al_ahli_minutes",
            }
        )
        .merge(
            newcastle_profile[
                [
                    "position_group",
                    "metric",
                    "metric_label",
                    "weighted_league_percentile",
                    "players_in_group",
                    "group_minutes",
                ]
            ].rename(
                columns={
                    "weighted_league_percentile":
                        "newcastle_epl_percentile",
                    "players_in_group":
                        "newcastle_players",
                    "group_minutes":
                        "newcastle_minutes",
                }
            ),
            on=[
                "position_group",
                "metric",
                "metric_label",
            ],
            how="outer",
        )
    )

    comparison[
        "percentile_difference_newcastle_minus_al_ahli"
    ] = (
        comparison[
            "newcastle_epl_percentile"
        ]
        - comparison[
            "al_ahli_saudi_percentile"
        ]
    ).round(
        1
    )

    order_map = {
        role: i
        for i, role in enumerate(
            POSITION_ORDER
        )
    }

    comparison["_order"] = comparison[
        "position_group"
    ].map(
        order_map
    )

    comparison = (
        comparison
        .sort_values(
            [
                "_order",
                "metric_label",
            ]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    for position_group in POSITION_ORDER:
        group = comparison[
            comparison["position_group"]
            == position_group
        ].copy()

        if group.empty:
            continue

        print()
        print("=" * 118)
        print(
            position_group.center(118)
        )
        print("=" * 118)

        print(
            f"{'Metric':<40}"
            f"{'Al-Ahli Saudi Pctl':>21}"
            f"{'Newcastle EPL Pctl':>23}"
            f"{'Δ Pctl':>12}"
        )

        print("-" * 96)

        for _, row in group.iterrows():
            a = row["al_ahli_saudi_percentile"]
            n = row["newcastle_epl_percentile"]
            d = row[
                "percentile_difference_newcastle_minus_al_ahli"
            ]

            a_txt = (
                f"{a:.1f}"
                if pd.notna(a)
                else "--"
            )

            n_txt = (
                f"{n:.1f}"
                if pd.notna(n)
                else "--"
            )

            d_txt = (
                f"{d:+.1f}"
                if pd.notna(d)
                else "--"
            )

            print(
                f"{row['metric_label']:<40}"
                f"{a_txt:>21}"
                f"{n_txt:>23}"
                f"{d_txt:>12}"
            )

        comparable = group[
            group[
                "al_ahli_saudi_percentile"
            ].notna()
            & group[
                "newcastle_epl_percentile"
            ].notna()
        ]

        if not comparable.empty:
            print()
            print(
                "Average comparable percentile: "
                f"Al-Ahli "
                f"{comparable['al_ahli_saudi_percentile'].mean():.1f} | "
                f"Newcastle "
                f"{comparable['newcastle_epl_percentile'].mean():.1f}"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 118)
    print("SAVED".center(118))
    print("=" * 118)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
