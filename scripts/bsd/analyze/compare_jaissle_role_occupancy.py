from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

NEWCASTLE_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "player_position_history.parquet"
)

NEWCASTLE_METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
    / "player_season_metrics_enriched.parquet"
)

SAUDI_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bsd"
    / "saudi_2025_26"
    / "player_position_history.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "jaissle_role_occupancy_comparison.parquet"
)

LEAGUE_PERCENTILE_FILE = (
    OUTPUT_DIR
    / "jaissle_role_occupancy_league_percentiles.parquet"
)

MATCH_OUTPUT_FILE = (
    OUTPUT_DIR
    / "jaissle_role_occupancy_match_counts.parquet"
)

MIN_CLASSIFIED_PLAYERS_PER_MATCH = 8
MIN_VALID_MATCHES_FOR_PERCENTILE = 3

POSITION_ORDER = [
    "GK",
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
]


def normalize_position_group_column(df: pd.DataFrame, label: str):
    df = df.copy()

    if "position_group" in df.columns:
        return df

    for candidate in [
        "inferred_position_group",
        "primary_position_group",
        "tactical_position_group",
    ]:
        if candidate in df.columns:
            print(
                f"{label}: using '{candidate}' "
                "as position_group"
            )
            df["position_group"] = df[candidate]
            return df

    raise RuntimeError(
        f"{label}: no position-group column found.\n"
        f"Available columns: {list(df.columns)}"
    )


def build_team_match_role_table(history: pd.DataFrame):
    data = history[
        history["position_group"].notna()
    ].copy()

    match_totals = (
        data
        .groupby(["team_id", "event_id"])
        .size()
        .rename("classified_players")
        .reset_index()
    )

    valid_matches = match_totals[
        match_totals["classified_players"]
        >= MIN_CLASSIFIED_PLAYERS_PER_MATCH
    ][["team_id", "event_id"]]

    data = data.merge(
        valid_matches,
        on=["team_id", "event_id"],
        how="inner",
    )

    role_counts = (
        data
        .groupby(
            [
                "team_id",
                "event_id",
                "position_group",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for role in POSITION_ORDER:
        if role not in role_counts.columns:
            role_counts[role] = 0

    role_counts["classified_players"] = role_counts[
        POSITION_ORDER
    ].sum(axis=1)

    for role in POSITION_ORDER:
        role_counts[f"{role}_share"] = (
            role_counts[role]
            / role_counts["classified_players"]
        )

    return role_counts


def build_league_role_percentiles(history: pd.DataFrame):
    match_roles = build_team_match_role_table(history)

    share_cols = [
        f"{role}_share"
        for role in POSITION_ORDER
    ]

    team_summary = (
        match_roles
        .groupby("team_id")[share_cols]
        .mean()
        .reset_index()
    )

    valid_match_counts = (
        match_roles
        .groupby("team_id")
        .size()
        .rename("valid_matches")
        .reset_index()
    )

    team_summary = team_summary.merge(
        valid_match_counts,
        on="team_id",
        how="left",
    )

    percentile_pool = team_summary[
        team_summary["valid_matches"]
        >= MIN_VALID_MATCHES_FOR_PERCENTILE
    ].copy()

    for role in POSITION_ORDER:
        share_col = f"{role}_share"
        pct_col = f"{role}_share_percentile"

        percentile_pool[pct_col] = (
            percentile_pool[share_col]
            .rank(
                pct=True,
                method="average",
            )
            * 100.0
        ).round(1)

    return (
        team_summary,
        percentile_pool,
        match_roles,
    )


def get_team_id_from_metrics(metrics: pd.DataFrame, team_name: str):
    ids = (
        metrics[
            metrics["team_name"]
            .astype(str)
            .str.contains(
                team_name,
                case=False,
                na=False,
                regex=False,
            )
        ]["team_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"Expected exactly one team ID for {team_name}, "
            f"found {ids}"
        )

    return int(ids[0])


def get_team_id_from_history(history: pd.DataFrame, team_name: str):
    if "team_name" not in history.columns:
        raise RuntimeError(
            "History file has no team_name column."
        )

    matches = (
        history[
            history["team_name"]
            .astype(str)
            .str.contains(
                team_name,
                case=False,
                na=False,
                regex=False,
            )
        ]["team_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(matches) == 0 and "Ahli" in team_name:
        matches = (
            history[
                history["team_name"]
                .astype(str)
                .str.contains(
                    "Ahli",
                    case=False,
                    na=False,
                    regex=False,
                )
            ]["team_id"]
            .dropna()
            .astype(int)
            .unique()
        )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one team ID for {team_name}, "
            f"found {matches}"
        )

    return int(matches[0])


def build_club_summary(
    team_summary: pd.DataFrame,
    percentile_pool: pd.DataFrame,
    team_id: int,
    label: str,
):
    base = team_summary[
        team_summary["team_id"] == team_id
    ]

    pct = percentile_pool[
        percentile_pool["team_id"] == team_id
    ]

    if base.empty:
        raise RuntimeError(
            f"No team summary row for {label}."
        )

    if pct.empty:
        raise RuntimeError(
            f"{label} has fewer than "
            f"{MIN_VALID_MATCHES_FOR_PERCENTILE} valid matches."
        )

    base_row = base.iloc[0]
    pct_row = pct.iloc[0]

    rows = []

    for role in POSITION_ORDER:
        rows.append(
            {
                "team": label,
                "position_group": role,
                "role_share_pct": round(
                    base_row[f"{role}_share"] * 100.0,
                    1,
                ),
                "league_percentile": float(
                    pct_row[f"{role}_share_percentile"]
                ),
                "valid_matches": int(
                    base_row["valid_matches"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 116)
    print(
        "JAISSLE ROLE OCCUPANCY + LEAGUE PERCENTILES — AL-AHLI 25/26 vs NEWCASTLE 26/27"
        .center(116)
    )
    print("=" * 116)

    newcastle_history = pd.read_parquet(
        NEWCASTLE_HISTORY_FILE
    )

    newcastle_metrics = pd.read_parquet(
        NEWCASTLE_METRICS_FILE
    )

    saudi_history = pd.read_parquet(
        SAUDI_HISTORY_FILE
    )

    newcastle_history = normalize_position_group_column(
        newcastle_history,
        "Newcastle/EPL",
    )

    saudi_history = normalize_position_group_column(
        saudi_history,
        "Saudi",
    )

    newcastle_team_id = get_team_id_from_metrics(
        newcastle_metrics,
        "Newcastle United",
    )

    al_ahli_team_id = get_team_id_from_history(
        saudi_history,
        "Al-Ahli",
    )

    print(f"\nNewcastle team ID: {newcastle_team_id}")
    print(f"Al-Ahli team ID: {al_ahli_team_id}")

    (
        epl_team_summary,
        epl_percentile_pool,
        epl_match_roles,
    ) = build_league_role_percentiles(
        newcastle_history
    )

    (
        saudi_team_summary,
        saudi_percentile_pool,
        saudi_match_roles,
    ) = build_league_role_percentiles(
        saudi_history
    )

    newcastle = build_club_summary(
        epl_team_summary,
        epl_percentile_pool,
        newcastle_team_id,
        "Newcastle 26/27",
    )

    al_ahli = build_club_summary(
        saudi_team_summary,
        saudi_percentile_pool,
        al_ahli_team_id,
        "Al-Ahli 25/26",
    )

    comparison = (
        al_ahli[
            [
                "position_group",
                "role_share_pct",
                "league_percentile",
                "valid_matches",
            ]
        ]
        .rename(
            columns={
                "role_share_pct": "al_ahli_share_pct",
                "league_percentile": "al_ahli_saudi_percentile",
                "valid_matches": "al_ahli_valid_matches",
            }
        )
        .merge(
            newcastle[
                [
                    "position_group",
                    "role_share_pct",
                    "league_percentile",
                    "valid_matches",
                ]
            ].rename(
                columns={
                    "role_share_pct": "newcastle_share_pct",
                    "league_percentile": "newcastle_epl_percentile",
                    "valid_matches": "newcastle_valid_matches",
                }
            ),
            on="position_group",
            how="outer",
        )
    )

    comparison[
        "share_difference_newcastle_minus_al_ahli"
    ] = (
        comparison["newcastle_share_pct"]
        - comparison["al_ahli_share_pct"]
    ).round(1)

    order = {
        role: i
        for i, role in enumerate(POSITION_ORDER)
    }

    comparison["_order"] = comparison["position_group"].map(order)
    comparison = (
        comparison
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    print()
    print(
        f"Minimum classified players per match: "
        f"{MIN_CLASSIFIED_PLAYERS_PER_MATCH}"
    )

    print(
        f"Minimum valid matches for league percentile: "
        f"{MIN_VALID_MATCHES_FOR_PERCENTILE}"
    )

    print()
    print("=" * 116)
    print("NORMALIZED ROLE OCCUPANCY + LEAGUE PERCENTILES".center(116))
    print("=" * 116)

    print(
        f"{'Role':<10}"
        f"{'Al-Ahli %':>12}"
        f"{'Saudi Pctl':>13}"
        f"{'Newcastle %':>15}"
        f"{'EPL Pctl':>11}"
        f"{'Δ Share':>12}"
    )

    print("-" * 73)

    for _, row in comparison.iterrows():
        print(
            f"{row['position_group']:<10}"
            f"{row['al_ahli_share_pct']:>11.1f}%"
            f"{row['al_ahli_saudi_percentile']:>13.1f}"
            f"{row['newcastle_share_pct']:>14.1f}%"
            f"{row['newcastle_epl_percentile']:>11.1f}"
            f"{row['share_difference_newcastle_minus_al_ahli']:>11.1f}"
        )

    print()
    print(
        f"Al-Ahli valid matches: "
        f"{int(comparison['al_ahli_valid_matches'].iloc[0])}"
    )

    print(
        f"Newcastle valid matches: "
        f"{int(comparison['newcastle_valid_matches'].iloc[0])}"
    )

    print(
        f"Saudi percentile pool teams: "
        f"{len(saudi_percentile_pool)}"
    )

    print(
        f"EPL percentile pool teams: "
        f"{len(epl_percentile_pool)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    league_percentiles = pd.concat(
        [
            epl_percentile_pool.assign(
                league="Premier League 26/27"
            ),
            saudi_percentile_pool.assign(
                league="Saudi Pro League 25/26"
            ),
        ],
        ignore_index=True,
    )

    league_percentiles.to_parquet(
        LEAGUE_PERCENTILE_FILE,
        index=False,
    )

    all_matches = pd.concat(
        [
            epl_match_roles.assign(
                league="Premier League 26/27"
            ),
            saudi_match_roles.assign(
                league="Saudi Pro League 25/26"
            ),
        ],
        ignore_index=True,
    )

    all_matches.to_parquet(
        MATCH_OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 116)
    print("SAVED".center(116))
    print("=" * 116)

    print(OUTPUT_FILE)
    print(LEAGUE_PERCENTILE_FILE)
    print(MATCH_OUTPUT_FILE)


if __name__ == "__main__":
    main()
