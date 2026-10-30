from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

PERCENTILES_FILE = PROCESSED_DIR / "player_percentiles.parquet"
ROLE_SCORES_FILE = PROCESSED_DIR / "player_role_scores.parquet"

PLAYER_OUTPUT_FILE = PROCESSED_DIR / "newcastle_player_diagnostic.parquet"
UNIT_OUTPUT_FILE = PROCESSED_DIR / "newcastle_unit_summary.parquet"

NEWCASTLE_TEAM_NAME = "Newcastle United"
MINUTES_WEIGHT_CAP = 600

DEPTH_THRESHOLDS = {
    "players_180_plus": 180,
    "players_360_plus": 360,
    "players_540_plus": 540,
}

POSITION_ORDER = ["GK", "CB", "FB/WB", "DM/CM", "AM", "W", "ST"]

POSITION_METRICS = {
    "GK": [],
    "CB": [
        "successful_passes_per90_percentile",
        "pass_accuracy_pct_percentile",
        "aerials_won_per90_percentile",
        "duels_won_per90_percentile",
        "recoveries_per90_percentile",
        "interceptions_per90_percentile",
        "tackles_won_per90_percentile",
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
    "AM": [
        "xg_per90_percentile",
        "progressive_carries_per90_percentile",
        "accurate_opposition_half_passes_per90_percentile",
        "touches_per90_percentile",
        "recoveries_per90_percentile",
        "interceptions_per90_percentile",
    ],
    "W": [
        "xg_per90_percentile",
        "shots_per90_percentile",
        "progressive_carries_per90_percentile",
        "total_cross_per90_percentile",
        "accurate_opposition_half_passes_per90_percentile",
        "touches_per90_percentile",
        "recoveries_per90_percentile",
        "interceptions_per90_percentile",
    ],
    "ST": [
        "xg_per90_percentile",
        "shots_in_box_per90_percentile",
        "shots_per90_percentile",
        "goals_per90_percentile",
        "aerials_won_per90_percentile",
        "duels_won_per90_percentile",
        "progressive_carries_per90_percentile",
        "touches_per90_percentile",
        "recoveries_per90_percentile",
        "interceptions_per90_percentile",
    ],
}

METRIC_LABELS = {
    "successful_passes_per90_percentile": "Successful Passes",
    "pass_accuracy_pct_percentile": "Pass Accuracy",
    "aerials_won_per90_percentile": "Aerials Won",
    "duels_won_per90_percentile": "Duels Won",
    "recoveries_per90_percentile": "Recoveries",
    "interceptions_per90_percentile": "Interceptions",
    "tackles_won_per90_percentile": "Tackles Won",
    "progressive_carries_per90_percentile": "Progressive Carries",
    "total_progressive_ball_carries_distance_per90_percentile": "Progressive Carry Distance",
    "accurate_opposition_half_passes_per90_percentile": "Accurate Opposition-Half Passes",
    "total_cross_per90_percentile": "Crosses",
    "total_long_balls_per90_percentile": "Long Balls",
    "touches_per90_percentile": "Touches",
    "xg_per90_percentile": "xG",
    "shots_per90_percentile": "Shots",
    "shots_in_box_per90_percentile": "Shots in Box",
    "goals_per90_percentile": "Goals",
}

def percentile_band(value):
    if pd.isna(value):
        return "N/A"
    if value >= 80:
        return "Elite"
    if value >= 65:
        return "Strong"
    if value >= 45:
        return "Average"
    if value >= 30:
        return "Below Average"
    return "Weak"

def weighted_percentile_mean(group, metric, weight_cap=MINUTES_WEIGHT_CAP):
    if metric not in group.columns:
        return np.nan

    valid = group[
        group[metric].notna()
        & group["minutes"].notna()
        & (group["minutes"] > 0)
    ].copy()

    if valid.empty:
        return np.nan

    valid["analysis_weight"] = valid["minutes"].clip(lower=0, upper=weight_cap).astype(float)
    total_weight = valid["analysis_weight"].sum()

    if total_weight <= 0:
        return np.nan

    return float(
        (valid[metric] * valid["analysis_weight"]).sum() / total_weight
    )

def role_scores_wide(role_scores):
    if role_scores.empty:
        return pd.DataFrame(columns=["player_id"])

    required = {"player_id", "role", "role_score"}
    missing = required - set(role_scores.columns)

    if missing:
        raise ValueError(
            f"player_role_scores.parquet is missing required columns: {sorted(missing)}"
        )

    pivot = (
        role_scores[["player_id", "role", "role_score"]]
        .drop_duplicates(["player_id", "role"], keep="first")
        .pivot(index="player_id", columns="role", values="role_score")
        .reset_index()
    )

    return pivot.rename(
        columns={
            "jaissle_striker": "jaissle_striker_score",
            "vertical_fullback": "vertical_fullback_score",
            "controller_midfielder": "controller_midfielder_score",
        }
    )

def build_player_diagnostic(players, role_scores):
    newcastle = players[
        players["team_name"] == NEWCASTLE_TEAM_NAME
    ].copy()

    newcastle = newcastle.merge(
        role_scores_wide(role_scores),
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    preferred = [
        "player_id",
        "player_name",
        "team_name",
        "position_group",
        "tactical_position",
        "primary_position_side",
        "minutes",
        "position_group_share",
        "position_side_share",
        "observed_position_groups",
        "observed_position_sides",
        "age_at_season_start",
        "market_value_eur",
        "jaissle_striker_score",
        "vertical_fullback_score",
        "controller_midfielder_score",
    ]

    metrics = []
    for metric_list in POSITION_METRICS.values():
        for metric in metric_list:
            if metric in newcastle.columns and metric not in metrics:
                metrics.append(metric)

    preferred = [c for c in preferred if c in newcastle.columns]
    return newcastle[preferred + metrics].copy()

def build_unit_summary(diagnostic):
    rows = []

    for position_group in POSITION_ORDER:
        group = diagnostic[
            diagnostic["position_group"] == position_group
        ].copy()

        if group.empty:
            continue

        depth_counts = {
            label: int((group["minutes"].fillna(0) >= threshold).sum())
            for label, threshold in DEPTH_THRESHOLDS.items()
        }

        base = {
            "position_group": position_group,
            "players": len(group),
            "total_minutes": int(group["minutes"].fillna(0).sum()),
            **depth_counts,
        }

        metrics = POSITION_METRICS.get(position_group, [])

        if not metrics:
            rows.append(
                {
                    **base,
                    "metric": None,
                    "metric_label": None,
                    "weighted_percentile": np.nan,
                    "band": "N/A",
                }
            )
            continue

        for metric in metrics:
            if metric not in group.columns:
                continue

            value = weighted_percentile_mean(group, metric)

            rows.append(
                {
                    **base,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "weighted_percentile": round(value, 1) if pd.notna(value) else np.nan,
                    "band": percentile_band(value),
                }
            )

    return pd.DataFrame(rows)

def print_player_card(row, metrics):
    print(f"\n{row['player_name']}")

    for column, label in [
        ("tactical_position", "Tactical position"),
        ("minutes", "Minutes"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            if column == "minutes":
                value = int(value)
            print(f"  {label}: {value}")

    if pd.notna(row.get("position_group_share")):
        print(f"  Position confidence: {row['position_group_share']:.0%}")

    if pd.notna(row.get("position_side_share")):
        print(f"  Side confidence: {row['position_side_share']:.0%}")

    if pd.notna(row.get("observed_position_groups")):
        print(f"  Observed roles: {row['observed_position_groups']}")

    for column, label in [
        ("jaissle_striker_score", "Jaissle striker"),
        ("vertical_fullback_score", "Vertical fullback"),
        ("controller_midfielder_score", "Controller midfielder"),
    ]:
        if column in row.index and pd.notna(row[column]):
            print(f"  {label} score: {row[column]:.1f}")

    valid_metrics = [
        metric for metric in metrics
        if metric in row.index and pd.notna(row[metric])
    ]

    if valid_metrics:
        print("  Percentiles:")
        for metric in valid_metrics:
            label = METRIC_LABELS.get(metric, metric)
            print(f"    {label:<38}{row[metric]:>6.1f}")

def print_unit_summary(group_summary):
    if group_summary.empty:
        return

    first = group_summary.iloc[0]

    print("\n  UNIT PROFILE")

    metric_rows = group_summary[group_summary["metric"].notna()]

    if metric_rows.empty:
        print("    No configured unit percentile metrics.")
    else:
        for _, row in metric_rows.iterrows():
            print(
                f"    {row['metric_label']:<38}"
                f"{row['weighted_percentile']:>6.1f}  "
                f"{row['band']}"
            )

    print("\n  DEPTH")
    print(f"    Players: {int(first['players'])}")
    print(f"    Total minutes: {int(first['total_minutes'])}")
    print(f"    Players >= 180 min: {int(first['players_180_plus'])}")
    print(f"    Players >= 360 min: {int(first['players_360_plus'])}")
    print(f"    Players >= 540 min: {int(first['players_540_plus'])}")

def print_squad_diagnostic(diagnostic, unit_summary):
    print()
    print("=" * 100)
    print("NEWCASTLE UNITED SQUAD DIAGNOSTIC".center(100))
    print("=" * 100)

    for position_group in POSITION_ORDER:
        group = diagnostic[
            diagnostic["position_group"] == position_group
        ].copy()

        if group.empty:
            continue

        print()
        print("=" * 88)
        print(f"{position_group} — {NEWCASTLE_TEAM_NAME}")
        print("=" * 88)

        group = group.sort_values(
            "minutes",
            ascending=False,
            na_position="last",
        )

        metrics = POSITION_METRICS.get(position_group, [])

        for _, row in group.iterrows():
            print_player_card(row, metrics)
            print("-" * 88)

        print_unit_summary(
            unit_summary[
                unit_summary["position_group"] == position_group
            ]
        )
        
def print_all_group_analysis(
    unit_summary: pd.DataFrame,
):
    print()
    print("=" * 110)
    print(
        "NEWCASTLE UNITED — ALL POSITION GROUP ANALYSIS"
        .center(110)
    )
    print("=" * 110)

    for position_group in POSITION_ORDER:

        group = unit_summary[
            unit_summary[
                "position_group"
            ]
            == position_group
        ].copy()

        if group.empty:
            continue

        first = group.iloc[0]

        print()
        print("-" * 110)
        print(
            f"{position_group}"
        )
        print("-" * 110)

        print(
            f"Players: {int(first['players'])} | "
            f"Minutes: {int(first['total_minutes'])} | "
            f"180+: {int(first['players_180_plus'])} | "
            f"360+: {int(first['players_360_plus'])} | "
            f"540+: {int(first['players_540_plus'])}"
        )

        metric_rows = group[
            group["metric"].notna()
        ].copy()

        if metric_rows.empty:
            print(
                "No configured percentile metrics."
            )
            continue

        metric_rows = (
            metric_rows
            .sort_values(
                "weighted_percentile",
                ascending=False,
            )
        )

        print()
        print(
            f"{'Metric':<42}"
            f"{'Pct':>8}"
            f"{'Band':>18}"
        )

        print(
            "-" * 68
        )

        for _, row in metric_rows.iterrows():

            print(
                f"{row['metric_label']:<42}"
                f"{row['weighted_percentile']:>8.1f}"
                f"{row['band']:>18}"
            )

        strongest = (
            metric_rows.iloc[0]
        )

        weakest = (
            metric_rows.iloc[-1]
        )

        average_percentile = (
            metric_rows[
                "weighted_percentile"
            ].mean()
        )

        print()
        print(
            f"Group average: "
            f"{average_percentile:.1f}"
        )

        print(
            f"Strongest metric: "
            f"{strongest['metric_label']} "
            f"({strongest['weighted_percentile']:.1f})"
        )

        print(
            f"Weakest metric: "
            f"{weakest['metric_label']} "
            f"({weakest['weighted_percentile']:.1f})"
        )

def main():
    for path in [PERCENTILES_FILE, ROLE_SCORES_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    players = pd.read_parquet(PERCENTILES_FILE)
    role_scores = pd.read_parquet(ROLE_SCORES_FILE)

    required = {
        "player_id",
        "player_name",
        "team_name",
        "position_group",
        "minutes",
    }

    missing = required - set(players.columns)
    if missing:
        raise ValueError(
            f"player_percentiles.parquet is missing required columns: {sorted(missing)}"
        )

    diagnostic = build_player_diagnostic(players, role_scores)
    unit_summary = build_unit_summary(diagnostic)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    diagnostic.to_parquet(
        PLAYER_OUTPUT_FILE,
        index=False,
    )

    unit_summary.to_parquet(
        UNIT_OUTPUT_FILE,
        index=False,
    )

    print_squad_diagnostic(
        diagnostic,
        unit_summary,
    )
    
    print_all_group_analysis(
        unit_summary
    )

    print()
    print("=" * 100)
    print("SAVED OUTPUTS".center(100))
    print("=" * 100)
    print(PLAYER_OUTPUT_FILE)
    print(UNIT_OUTPUT_FILE)

    print()
    print(
        f"Unit percentiles are minutes-weighted with each player's "
        f"weight capped at {MINUTES_WEIGHT_CAP} minutes."
    )
    print(
        "This script is descriptive only; it does not assign recruitment-need ratings."
    )

if __name__ == "__main__":
    main()
