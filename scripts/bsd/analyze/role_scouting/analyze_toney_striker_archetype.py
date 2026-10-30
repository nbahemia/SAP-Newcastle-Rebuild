from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = ROOT / "data" / "processed" / "bsd"

ENRICHED_FILE = PROCESSED_DIR / "player_season_metrics_enriched.parquet"
PERCENTILES_FILE = PROCESSED_DIR / "player_percentiles.parquet"
OUTPUT_FILE = PROCESSED_DIR / "newcastle_toney_striker_comparison.parquet"

NEWCASTLE_TEAM = "Newcastle United"
NEWCASTLE_STRIKERS = ["Yoane Wissa", "William Osula", "Bazoumana Touré"]

# Ivan Toney 2026/27 FotMob reference supplied manually by the user.
# All values are per 90 unless a percentage is stated explicitly.
TONEY_REFERENCE = {
    "player_name": "Ivan Toney",
    "minutes": 774.0,
    "goals_per90": 1.40,
    "xg_per90": 1.37,
    "xgot_per90": 1.28,
    "penalty_goals_per90": 0.58,
    "npxg_per90": 0.82,
    "shots_per90": 4.19,
    "shots_on_target_per90": 2.56,
    "headed_shots_per90": 0.93,
    "assists_per90": 0.23,
    "xa_per90": 0.15,
    "successful_passes_per90": 14.65,
    "successful_passes_pct": 74.1,
    "accurate_long_balls_per90": 0.58,
    "accurate_long_balls_pct": 50.0,
    "line_breaking_passes_per90": 1.63,
    "chances_created_per90": 1.28,
    "big_chances_created_per90": 0.47,
    "successful_dribbles_per90": 0.35,
    "successful_dribbles_pct": 60.0,
    "duels_won_per90": 4.42,
    "duels_won_pct": 55.1,
    "aerials_won_per90": 1.63,
    "aerials_won_pct": 63.6,
    "touches_per90": 33.26,
    "box_touches_per90": 6.63,
    "dispossessed_per90": 1.05,
    "fouls_won_per90": 1.98,
    "penalties_won_per90": 0.35,
    "defensive_contributions_per90": 1.28,
    "tackles_per90": 0.47,
    "interceptions_per90": 0.12,
    "blocked_shots_per90": 0.12,
    "fouls_committed_per90": 1.16,
    "recoveries_per90": 1.63,
    "dribbled_past_per90": 0.35,
    "clearances_per90": 0.58,
}

# Role weights. Missing metrics are automatically excluded and the score is
# renormalized over the metrics that really exist in the Newcastle dataset.
ROLE_WEIGHTS = {
    "npxg_per90": 0.18,
    "xg_per90": 0.10,
    "shots_per90": 0.10,
    "shots_on_target_per90": 0.07,
    "box_touches_per90": 0.12,
    "aerials_won_per90": 0.10,
    "duels_won_per90": 0.08,
    "line_breaking_passes_per90": 0.08,
    "chances_created_per90": 0.06,
    "xa_per90": 0.04,
    "successful_passes_per90": 0.03,
    "touches_per90": 0.02,
    "dispossessed_per90": 0.02,
}

DISPLAY = {
    "goals_per90": "Goals",
    "xg_per90": "xG",
    "xgot_per90": "xGOT",
    "npxg_per90": "Non-penalty xG",
    "shots_per90": "Shots",
    "shots_on_target_per90": "Shots on Target",
    "headed_shots_per90": "Headed Shots",
    "assists_per90": "Assists",
    "xa_per90": "xA",
    "successful_passes_per90": "Successful Passes",
    "successful_passes_pct": "Pass Accuracy %",
    "accurate_long_balls_per90": "Accurate Long Balls",
    "accurate_long_balls_pct": "Long Ball Accuracy %",
    "line_breaking_passes_per90": "Line-Breaking Passes",
    "chances_created_per90": "Chances Created",
    "big_chances_created_per90": "Big Chances Created",
    "successful_dribbles_per90": "Successful Dribbles",
    "successful_dribbles_pct": "Dribble Success %",
    "duels_won_per90": "Duels Won",
    "duels_won_pct": "Duel Win %",
    "aerials_won_per90": "Aerial Duels Won",
    "aerials_won_pct": "Aerial Win %",
    "touches_per90": "Touches",
    "box_touches_per90": "Touches in Opposition Box",
    "dispossessed_per90": "Dispossessed",
    "fouls_won_per90": "Fouls Won",
    "recoveries_per90": "Recoveries",
    "interceptions_per90": "Interceptions",
}

ALIASES = {
    "goals_per90": ["goals_per90"],
    "xg_per90": ["xg_per90"],
    "shots_per90": ["shots_per90"],
    "shots_on_target_per90": ["shots_on_target_per90"],
    "assists_per90": ["assists_per90"],
    "xa_per90": ["xa_per90"],
    "successful_passes_per90": ["successful_passes_per90"],
    "successful_passes_pct": ["pass_accuracy_pct", "successful_passes_pct"],
    "accurate_long_balls_per90": ["accurate_long_balls_per90"],
    "duels_won_per90": ["duels_won_per90"],
    "aerials_won_per90": ["aerials_won_per90"],
    "touches_per90": ["touches_per90"],
    "dispossessed_per90": ["possession_lost_per90", "dispossessed_per90"],
    "recoveries_per90": ["recoveries_per90"],
    "interceptions_per90": ["interceptions_per90"],
}


def banner(text: str, width: int = 118) -> None:
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def lower_is_better(metric: str) -> bool:
    return metric == "dispossessed_per90"


def toney_similarity(candidate: float, reference: float, metric: str) -> float:
    if pd.isna(candidate) or pd.isna(reference):
        return np.nan
    candidate = float(candidate)
    reference = float(reference)
    if lower_is_better(metric):
        if candidate <= 0 or candidate <= reference:
            return 100.0
        return max(0.0, min(100.0, reference / candidate * 100.0))
    if reference <= 0:
        return np.nan
    if candidate >= reference:
        return 100.0
    return max(0.0, min(100.0, candidate / reference * 100.0))


def weighted_role_fit(row: pd.Series, metric_columns: dict[str, str]) -> tuple[float, float]:
    weighted_sum = 0.0
    weight_used = 0.0
    for metric, weight in ROLE_WEIGHTS.items():
        col = metric_columns.get(metric)
        if col is None:
            continue
        value = row.get(col, np.nan)
        ref = TONEY_REFERENCE.get(metric, np.nan)
        if pd.isna(value) or pd.isna(ref):
            continue
        score = toney_similarity(value, ref, metric)
        if pd.isna(score):
            continue
        weighted_sum += score * weight
        weight_used += weight
    if weight_used == 0:
        return np.nan, 0.0
    return weighted_sum / weight_used, weight_used


def load_newcastle() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not ENRICHED_FILE.exists():
        raise FileNotFoundError(f"Missing file:\n{ENRICHED_FILE}")
    enriched = pd.read_parquet(ENRICHED_FILE)
    percentiles = pd.read_parquet(PERCENTILES_FILE) if PERCENTILES_FILE.exists() else None
    return enriched, percentiles


def filter_strikers(df: pd.DataFrame) -> pd.DataFrame:
    name_col = find_first_existing(df, ["player_name", "name"])
    if name_col is None:
        raise RuntimeError("Could not find player_name/name in Newcastle metrics.")
    mask = df[name_col].isin(NEWCASTLE_STRIKERS)
    if "team_name" in df.columns:
        mask &= df["team_name"].eq(NEWCASTLE_TEAM)
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError("Configured Newcastle strikers were not found.")
    if name_col != "player_name":
        out = out.rename(columns={name_col: "player_name"})
    return out


def print_toney_reference() -> None:
    banner("IVAN TONEY 2026/27 — ROLE REFERENCE")
    print(f"Minutes: {TONEY_REFERENCE['minutes']:.0f}\n")
    groups = {
        "BOX / SHOOTING": [
            "goals_per90", "xg_per90", "npxg_per90", "shots_per90",
            "shots_on_target_per90", "headed_shots_per90", "box_touches_per90",
        ],
        "LINK / CREATION": [
            "xa_per90", "successful_passes_per90", "successful_passes_pct",
            "accurate_long_balls_per90", "line_breaking_passes_per90",
            "chances_created_per90", "big_chances_created_per90",
        ],
        "PHYSICAL / POSSESSION": [
            "duels_won_per90", "duels_won_pct", "aerials_won_per90",
            "aerials_won_pct", "touches_per90", "dispossessed_per90",
            "fouls_won_per90",
        ],
    }
    for title, metrics in groups.items():
        print(title)
        for metric in metrics:
            print(f"  {DISPLAY.get(metric, metric):32s}{TONEY_REFERENCE[metric]:8.2f}")
        print()


def build_comparison(newcastle: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    metric_columns = {
        metric: find_first_existing(newcastle, candidates)
        for metric, candidates in ALIASES.items()
    }
    available = {
        metric: col for metric, col in metric_columns.items()
        if col is not None and metric in TONEY_REFERENCE
    }
    records = []
    for _, row in newcastle.iterrows():
        fit, weight_used = weighted_role_fit(row, available)
        rec = {
            "player_name": row["player_name"],
            "minutes": row.get("minutes", np.nan),
            "toney_role_fit_score": fit,
            "role_weight_coverage": weight_used,
        }
        for metric, col in available.items():
            value = row.get(col, np.nan)
            rec[f"{metric}_candidate"] = value
            rec[f"{metric}_toney"] = TONEY_REFERENCE[metric]
            rec[f"{metric}_fit"] = toney_similarity(value, TONEY_REFERENCE[metric], metric)
        records.append(rec)
    result = pd.DataFrame(records)
    return result.sort_values("toney_role_fit_score", ascending=False, na_position="last"), available


def print_comparison(newcastle: pd.DataFrame, comparison: pd.DataFrame, available: dict[str, str]) -> None:
    banner("NEWCASTLE STRIKERS vs TONEY ARCHETYPE")
    print("Role-fit score is a style/trait-attainment score only; it is not league-strength adjusted.\n")

    summary = comparison[["player_name", "minutes", "toney_role_fit_score", "role_weight_coverage"]].copy()
    summary["toney_role_fit_score"] = summary["toney_role_fit_score"].round(1)
    summary["role_weight_coverage"] = (summary["role_weight_coverage"] * 100).round(1)
    summary = summary.rename(columns={
        "toney_role_fit_score": "role_fit",
        "role_weight_coverage": "weight_coverage_pct",
    })
    print(summary.to_string(index=False))

    banner("OVERLAPPING RAW METRICS")
    lookup = newcastle.set_index("player_name")
    for metric, col in available.items():
        print(f"\n{DISPLAY.get(metric, metric)} (Toney = {TONEY_REFERENCE[metric]:.2f})")
        for name in NEWCASTLE_STRIKERS:
            if name not in lookup.index:
                continue
            row = lookup.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            value = row.get(col, np.nan)
            value_text = "--" if pd.isna(value) else f"{float(value):.2f}"
            fit = toney_similarity(value, TONEY_REFERENCE[metric], metric)
            fit_text = "--" if pd.isna(fit) else f"{fit:.1f}"
            print(f"  {name:22s}{value_text:>9s}   fit={fit_text:>6s}")


def print_reference_only(available: dict[str, str]) -> None:
    banner("TONEY TRAITS NOT AVAILABLE IN THE NEWCASTLE DATASET")
    important = [
        "npxg_per90", "box_touches_per90", "line_breaking_passes_per90",
        "chances_created_per90", "big_chances_created_per90", "headed_shots_per90",
        "duels_won_pct", "aerials_won_pct",
    ]
    missing = [m for m in important if m not in available]
    if not missing:
        print("All configured Toney role traits are available.")
        return
    for metric in missing:
        print(f"{DISPLAY.get(metric, metric):32s}Toney reference = {TONEY_REFERENCE[metric]:.2f}")
    print("\nThese remain reference-only. Do not fill missing Newcastle values with zero.")


def print_percentile_context(percentiles: pd.DataFrame | None) -> None:
    if percentiles is None or "player_name" not in percentiles.columns:
        return
    players = percentiles[percentiles["player_name"].isin(NEWCASTLE_STRIKERS)].copy()
    if players.empty:
        return
    banner("NEWCASTLE EPL STRIKER PERCENTILE CONTEXT")
    preferred = [
        "player_name", "minutes", "goals_per90_percentile", "xg_per90_percentile",
        "shots_per90_percentile", "shots_on_target_per90_percentile",
        "touches_per90_percentile", "aerials_won_per90_percentile",
        "duels_won_per90_percentile", "progressive_carries_per90_percentile",
    ]
    cols = [c for c in preferred if c in players.columns]
    sort_col = "minutes" if "minutes" in cols else "player_name"
    print(players[cols].sort_values(sort_col, ascending=False).to_string(index=False))


def print_interpretation() -> None:
    banner("INTERPRETATION GUIDE")
    print("1. Check whether Newcastle's strikers reproduce Toney's box-threat metrics.")
    print("2. Check whether they also provide his physical outlet traits: aerials + duels.")
    print("3. Treat unavailable link-play traits as explicit future scouting requirements.")
    print("4. Strong role fit but weaker xG/shot output supports the 'service/creation' hypothesis.")
    print("5. Weak box threat plus weak physical/link traits supports striker as a true personnel need.")


def main() -> None:
    banner("TONEY STRIKER ARCHETYPE — NEWCASTLE ROLE-FIT ANALYSIS")
    enriched, percentiles = load_newcastle()
    newcastle = filter_strikers(enriched)
    print_toney_reference()
    comparison, available = build_comparison(newcastle)
    print_comparison(newcastle, comparison, available)
    print_reference_only(available)
    print_percentile_context(percentiles)
    print_interpretation()
    comparison.to_parquet(OUTPUT_FILE, index=False)
    banner("SAVED")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
