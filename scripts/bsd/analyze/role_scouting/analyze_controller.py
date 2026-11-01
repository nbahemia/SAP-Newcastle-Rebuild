
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BSD = ROOT / "data" / "processed" / "bsd"
SHORTLIST = BSD / "shortlist"

# Recruitment universe
POOL_CSV = SHORTLIST / "holding_pivot.csv"
EUROPE_PARQUET = BSD / "europe_2025_26_player_season_metrics.parquet"

# Current Newcastle diagnostic layer
CURRENT_CANDIDATES = [
    BSD / "player_season_metrics_enriched.parquet",
    BSD / "player_team_season_metrics.parquet",
    BSD / "player_season_metrics.parquet",
]

POSITION_HISTORY = BSD / "player_position_history.parquet"

OUT_CSV = SHORTLIST / "holding_controller_jaissle_detailed.csv"
OUT_HTML = SHORTLIST / "holding_controller_jaissle_detailed.html"

MIN_HISTORICAL_MINUTES = 900
MIN_CURRENT_MINUTES = 180

# Intended Deep Transition Controller model.
# IMPORTANT:
# Final comparison uses only metrics supported in BOTH datasets.
INTENDED_WEIGHTS = {
    "recoveries": 0.15,
    "interceptions": 0.15,
    "tackles": 0.10,
    "line_breaking": 0.20,
    "successful_passes": 0.10,
    "dispossessed": 0.15,     # inverse
    "pass_accuracy": 0.05,
    "touches": 0.10,
}

ALIASES = {
    "player": [
        "player_name", "Player", "Player Name", "name"
    ],
    "team": [
        "team_name", "club_name", "display_club_name", "Club", "Team"
    ],
    "age": ["age", "Age"],
    "minutes": ["minutes", "Minutes", "mins", "minutes_played"],
    "value": ["market_value_eur_m", "Value €M", "value_eur_m"],

    "recoveries": [
        "recoveries_per90", "recoveries_p90"
    ],
    "interceptions": [
        "interceptions_per90", "interceptions_p90"
    ],
    "tackles": [
        "tackles_per90", "tackles_p90",
        "tackles_won_per90", "tackles_won_p90"
    ],

    "line_breaking": [
        "line_breaking_passes_per90",
        "line_breaking_passes_p90",
        "line_breaking_per90",
    ],
    "successful_passes": [
        "successful_passes_per90",
        "successful_passes_p90",
        "passes_completed_per90",
        "completed_passes_per90",
    ],
    "pass_accuracy": [
        "pass_accuracy_pct",
        "pass_accuracy",
        "successful_passes_pct",
        "passing_accuracy_pct",
    ],

    "dispossessed": [
        "dispossessed_per90",
        "dispossessions_per90",
        "times_dispossessed_per90",
    ],
    "touches": [
        "touches_per90", "touches_p90"
    ],
}


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def norm_name(s) -> str:
    return (
        str(s)
        .replace("é", "e").replace("í", "i").replace("á", "a")
        .replace("ö", "o").replace("ü", "u").replace("ç", "c")
        .replace("č", "c").replace("ć", "c").replace("š", "s")
        .replace("ž", "z").replace("ñ", "n")
        .strip().lower()
    )


def metric_supported(series: pd.Series) -> bool:
    vals = num(series).dropna()

    if vals.empty:
        return False

    # Treat all-zero columns as unsupported placeholders.
    return not np.isclose(vals.abs().sum(), 0.0)


def canonicalize(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    player_col = first_col(df, ALIASES["player"])
    if player_col is None:
        raise RuntimeError(f"{source_label}: no recognizable player column.")

    out["Player"] = df[player_col]

    team_col = first_col(df, ALIASES["team"])
    out["Club"] = df[team_col] if team_col else pd.NA

    age_col = first_col(df, ALIASES["age"])
    out["Age"] = num(df[age_col]) if age_col else np.nan

    mins_col = first_col(df, ALIASES["minutes"])
    out["Minutes"] = num(df[mins_col]) if mins_col else np.nan

    value_col = first_col(df, ALIASES["value"])
    out["Value €M"] = num(df[value_col]) if value_col else np.nan

    if not out["Value €M"].dropna().empty:
        if out["Value €M"].dropna().median() > 1000:
            out["Value €M"] = out["Value €M"] / 1_000_000

    for metric, aliases in ALIASES.items():
        if metric in {"player", "team", "age", "minutes", "value"}:
            continue

        col = first_col(df, aliases)

        if col:
            out[metric] = num(df[col])
        else:
            out[metric] = np.nan

    out["Source"] = source_label
    return out


def load_historical_pool() -> pd.DataFrame:
    raw = pd.read_parquet(EUROPE_PARQUET)
    hist = canonicalize(raw, "Europe 2025/26")

    if POOL_CSV.exists():
        pool = pd.read_csv(POOL_CSV)
        pcol = first_col(pool, ALIASES["player"])

        if pcol:
            allowed = set(pool[pcol].astype(str).map(norm_name))
            hist = hist[
                hist["Player"].astype(str).map(norm_name).isin(allowed)
            ].copy()

    # collapse duplicate player-season rows if present
    hist["_player_key"] = hist["Player"].map(norm_name)

    agg = {
        "Player": "first",
        "Club": "first",
        "Age": "first",
        "Minutes": "sum",
        "Value €M": "first",
        "Source": "first",
    }

    for metric in INTENDED_WEIGHTS:
        agg[metric] = "median"

    hist = (
        hist.groupby("_player_key", as_index=False)
        .agg(agg)
    )

    mins = num(hist["Minutes"])
    hist = hist[mins.isna() | mins.ge(MIN_HISTORICAL_MINUTES)].copy()

    return hist


def load_current_newcastle() -> pd.DataFrame:
    current_path = next((p for p in CURRENT_CANDIDATES if p.exists()), None)

    if current_path is None:
        raise FileNotFoundError(
            "Could not find current Newcastle season metrics parquet."
        )

    raw = pd.read_parquet(current_path)
    current = canonicalize(raw, "Newcastle 2026/27")

    if current["Club"].notna().any():
        mask = (
            current["Club"]
            .astype(str)
            .str.contains("Newcastle", case=False, na=False)
        )
        if mask.any():
            current = current[mask].copy()

    known_midfielders = {
        norm_name(x)
        for x in [
            "Nico Gonzalez",
            "Nico González",
            "Aladji Bamba",
            "Joelinton",
            "Lewis Miley",
            "Jacob Ramsey",
            "Joe Willock",
            "Sean Steur",
        ]
    }

    # Position-history filter where available.
    if POSITION_HISTORY.exists():
        pos = pd.read_parquet(POSITION_HISTORY)
        pcol = first_col(pos, ALIASES["player"])

        if pcol:
            position_cols = [
                c for c in pos.columns
                if any(
                    token in c.lower()
                    for token in ["position", "role", "pos"]
                )
            ]

            if position_cols:
                pos_text = (
                    pos[position_cols]
                    .astype(str)
                    .agg(" ".join, axis=1)
                    .str.upper()
                )

                midfield_mask = pos_text.str.contains(
                    r"\b(DM|CM|AM|MID|MIDFIELDER)\b",
                    regex=True,
                    na=False,
                )

                mid_names = set(
                    pos.loc[midfield_mask, pcol]
                    .astype(str)
                    .map(norm_name)
                )

                if mid_names:
                    current = current[
                        current["Player"]
                        .astype(str)
                        .map(norm_name)
                        .isin(mid_names)
                    ].copy()

    # Guardrail to central-midfield group.
    keys = current["Player"].astype(str).map(norm_name)

    if keys.isin(known_midfielders).any():
        current = current[keys.isin(known_midfielders)].copy()

    mins = num(current["Minutes"])
    current = current[
        mins.isna() | mins.ge(MIN_CURRENT_MINUTES)
    ].copy()

    return current


def common_supported_metrics(
    historical: pd.DataFrame,
    current: pd.DataFrame,
) -> list[str]:
    """
    Only metrics with meaningful data in BOTH datasets are allowed into the
    direct Newcastle-vs-recruitment comparison.
    """
    supported = []

    for metric in INTENDED_WEIGHTS:
        hist_ok = (
            metric in historical.columns
            and metric_supported(historical[metric])
        )
        curr_ok = (
            metric in current.columns
            and metric_supported(current[metric])
        )

        if hist_ok and curr_ok:
            supported.append(metric)

    return supported


def score_combined(
    combined: pd.DataFrame,
    supported_metrics: list[str],
) -> pd.DataFrame:
    out = combined.copy()

    raw_available_weight = sum(
        INTENDED_WEIGHTS[m] for m in supported_metrics
    )

    if raw_available_weight <= 0:
        raise RuntimeError(
            "No common supported metrics between historical and current datasets."
        )

    # Renormalize common supported weights to 100% of the comparison score,
    # while keeping Coverage % equal to the original intended-weight share.
    normalized_weights = {
        m: INTENDED_WEIGHTS[m] / raw_available_weight
        for m in supported_metrics
    }

    for metric in supported_metrics:
        vals = num(out[metric])
        pct = vals.rank(pct=True, method="average")

        if metric == "dispossessed":
            pct = 1.0 - pct

        out[f"{metric}_score"] = 35.0 + 30.0 * pct

    weighted_sum = pd.Series(0.0, index=out.index)
    row_weight = pd.Series(0.0, index=out.index)

    for metric, weight in normalized_weights.items():
        valid = num(out[metric]).notna()

        weighted_sum.loc[valid] += (
            out.loc[valid, f"{metric}_score"] * weight
        )
        row_weight.loc[valid] += weight

    out["Jaissle Controller"] = (
        weighted_sum / row_weight.replace(0, np.nan)
    )

    # Honest model coverage relative to intended full role model.
    coverage_weight = pd.Series(0.0, index=out.index)

    for metric in supported_metrics:
        valid = num(out[metric]).notna()
        coverage_weight.loc[valid] += INTENDED_WEIGHTS[metric]

    out["Coverage %"] = coverage_weight * 100.0

    # Interpretable sub-components using only COMMON supported metrics.
    transition_metrics = [
        m for m in ["recoveries", "interceptions", "tackles"]
        if m in supported_metrics
    ]
    progression_metrics = [
        m for m in ["line_breaking", "successful_passes"]
        if m in supported_metrics
    ]
    security_metrics = [
        m for m in ["dispossessed", "pass_accuracy"]
        if m in supported_metrics
    ]
    involvement_metrics = [
        m for m in ["touches"]
        if m in supported_metrics
    ]

    def mean_component(metrics):
        cols = [f"{m}_score" for m in metrics if f"{m}_score" in out.columns]
        return out[cols].mean(axis=1) if cols else np.nan

    out["Transition Protection"] = mean_component(transition_metrics)
    out["Deep Progression"] = mean_component(progression_metrics)
    out["Possession Security"] = mean_component(security_metrics)
    out["Deep Involvement"] = mean_component(involvement_metrics)

    return out, normalized_weights


def main():
    print("=" * 124)
    print("JAISSLE DEEP TRANSITION CONTROLLER ANALYSIS")
    print("=" * 124)

    historical = load_historical_pool()
    current = load_current_newcastle()

    historical["NUFC Benchmark"] = ""
    current["NUFC Benchmark"] = "CURRENT"

    supported_metrics = common_supported_metrics(
        historical,
        current,
    )

    combined = pd.concat(
        [historical, current],
        ignore_index=True,
        sort=False,
    )

    combined, normalized_weights = score_combined(
        combined,
        supported_metrics,
    )

    print()
    print("COMMON COMPARISON METRICS")
    print("-" * 124)

    for metric, intended_weight in INTENDED_WEIGHTS.items():
        hist_ok = metric_supported(historical[metric]) if metric in historical else False
        curr_ok = metric_supported(current[metric]) if metric in current else False

        status = (
            "USED"
            if metric in supported_metrics
            else f"EXCLUDED (historical={hist_ok}, current={curr_ok})"
        )

        inverse = " inverse" if metric == "dispossessed" else ""

        print(
            f"{metric:<22} "
            f"intended={intended_weight:>5.0%}   "
            f"{status}{inverse}"
        )

    common_coverage = sum(
        INTENDED_WEIGHTS[m] for m in supported_metrics
    )

    print()
    print(
        f"COMMON MODEL COVERAGE: {common_coverage:.0%} "
        f"of intended role definition"
    )
    print("Score weights are renormalized only across these common metrics.")

    print()
    print("RENORMALIZED SCORING WEIGHTS")
    print("-" * 124)

    for metric, weight in normalized_weights.items():
        print(f"{metric:<22} {weight:>6.2%}")

    display_cols = [
        "Player",
        "Club",
        "Age",
        "Source",
        "NUFC Benchmark",
        "Jaissle Controller",
        "Transition Protection",
        "Deep Progression",
        "Possession Security",
        "Deep Involvement",
        "Coverage %",
        "recoveries",
        "interceptions",
        "tackles",
        "line_breaking",
        "successful_passes",
        "dispossessed",
        "pass_accuracy",
        "touches",
        "Minutes",
        "Value €M",
    ]

    display_cols = [c for c in display_cols if c in combined.columns]

    out = combined[display_cols].rename(
        columns={
            "recoveries": "Recoveries/90",
            "interceptions": "Interceptions/90",
            "tackles": "Tackles/90",
            "line_breaking": "Line-Breaking Passes/90",
            "successful_passes": "Successful Passes/90",
            "dispossessed": "Dispossessed/90",
            "pass_accuracy": "Pass %",
            "touches": "Touches/90",
        }
    )

    external = out[
        out["NUFC Benchmark"].ne("CURRENT")
    ].sort_values(
        "Jaissle Controller",
        ascending=False,
        na_position="last",
    )

    nufc = out[
        out["NUFC Benchmark"].eq("CURRENT")
    ].sort_values(
        "Jaissle Controller",
        ascending=False,
        na_position="last",
    )

    final = pd.concat(
        [external, nufc],
        ignore_index=True,
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_CSV, index=False)
    final.to_html(
        OUT_HTML,
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )

    print()
    print("TOP EXTERNAL DEEP-TRANSITION CONTROLLER TARGETS")
    print("-" * 124)
    print(external.head(30).to_string(index=False))

    print()
    print("=" * 124)
    print("CURRENT NEWCASTLE MIDFIELD COMPARISON")
    print("=" * 124)
    print(nufc.to_string(index=False))

    print()
    print(
        "Interpretation: Newcastle and recruitment targets are scored on "
        "the exact same metric set. Missing historical line-breaking/pass "
        "accuracy data can no longer advantage the current Newcastle group."
    )

    print()
    print("Saved:")
    print(" ", OUT_CSV)
    print(" ", OUT_HTML)


if __name__ == "__main__":
    main()
