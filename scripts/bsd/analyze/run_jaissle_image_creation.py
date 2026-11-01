#!/usr/bin/env python3
"""
run_jaissle_image_creation.py

Create a role-based visualization pack for the Jaissle recruitment pipeline.

Key upgrades in this version
----------------------------
1. Outputs to the repo-top ``visualizations/`` folder with subfolders per role.
2. Annotates more player names on market charts (top-ranked + useful outliers).
3. Adds a Newcastle benchmark side-panel to each role chart.
4. Tries to avoid overly linear / redundant axis pairs by switching to an
   alternate metric when the chosen x/y pair is almost perfectly correlated.
5. Keeps the squad coverage and role-gap summary views.

Usage
-----
python scripts/bsd/analyze/run_jaissle_image_creation.py
python scripts/bsd/analyze/run_jaissle_image_creation.py --top-n 40 --price-cap 70
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent)

    seen = set()
    for seed in candidates:
        cur = seed.parent if seed.is_file() else seed
        while True:
            if cur in seen:
                break
            seen.add(cur)
            if (cur / "data").exists() and (cur / "scripts").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    raise FileNotFoundError(
        "Could not determine project root. Run this inside SAP-Newcastle-Rebuild "
        "or place the script inside the repo."
    )


def normalize_key(text: object) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).split())


def choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {normalize_key(c): c for c in df.columns}
    for cand in candidates:
        hit = lookup.get(normalize_key(cand))
        if hit is not None:
            return hit
    return None


def choose_existing_file(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def numeric(s: object) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_text(v: object) -> str:
    return "NA" if pd.isna(v) else str(v)


def shortlist_root(project_root: Path) -> Path:
    return project_root / "data" / "processed" / "bsd" / "shortlist"


def visualizations_root(project_root: Path) -> Path:
    out = project_root / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    return out


def visualization_dir(project_root: Path, folder: str) -> Path:
    out = visualizations_root(project_root) / folder
    out.mkdir(parents=True, exist_ok=True)
    return out


def league_to_color(values: pd.Series) -> np.ndarray:
    mapping = {
        "premier_league": "tab:red",
        "serie_a": "tab:blue",
        "la_liga": "tab:orange",
        "bundesliga": "tab:green",
        "ligue_1": "tab:purple",
        "eredivisie": "tab:brown",
        "primeira_liga": "tab:pink",
        "liga_portugal": "tab:pink",
        "belgian_pro_league": "tab:olive",
        "jupiler_pro_league": "tab:olive",
        "turkish_super_lig": "tab:cyan",
        "swiss_super_league": "goldenrod",
        "scottish_premiership": "slategray",
    }
    return np.array([mapping.get(str(v), "tab:gray") for v in values])


def bubble_sizes(vals: pd.Series) -> np.ndarray:
    x = numeric(vals).fillna(0)
    if len(x) == 0:
        return np.array([])
    if x.max() <= 0:
        return np.full(len(x), 210.0)
    return 110 + 520 * np.sqrt(x.clip(lower=0) / x.max())


def role_order_key(role_name: str) -> tuple[int, str]:
    n = normalize_key(role_name)
    if "goalkeeper" in n or n == "gk":
        return (0, n)
    if "center back" in n or "centre back" in n or n == "cb":
        return (1, n)
    if "wingback" in n or "fullback" in n or "left back" in n or "right back" in n:
        return (2, n)
    if "holding" in n or "pivot" in n or "cdm" in n:
        return (3, n)
    if "workhorse" in n or "central midfielder" in n or n == "cm":
        return (4, n)
    if "creative 10" in n or "cam" in n or "attacking midfielder" in n:
        return (5, n)
    if "winger" in n or "inside creator" in n or "wide" in n:
        return (6, n)
    if "striker" in n or "forward" in n:
        return (7, n)
    return (8, n)


# -----------------------------------------------------------------------------
# Role specs
# -----------------------------------------------------------------------------

ROLE_SPECS = [
    {
        "slug": "cam",
        "folder": "cam",
        "label": "Creative 10 / CAM",
        "benchmark_labels": ["Creative 10 / CAM", "Creative 10", "CAM"],
        "patterns": ["creative_10_jaissle_detailed.csv", "creative_10.csv", "*cam*.csv"],
        "score_candidates": ["CAM Fit", "Final CAM Score", "jaissle_cam_score", "role_fit", "Role Fit", "Role Score"],
        "x_candidates": ["Value €M", "market_value_eur_m", "Age"],
        "y_candidates": ["CAM Fit", "role_fit", "Role Fit", "Role Score"],
        "alt_y_candidates": ["Coverage %"],
        "x_label": "Market value (€M)",
        "y_label": "CAM fit",
        "caption": "Left panel: external market. X = market value, Y = CAM fit. Bubble size = value. Right panel: Newcastle role benchmarks.",
    },
    {
        "slug": "striker",
        "folder": "striker",
        "label": "Focal Striker",
        "benchmark_labels": ["Focal Striker", "Striker"],
        "patterns": ["focal_striker_jaissle_detailed.csv", "focal_striker.csv", "*striker*.csv"],
        "score_candidates": ["Jaissle ST", "jaissle_striker_score", "Combined Score", "Focal Fit", "role_fit", "Role Fit", "Role Score"],
        "x_candidates": ["Focal Fit", "role_fit", "Role Fit", "Role Score"],
        "y_candidates": ["Pressing", "pressing_score", "Jaissle ST", "Combined Score"],
        "alt_y_candidates": ["Box Threat", "Focal Presence", "Link Play"],
        "x_label": "Focal-striker fit",
        "y_label": "Pressing / overall Jaissle score",
        "caption": "Left panel: focal fit vs pressing / total striker score. Bubble size = value. Right panel: Newcastle striker benchmarks.",
    },
    {
        "slug": "center_back",
        "folder": "center_back",
        "label": "High-Line Center Back",
        "benchmark_labels": ["High-Line Center Back", "Center Back", "Centre Back"],
        "patterns": ["center_back_jaissle_detailed.csv", "center_back.csv", "*center*back*.csv", "*centre*back*.csv", "*cb*.csv"],
        "score_candidates": ["Jaissle CB", "jaissle_cb_score", "CB Fit", "role_fit", "Role Fit", "Role Score"],
        "x_candidates": ["Build-Up", "Progression", "Passing", "CB Fit", "role_fit", "Role Fit"],
        "y_candidates": ["Front-Foot", "Physical", "Defending", "Duels", "Aerial", "Jaissle CB"],
        "alt_y_candidates": ["Recovery Pace", "role_fit", "Role Fit"],
        "x_label": "Build-up / role fit",
        "y_label": "Front-foot / physical defending",
        "caption": "Left panel: high-line CB profile. Bubble size = value. Right panel: Newcastle center-back benchmarks.",
    },
    {
        "slug": "holding_controller",
        "folder": "central_midfield",
        "label": "Holding Controller / No. 6",
        "benchmark_labels": ["Holding Controller / No. 6", "Holding Pivot", "Controller"],
        "patterns": ["holding_pivot.csv", "*holding*pivot*.csv", "*controller*.csv"],
        "score_candidates": ["role_fit", "Role Fit", "Role Score", "Fit"],
        "x_candidates": ["Value €M", "market_value_eur_m", "Age"],
        "y_candidates": ["role_fit", "Role Fit", "Role Score", "Fit"],
        "alt_y_candidates": ["Coverage %"],
        "x_label": "Market value (€M)",
        "y_label": "Holding / controller fit",
        "caption": "External controller pool. This is a role-fit model rather than a dedicated Jaissle analyzer; useful for identifying a possible experienced mentor / No. 6.",
    },
    {
        "slug": "workhorse_cm",
        "folder": "central_midfield",
        "label": "Workhorse Central Midfielder",
        "benchmark_labels": ["Workhorse Central Midfielder", "Central Midfielder", "CM", "Workhorse CM"],
        "patterns": ["workhorse_cm_jaissle_detailed.csv", "advanced_pivot.csv", "*workhorse*cm*.csv", "*workhorse*.csv"],
        "score_candidates": ["Jaissle CM", "jaissle_workhorse_score", "CM Fit", "role_fit", "Role Fit", "Role Score"],
        "x_candidates": ["Workhorse", "Ball Winning", "Progression", "CM Fit", "role_fit", "Role Fit"],
        "y_candidates": ["Security/Link", "Attack Support", "Pressing", "Physical", "Jaissle CM"],
        "alt_y_candidates": ["Recoveries/90", "Duels Won/90", "role_fit"],
        "x_label": "Workhorse / ball-winning score",
        "y_label": "Security / link support",
        "caption": "Left panel: workhorse / security profile. Bubble size = value. Right panel: Newcastle midfield benchmarks.",
    },
    {
        "slug": "left_wb",
        "folder": "fullback",
        "label": "Left Wingback / Fullback",
        "benchmark_labels": ["Left Wingback / Fullback", "Left Wingback", "Left Back", "Fullback"],
        "patterns": ["left_wingback_recruitment_progressive.csv", "left_wingback_recruitment_creation.csv", "wingback.csv", "*left*wingback*.csv", "*leftback*.csv"],
        "score_candidates": ["Jaissle FB Score", "jaissle_fb_score", "WB Fit", "role_fit", "Role Fit", "Role Score"],
        "x_candidates": ["Creation", "Prog/Create", "Attacking Creation", "WB Fit", "role_fit"],
        "y_candidates": ["Physical", "Defending", "Jaissle FB Score", "jaissle_fb_score"],
        "alt_y_candidates": ["WB Fit", "role_fit", "Role Fit", "xA/90", "Key Passes/90"],
        "x_label": "Creation / progression",
        "y_label": "Physical / defensive activity",
        "caption": "Left panel: creation/progression vs physical base. Bubble size = value. Right panel: Newcastle fullback benchmarks.",
    },
    {
        "slug": "wide_creator",
        "folder": "winger/wide_creator",
        "label": "Wide Creator",
        "benchmark_labels": ["Winger", "Wide Creator", "Vertical Winger"],
        "patterns": ["winger_jaissle_extended.csv", "winger_jaissle_archetypes.csv", "*winger*.csv"],
        "score_candidates": ["wide_creator_score", "Creator", "Best Score"],
        "x_candidates": ["xA/90", "Creator", "Value €M"],
        "y_candidates": ["Creator", "Best Score"],
        "alt_y_candidates": ["Hybrid", "Vertical/Box"],
        "x_label": "Chance creation",
        "y_label": "Wide-creator score",
        "caption": "Wide-creator archetype: progression, final ball and creation.",
    },
    {
        "slug": "vertical_box_winger",
        "folder": "winger/vertical_box",
        "label": "Vertical / Box Winger",
        "benchmark_labels": ["Winger", "Vertical / Box Winger", "Vertical Winger"],
        "patterns": ["winger_jaissle_extended.csv", "winger_jaissle_archetypes.csv", "*winger*.csv"],
        "score_candidates": ["vertical_box_score", "Vertical/Box", "Best Score"],
        "x_candidates": ["xG/90", "Shots/90", "Vertical/Box"],
        "y_candidates": ["Vertical/Box", "Best Score"],
        "alt_y_candidates": ["Hybrid", "Creator"],
        "x_label": "Direct / box threat",
        "y_label": "Vertical / box score",
        "caption": "Vertical/box archetype: direct running, scoring threat and box occupation.",
    },
    {
        "slug": "overall_winger",
        "folder": "winger/overall",
        "label": "Overall Winger",
        "benchmark_labels": ["Winger", "Hybrid Winger"],
        "patterns": ["winger_jaissle_extended.csv", "winger_jaissle_archetypes.csv", "*winger*.csv"],
        "score_candidates": ["overall_winger_score", "Best Score", "hybrid_score", "Hybrid"],
        "x_candidates": ["Age", "age", "Creator"],
        "y_candidates": ["overall_winger_score", "Best Score", "Hybrid"],
        "alt_y_candidates": ["Best Score"],
        "x_label": "Age",
        "y_label": "Overall winger score",
        "caption": "Combined recruitment view: overall winger score blends primary archetype, hybrid versatility, off-ball work and secondary archetype. Bubble size = value.",
    }
]

COMMON_FIELD_CANDIDATES = {
    "player": ["Player", "player_name", "name"],
    "club": ["Club", "club_name", "current_club_name", "team_name"],
    "league": ["League", "league_slug"],
    "age": ["Age", "age"],
    "value": ["Value €M", "market_value_eur_m", "market_value_eur", "Value", "Transfer Value"],
}


# -----------------------------------------------------------------------------
# Loading role data
# -----------------------------------------------------------------------------

def load_role_dataframe(project_root: Path, spec: dict) -> tuple[pd.DataFrame, Path, str, str]:
    root = shortlist_root(project_root)
    path = choose_existing_file(root, spec["patterns"])
    if path is None:
        path = choose_existing_file(project_root / "data" / "processed" / "bsd", spec["patterns"])
    if path is None:
        raise FileNotFoundError(f"Could not find file for role: {spec['label']}")

    df = read_table(path).copy()

    for out_name, aliases in COMMON_FIELD_CANDIDATES.items():
        col = choose_column(df, aliases)
        if col is not None and out_name not in df.columns:
            df[out_name] = df[col]

    score_col = choose_column(df, spec["score_candidates"])
    xcol = choose_column(df, spec["x_candidates"])
    ycol = choose_column(df, spec["y_candidates"])

    if score_col is None:
        raise KeyError(f"No score column found for {spec['label']} in {path.name}")

    if xcol is None:
        xcol = score_col
    if ycol is None:
        ycol = score_col

    df["_score"] = numeric(df[score_col])
    df["_x"] = numeric(df[xcol])
    df["_y"] = numeric(df[ycol])
    df["_player"] = df.get("player", pd.Series("Unknown", index=df.index))
    df["_club"] = df.get("club", pd.Series(np.nan, index=df.index))
    df["_league"] = df.get("league", pd.Series(np.nan, index=df.index))
    df["_value"] = numeric(df["value"]) if "value" in df.columns else np.nan
    df["_age"] = numeric(df["age"]) if "age" in df.columns else np.nan
    return df, path, xcol, ycol



def zscore_series(series: pd.Series) -> pd.Series:
    x = numeric(series)
    if x.notna().sum() < 3:
        return pd.Series(np.nan, index=x.index)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / sd


def scaled_composite(df: pd.DataFrame, metric_candidates: list[tuple[list[str], float]]) -> pd.Series:
    """Build a 50-centered composite from available non-constant source metrics."""
    parts = []
    weights = []
    for aliases, weight in metric_candidates:
        col = choose_column(df, aliases)
        if col is None:
            continue
        z = zscore_series(df[col])
        if z.notna().sum() < 3:
            continue
        parts.append(z)
        weights.append(float(weight))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    total = sum(weights)
    out = sum(part * w for part, w in zip(parts, weights)) / total
    return 50.0 + 10.0 * out


def add_independent_plot_metrics(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, str, str]:
    """
    Override highly coupled model axes for the two roles where the previous
    chart was effectively a 1:1 transformation.
    """
    out = df.copy()
    slug = spec.get("slug")

    if slug == "left_wb":
        # Creation remains the attacking axis, while the Y-axis is rebuilt
        # only from defensive/physical event metrics. If source columns are
        # unavailable, fall back to the analyzer's Physical score rather than
        # dropping the entire role from the visualization.
        creation_col = choose_column(out, ["Creation", "Attacking Creation", "WB Fit"])
        if creation_col is not None:
            out["_x"] = numeric(out[creation_col])

        defensive = scaled_composite(
            out,
            [
                (["Duels Won/90", "duels_won_per90"], 0.30),
                (["Recoveries/90", "recoveries_per90"], 0.25),
                (["Interceptions/90", "interceptions_per90"], 0.20),
                (["Tackles Won/90", "tackles_won_per90"], 0.25),
            ],
        )
        if defensive.notna().sum() >= 6:
            out["_y"] = defensive
            ylabel = "Defensive / physical composite"
        else:
            fallback = choose_column(out, ["Physical", "WB Fit", "Jaissle FB Score"])
            if fallback is not None:
                out["_y"] = numeric(out[fallback])
            ylabel = fallback or "Physical"
        return out, "Creation / progression", ylabel

    if slug in {"winger", "wide_creator", "vertical_box_winger"}:
        # Separate chance creation from direct / box threat using raw metrics.
        # If those metrics are too sparse in an export, use the archetype
        # columns directly instead of producing an empty chart.
        creation = scaled_composite(
            out,
            [
                (["xA/90", "xa_per90"], 0.70),
                (["Key Passes/90", "key_passes_per90"], 0.30),
            ],
        )
        direct = scaled_composite(
            out,
            [
                (["xG/90", "xg_per90"], 0.50),
                (["Shots/90", "shots_per90"], 0.35),
                (["Box Touches/90", "box_touches_per90"], 0.15),
            ],
        )

        if creation.notna().sum() >= 6:
            out["_x"] = creation
            xlabel = "Chance-creation composite"
        else:
            fallback_x = choose_column(out, ["Creator", "Hybrid", "Best Score"])
            if fallback_x is not None:
                out["_x"] = numeric(out[fallback_x])
            xlabel = fallback_x or "Creator"

        if direct.notna().sum() >= 6:
            out["_y"] = direct
            ylabel = "Direct / box-threat composite"
        else:
            fallback_y = choose_column(out, ["Vertical/Box", "Hybrid", "Best Score"])
            if fallback_y is not None:
                out["_y"] = numeric(out[fallback_y])
            ylabel = fallback_y or "Vertical / Box"

        return out, xlabel, ylabel

    return out, spec.get("x_label", "X"), spec.get("y_label", "Y")


def source_newcastle_mask(df: pd.DataFrame) -> pd.Series:
    """Identify Newcastle benchmark/reference rows in a role source table."""
    mask = pd.Series(False, index=df.index)
    club_col = choose_column(df, ["Club", "club", "club_name"])
    if club_col is not None:
        mask |= df[club_col].astype(str).str.contains("Newcastle", case=False, na=False)
    for marker in ["NUFC", "NUFC Benchmark", "Newcastle Benchmark"]:
        col = choose_column(df, [marker])
        if col is not None:
            vals = df[col]
            valid = vals.notna() & vals.astype(str).str.strip().ne("") & ~vals.astype(str).str.lower().isin({"nan", "none", "<na>"})
            mask |= valid
    return mask


def benchmark_name_candidates(full_name: str) -> set[str]:
    """Generate loose short-name variants for matching benchmark names to source rows."""
    if not isinstance(full_name, str) or not full_name.strip():
        return set()
    bits = full_name.strip().split()
    variants = {normalize_key(full_name)}
    if len(bits) >= 2:
        variants.add(normalize_key(f"{bits[0][0]}. {bits[-1]}"))
        variants.add(normalize_key(f"{bits[0][0]} {bits[-1]}"))
        variants.add(normalize_key(bits[-1]))
    return variants


def force_include_newcastle_rows(source_df: pd.DataFrame, plotted_df: pd.DataFrame,
                                  bench_rows: pd.DataFrame) -> pd.DataFrame:
    """Append Newcastle role references even when they sit outside the market top-N cut."""
    src = source_df.copy()
    mask = source_newcastle_mask(src)

    player_col = choose_column(src, ["Player", "player_name", "name"])
    if player_col is not None and bench_rows is not None and not bench_rows.empty:
        bench_variants = set()
        for name in bench_rows["_player"].dropna().astype(str):
            bench_variants.update(benchmark_name_candidates(name))
        src_names = src[player_col].astype(str).map(normalize_key)
        mask |= src_names.isin(bench_variants)

    forced = src[mask].copy()
    if forced.empty:
        return plotted_df

    # Ensure standardized helper columns exist on appended rows.
    for col in ["_score", "_x", "_y", "_player", "_club", "_league", "_value", "_age"]:
        if col not in forced.columns and col in source_df.columns:
            forced[col] = source_df.loc[forced.index, col]

    out = pd.concat([plotted_df, forced], axis=0)
    out = out[~out.index.duplicated(keep="first")].copy()
    out["_is_newcastle"] = source_newcastle_mask(out)
    if player_col is not None and bench_rows is not None and not bench_rows.empty:
        bench_variants = set()
        for name in bench_rows["_player"].dropna().astype(str):
            bench_variants.update(benchmark_name_candidates(name))
        out["_is_newcastle"] |= out[player_col].astype(str).map(normalize_key).isin(bench_variants)
    return out


# -----------------------------------------------------------------------------
# Newcastle benchmark loading
# -----------------------------------------------------------------------------

def load_newcastle_benchmarks(project_root: Path) -> pd.DataFrame:
    search_dirs = [shortlist_root(project_root), project_root / "data" / "processed" / "bsd"]
    patterns = [
        "newcastle_jaissle_role_benchmarks.csv",
        "newcastle_jaissle_role_benchmarks.parquet",
        "*newcastle*benchmark*.csv",
        "*newcastle*benchmark*.parquet",
        "*retain*review*benchmark*.csv",
        "*retain*review*.csv",
        "*newcastle*historical*.csv",
        "*newcastle*historical*.parquet",
    ]

    for search_dir in search_dirs:
        for pattern in patterns:
            for path in sorted(search_dir.rglob(pattern)):
                try:
                    df = read_table(path)
                except Exception:
                    continue
                player_col = choose_column(df, ["Player", "player_name", "Player Name"])
                role_col = choose_column(df, ["Role", "role", "role_name", "Role Name", "role_key"])
                score_col = choose_column(
                    df,
                    [
                        "jaissle_role_score",
                        "Role Score",
                        "Score",
                        "jaissle_score",
                        "Role Fit",
                        "role_fit",
                    ],
                )
                if not (player_col and role_col and score_col):
                    continue
                out = df.copy()
                out["_player"] = out[player_col]
                out["_role"] = out[role_col]
                out["_score"] = numeric(out[score_col])
                fit_col = choose_column(out, ["role_fit", "Role Fit", "Fit"])
                out["_fit"] = numeric(out[fit_col]) if fit_col else np.nan
                role_key_col = choose_column(out, ["role_key", "Role Key"])
                out["_role_key"] = out[role_key_col].astype(str) if role_key_col else out["_role"].map(normalize_key)
                archetype_col = choose_column(out, ["Archetype", "Best Archetype", "best_archetype"])
                out["_archetype"] = out[archetype_col] if archetype_col else np.nan
                value_col = choose_column(out, ["Value €M", "market_value_eur_m", "market_value_eur"])
                decision_col = choose_column(out, ["Decision Flag", "decision_flag"])
                age_col = choose_column(out, ["Age", "age"])
                out["_value"] = numeric(out[value_col]) if value_col else np.nan
                out["_decision"] = out[decision_col] if decision_col else np.nan
                out["_age"] = numeric(out[age_col]) if age_col else np.nan
                out["_norm_role"] = out["_role"].map(normalize_key)
                return out

    raise FileNotFoundError(
        "Could not find a Newcastle benchmark table. Expected a CSV/parquet with Player / Role / Role Score columns."
    )


def benchmark_rows_for_spec(bench: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if bench.empty:
        return bench.copy()
    targets = {normalize_key(x) for x in spec.get("benchmark_labels", [spec["label"]])}
    rows = bench[bench["_norm_role"].isin(targets)].copy()
    return rows.sort_values("_score", ascending=False)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def maybe_reduce_correlation(df: pd.DataFrame, spec: dict, original_df: pd.DataFrame,
                             xcol_used: str, ycol_used: str) -> tuple[pd.DataFrame, str]:
    """If x/y are nearly perfectly correlated, try an alternate Y metric."""
    if df[["_x", "_y"]].dropna().shape[0] < 6:
        return df, ycol_used
    corr = df[["_x", "_y"]].corr().iloc[0, 1]
    if not np.isfinite(corr) or abs(corr) < 0.97:
        return df, ycol_used

    for cand in spec.get("alt_y_candidates", []):
        actual = choose_column(original_df, [cand])
        if actual is None:
            continue
        alt = numeric(original_df[actual])
        tmp = df.copy()
        tmp["_y"] = alt.loc[tmp.index]
        valid = tmp[["_x", "_y"]].dropna()
        if len(valid) < 6:
            continue
        alt_corr = valid.corr().iloc[0, 1]
        if np.isfinite(alt_corr) and abs(alt_corr) < abs(corr) - 0.05:
            return tmp, actual
    return df, ycol_used


def build_annotation_mask(df: pd.DataFrame, annotate_n: int) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    mask.loc[df.head(min(annotate_n, len(df))).index] = True

    # Also annotate useful outliers/value plays.
    if "_value" in df.columns:
        value_eff = df["_score"] / df["_value"].replace(0, np.nan)
        extra = value_eff.sort_values(ascending=False).head(5).index
        mask.loc[extra] = True

    if len(df) >= 8:
        mask.loc[df["_x"].nlargest(3).index] = True
        mask.loc[df["_y"].nlargest(3).index] = True
        mask.loc[df["_score"].nlargest(6).index] = True
    return mask.fillna(False)


def annotate_points(ax, df: pd.DataFrame, label_mask: pd.Series):
    subset = df[label_mask].copy()
    if subset.empty:
        return
    for i, (_, row) in enumerate(subset.iterrows()):
        x = row["_x"]
        y = row["_y"]
        name = safe_text(row["_player"])
        dx = 0.18 if i % 2 == 0 else -0.18
        dy = 0.24 if i % 3 == 0 else (-0.20 if i % 3 == 1 else 0.05)
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            name,
            (x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            fontsize=8,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.45),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
        )


def draw_league_legend(ax, df: pd.DataFrame):
    if "_league" not in df.columns:
        return
    leagues = [v for v in sorted(df["_league"].dropna().astype(str).unique())]
    if not leagues:
        return
    handles = []
    for lg in leagues:
        color = league_to_color(pd.Series([lg]))[0]
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=lg,
                       markerfacecolor=color, markeredgecolor="black", markersize=8)
        )
    ax.legend(handles=handles, title="League", fontsize=8, loc="best")


def draw_newcastle_panel(ax, bench_rows: pd.DataFrame, threshold: float, spec: dict):
    ax.clear()
    ax.set_title("Newcastle benchmarks", fontsize=12, pad=10)
    if bench_rows.empty:
        ax.text(0.5, 0.5, "No benchmark rows found",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return

    plot = bench_rows.copy()
    plot = plot.dropna(subset=["_player", "_score"]).copy()
    if plot.empty:
        ax.text(0.5, 0.5, "No usable benchmark rows",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return

    plot = plot.sort_values("_score", ascending=True)
    y = np.arange(len(plot))
    colors = ["tab:green" if s >= threshold else "tab:orange" for s in plot["_score"]]
    edgecolors = ["black"] * len(plot)
    lw = [1.8 if s < threshold else 1.0 for s in plot["_score"]]
    ax.barh(y, plot["_score"], color=colors, edgecolor=edgecolors, linewidth=lw, alpha=0.78)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["_player"], fontsize=8)
    ax.set_xlabel("Role score")
    ax.grid(True, axis="x", alpha=0.22)

    max_x = max(float(plot["_score"].max()), threshold) + 6
    ax.set_xlim(0, max(70, max_x))

    for yi, (_, row) in enumerate(plot.iterrows()):
        score = float(row["_score"])
        extra = []
        if pd.notna(row.get("_value", np.nan)):
            extra.append(f"€{float(row['_value']):.0f}M")
        if isinstance(row.get("_decision", None), str):
            dec = row["_decision"].strip()
            if dec:
                extra.append(dec.replace("REPLACE / SELL REVIEW", "review").replace("KEEP / ROLE FIT", "keep"))
        suffix = f" ({', '.join(extra[:2])})" if extra else ""
        ax.text(score + 0.4, yi, f"{score:.1f}{suffix}", va="center", fontsize=8)

    ax.text(
        0.01,
        -0.15,
        f"Threshold = {threshold:.1f}. Orange bars are below threshold and should be reviewed more closely.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f5f5f5", ec="#cccccc"),
    )


# -----------------------------------------------------------------------------
# Main role chart creator
# -----------------------------------------------------------------------------

def create_role_chart(project_root: Path, spec: dict, top_n: int = 40, annotate_n: int = 18,
                      price_cap: Optional[float] = None, threshold: float = 52.5,
                      benchmarks: Optional[pd.DataFrame] = None) -> Optional[Path]:
    try:
        original_df, src, xcol_used, ycol_used = load_role_dataframe(project_root, spec)
    except Exception as exc:
        print(f"[skip] {spec['label']}: {exc}")
        return None

    # Build role-specific independent plotting axes before market selection.
    original_df, x_label_override, y_label_override = add_independent_plot_metrics(original_df, spec)

    bench_rows = benchmark_rows_for_spec(benchmarks, spec) if benchmarks is not None else pd.DataFrame()

    eligible = original_df.dropna(subset=["_score", "_x", "_y"]).copy()
    if price_cap is not None:
        market_mask = eligible["_value"].isna() | (eligible["_value"] <= price_cap)
    else:
        market_mask = pd.Series(True, index=eligible.index)

    # Newcastle reference rows should not be removed by the external price cap.
    nufc_mask = source_newcastle_mask(eligible)
    external = eligible[market_mask & ~nufc_mask].copy()
    external = external.sort_values(["_score", "_x", "_y"], ascending=False).head(top_n).copy()

    # Force benchmark/reference rows into the plotted data, even if they are
    # weak enough to fall outside the top 40 external options.
    plotted = force_include_newcastle_rows(eligible, external, bench_rows)
    if plotted.empty:
        print(f"[skip] {spec['label']}: no rows after filtering")
        return None

    if "_is_newcastle" not in plotted.columns:
        plotted["_is_newcastle"] = source_newcastle_mask(plotted)

    plotted["_bubble"] = 220.0 if spec.get("slug") == "cam" else bubble_sizes(plotted["_value"])
    plotted["_color"] = league_to_color(plotted["_league"])

    # Large left chart; right benchmark panel stays about the same physical size.
    if not bench_rows.empty:
        fig, (ax, bx) = plt.subplots(
            1, 2,
            figsize=(20.5, 8.8),
            gridspec_kw={"width_ratios": [5.0, 1.55], "wspace": 0.18},
        )
    else:
        fig, ax = plt.subplots(figsize=(16.5, 8.6))
        bx = None

    market = plotted[~plotted["_is_newcastle"]].copy()
    nufc = plotted[plotted["_is_newcastle"]].copy()

    if not market.empty:
        ax.scatter(
            market["_x"], market["_y"],
            s=market["_bubble"], c=market["_color"],
            alpha=0.70, edgecolors="black", linewidths=0.5,
            zorder=2,
        )

    if not nufc.empty:
        ax.scatter(
            nufc["_x"], nufc["_y"],
            s=np.maximum(nufc["_bubble"], 420),
            marker="*", c="gold", edgecolors="black", linewidths=1.3,
            label="Newcastle benchmark", zorder=6,
        )

    # Threshold only makes sense on score-scaled Y axes.
    if spec.get("slug") not in {"left_wb", "winger"}:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, alpha=0.65)

    label_mask = build_annotation_mask(market, annotate_n=annotate_n) if not market.empty else pd.Series(False, index=market.index)
    annotate_points(ax, market, label_mask)

    # Always label Newcastle references.
    if not nufc.empty:
        for i, (_, row) in enumerate(nufc.iterrows()):
            ax.annotate(
                safe_text(row["_player"]),
                (row["_x"], row["_y"]),
                xytext=(7, 7 + (i % 3) * 5), textcoords="offset points",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="#fff5bf", ec="black", alpha=0.9),
                arrowprops=dict(arrowstyle="-", lw=0.55, alpha=0.7),
                zorder=7,
            )

    # Median guides from market players only.
    if not market.empty:
        x_med = float(market["_x"].median())
        y_med = float(market["_y"].median())
        if np.isfinite(x_med):
            ax.axvline(x_med, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        if np.isfinite(y_med):
            ax.axhline(y_med, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_title(f"{spec['label']} — Top {min(top_n, len(external))} External Options + Newcastle", fontsize=15, pad=14)
    ax.set_xlabel(x_label_override)
    ax.set_ylabel(y_label_override)
    ax.grid(True, alpha=0.23)
    draw_league_legend(ax, market)

    # Add Newcastle marker to the legend without replacing the league legend.
    handles, labels = ax.get_legend_handles_labels()
    if not nufc.empty and "Newcastle benchmark" not in labels:
        handles.append(plt.Line2D([0], [0], marker="*", color="w", label="Newcastle benchmark",
                                  markerfacecolor="gold", markeredgecolor="black", markersize=14))
        labels.append("Newcastle benchmark")
    if handles:
        ax.legend(handles=handles, labels=labels, title="League / benchmark", fontsize=8, loc="best")

    corr = np.nan
    if market[["_x", "_y"]].dropna().shape[0] >= 3:
        corr = market[["_x", "_y"]].corr().iloc[0, 1]
    corr_text = f"x/y corr = {corr:.2f}" if np.isfinite(corr) else "x/y corr = NA"

    value_outliers = pd.DataFrame()
    if not market.empty and market["_value"].notna().any():
        median_value = market["_value"].median()
        q80 = market["_score"].quantile(0.80)
        value_outliers = market[(market["_score"] >= q80) & (market["_value"] <= median_value)]
    outlier_note = ""
    if not value_outliers.empty:
        names = ", ".join(value_outliers["_player"].head(4).astype(str).tolist())
        outlier_note = f" Value outliers: {names}."

    caption = (
        f"Source: {src.name}\n"
        f"{spec['caption']} {corr_text}.{outlier_note} Newcastle reference players are starred and are forced onto the plot even if outside the top-{top_n} external cut."
    )
    ax.text(
        0.01, -0.17, caption,
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc"),
    )

    if bx is not None:
        draw_newcastle_panel(bx, bench_rows, threshold=threshold, spec=spec)

    outpath = visualization_dir(project_root, spec.get("folder", spec["slug"])) / f"{spec['slug']}_top_targets_chart.png"
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")
    return outpath


# -----------------------------------------------------------------------------
# Squad summary visuals
# -----------------------------------------------------------------------------

# Fixed tactical role map for squad-level visuals.
# The goal is a depth-chart view rather than a raw dump of model role strings.
SQUAD_ROLE_COLUMNS = [
    "Sweeper / Build-Up GK",
    "High-Line CB",
    "Left WB / FB",
    "Right WB / FB",
    "Holding Controller",
    "Advanced / Workhorse 8",
    "Creative 10",
    "Wide Creator",
    "Vertical / Box Winger",
    "Hybrid Winger",
    "Focal Striker",
]

GK_MANUAL_STATUS = {
    "Lukáš Horníček": "CONFIDENT",
    "Nick Pope": "DEPTH",
    "Jaouen": "DEV",
    "Mark Gillespie": "DEPTH",
}

WING_ARCHETYPE_BY_PLAYER = {
    "Bazoumana Toure": "Wide Creator",
    "Bazoumana Touré": "Wide Creator",
    "Anthony Elanga": "Wide Creator",
    "Matias Fernandez-Pardo": "Vertical / Box Winger",
    "Harvey Barnes": "Vertical / Box Winger",
    "Jacob Murphy": "Vertical / Box Winger",
}

LEFT_WB_PLAYERS = {"Lewis Hall"}
RIGHT_WB_PLAYERS = {"Amar Dedić", "Amar Dedic", "Tino Livramento"}


def canonical_benchmark_role(row: pd.Series) -> Optional[str]:
    player = str(row.get("_player", ""))
    role = normalize_key(row.get("_role", ""))
    role_key = normalize_key(row.get("_role_key", ""))

    if "center back" in role or role_key == "center back":
        return "High-Line CB"

    if "wingback" in role or "full back" in role or "vertical wingback" in role_key:
        if player in LEFT_WB_PLAYERS:
            return "Left WB / FB"
        if player in RIGHT_WB_PLAYERS:
            return "Right WB / FB"
        return "Left WB / FB"

    if "advanced pivot" in role or "workhorse" in role or "advanced pivot" in role_key:
        return "Advanced / Workhorse 8"

    if "creative 10" in role or "creative cam" in role_key:
        return "Creative 10"

    if "focal striker" in role or "focal striker" in role_key:
        return "Focal Striker"

    if "winger" in role or "vertical winger" in role_key:
        archetype = row.get("_archetype", np.nan)
        if isinstance(archetype, str) and archetype.strip():
            a = normalize_key(archetype)
            if "wide creator" in a:
                return "Wide Creator"
            if "vertical" in a or "box winger" in a:
                return "Vertical / Box Winger"
            if "hybrid" in a:
                return "Hybrid Winger"
        return WING_ARCHETYPE_BY_PLAYER.get(player, "Vertical / Box Winger")

    if "holding" in role or "controller" in role or "holding pivot" in role_key:
        return "Holding Controller"

    return None


def build_squad_role_matrix(bench: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []

    for _, row in bench.iterrows():
        role = canonical_benchmark_role(row)
        if role is None:
            continue
        rows.append({
            "Player": row["_player"],
            "Role": role,
            "Score": row["_score"],
        })

    long_df = pd.DataFrame(rows)

    # Ensure all manual GK rows appear even though the current pipeline does
    # not model goalkeeper metrics.
    manual_cells = {}
    for player, status in GK_MANUAL_STATUS.items():
        manual_cells[(player, "Sweeper / Build-Up GK")] = status

    # Controller is strategically important but currently not benchmarked as a
    # dedicated Newcastle role. Mark Nico as the current reference without
    # inventing a numerical score.
    manual_cells[("Nico Gonzalez", "Holding Controller")] = "CURRENT REF"

    players = list(GK_MANUAL_STATUS.keys())
    if not long_df.empty:
        model_players = (
            long_df.groupby("Player")["Score"]
            .max()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        players += [p for p in model_players if p not in players]

    matrix = pd.DataFrame(np.nan, index=players, columns=SQUAD_ROLE_COLUMNS)

    if not long_df.empty:
        best = (
            long_df.sort_values("Score", ascending=False)
            .drop_duplicates(["Player", "Role"])
        )
        for _, row in best.iterrows():
            if row["Player"] in matrix.index and row["Role"] in matrix.columns:
                matrix.loc[row["Player"], row["Role"]] = row["Score"]

    return matrix, manual_cells


def create_newcastle_coverage_heatmap(project_root: Path, bench: pd.DataFrame) -> Optional[Path]:
    matrix, manual_cells = build_squad_role_matrix(bench)

    if matrix.empty:
        print("[skip] Newcastle coverage heatmap: no usable rows")
        return None

    arr = matrix.to_numpy(dtype=float)
    fig_h = max(9.5, 0.42 * len(matrix.index) + 2.5)
    fig_w = max(18.5, 1.55 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=40,
        vmax=70,
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(
        "Newcastle Squad — Tactical Role Coverage",
        fontsize=16,
        pad=14,
    )

    # Numeric benchmark cells.
    for i, player in enumerate(matrix.index):
        for j, role in enumerate(matrix.columns):
            val = matrix.iloc[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=7.5,
                    color="white" if val >= 58 else "black",
                    fontweight="bold" if val >= 58 else "normal",
                )

    # Manual / non-modeled cells get a neutral box instead of being treated as 0.
    from matplotlib.patches import Rectangle
    for (player, role), label in manual_cells.items():
        if player not in matrix.index or role not in matrix.columns:
            continue
        i = matrix.index.get_loc(player)
        j = matrix.columns.get_loc(role)
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            )
        )
        ax.text(
            j, i, label,
            ha="center", va="center",
            fontsize=7.5,
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Jaissle role score")

    ax.text(
        0,
        -0.12,
        "Numeric cells use Newcastle Jaissle benchmark scores. Gray cells are manual/non-modeled context, not zero scores. "
        "Wingers are split by tactical archetype; left/right fullbacks are separated.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"),
    )

    outpath = visualization_dir(project_root, "squad") / "newcastle_role_coverage_heatmap.png"
    fig.subplots_adjust(left=0.18, right=0.93, top=0.92, bottom=0.25)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")
    return outpath


def load_simple_role_ceiling(
    project_root: Path,
    patterns: list[str],
    score_candidates: list[str],
    price_cap: Optional[float],
) -> float:
    path = choose_existing_file(shortlist_root(project_root), patterns)
    if path is None:
        return np.nan
    try:
        df = read_table(path)
    except Exception:
        return np.nan

    score_col = choose_column(df, score_candidates)
    if score_col is None:
        return np.nan

    values = numeric(df[score_col])
    value_col = choose_column(
        df,
        ["Value €M", "market_value_eur_m", "market_value_eur", "Value"],
    )
    if price_cap is not None and value_col is not None:
        market_value = numeric(df[value_col])
        keep = market_value.isna() | (market_value <= price_cap)
        values = values[keep]

    return float(values.max()) if values.notna().any() else np.nan


def external_ceiling_by_role(
    project_root: Path,
    price_cap: Optional[float],
) -> dict[str, float]:
    ceilings = {
        "Sweeper / Build-Up GK": np.nan,
        "High-Line CB": load_simple_role_ceiling(
            project_root,
            ["center_back_jaissle_detailed.csv", "center_back.csv"],
            ["Jaissle CB", "jaissle_cb_score", "Role Score"],
            price_cap,
        ),
        "Left WB / FB": load_simple_role_ceiling(
            project_root,
            ["left_wingback_recruitment_progressive.csv", "wingback.csv"],
            ["Jaissle FB Score", "jaissle_fb_score", "Role Score"],
            price_cap,
        ),
        # Until we have a right-side-specific Jaissle analyzer, use the generic
        # wingback market as the candidate universe and benchmark upgrades
        # against Livramento on the Newcastle side.
        "Right WB / FB": load_simple_role_ceiling(
            project_root,
            ["wingback.csv", "left_wingback_recruitment_progressive.csv"],
            ["role_fit", "Role Fit", "Jaissle FB Score", "Role Score"],
            price_cap,
        ),
        "Holding Controller": load_simple_role_ceiling(
            project_root,
            ["holding_pivot.csv", "*holding*pivot*.csv"],
            ["role_fit", "Role Fit", "Role Score"],
            price_cap,
        ),
        "Advanced / Workhorse 8": load_simple_role_ceiling(
            project_root,
            ["workhorse_cm_jaissle_detailed.csv", "advanced_pivot.csv"],
            ["Jaissle CM", "jaissle_workhorse_score", "Role Score"],
            price_cap,
        ),
        "Creative 10": load_simple_role_ceiling(
            project_root,
            ["creative_10_jaissle_detailed.csv", "creative_10.csv"],
            ["CAM Fit", "jaissle_cam_score", "Role Score", "role_fit"],
            price_cap,
        ),
        "Wide Creator": load_simple_role_ceiling(
            project_root,
            ["winger_jaissle_archetypes.csv"],
            ["Creator"],
            price_cap,
        ),
        "Vertical / Box Winger": load_simple_role_ceiling(
            project_root,
            ["winger_jaissle_archetypes.csv"],
            ["Vertical/Box"],
            price_cap,
        ),
        "Hybrid Winger": load_simple_role_ceiling(
            project_root,
            ["winger_jaissle_archetypes.csv"],
            ["Hybrid"],
            price_cap,
        ),
        "Focal Striker": load_simple_role_ceiling(
            project_root,
            ["focal_striker_jaissle_detailed.csv", "focal_striker.csv"],
            ["Jaissle ST", "jaissle_striker_score", "Role Score"],
            price_cap,
        ),
    }
    return ceilings


def current_benchmark_by_role(bench: pd.DataFrame) -> dict[str, tuple[float, str]]:
    current = {}

    for _, row in bench.iterrows():
        role = canonical_benchmark_role(row)
        if role is None or pd.isna(row["_score"]):
            continue
        player = str(row["_player"])
        score = float(row["_score"])

        # User specifically wants Livramento as the right-back upgrade
        # benchmark rather than hiding the need behind Dedić's stronger score.
        if role == "Right WB / FB" and player != "Tino Livramento":
            continue

        # Keep the best player within each remaining role.
        if role not in current or score > current[role][0]:
            current[role] = (score, player)

    return current


def create_newcastle_role_gap_chart(
    project_root: Path,
    bench: pd.DataFrame,
    threshold: float = 52.5,
    price_cap: Optional[float] = 70.0,
) -> Optional[Path]:
    current = current_benchmark_by_role(bench)
    external = external_ceiling_by_role(project_root, price_cap)

    roles = SQUAD_ROLE_COLUMNS
    y = np.arange(len(roles))
    width = 0.34

    fig, ax = plt.subplots(figsize=(15.5, max(8.5, 0.78 * len(roles) + 2)))

    current_vals = []
    external_vals = []
    for role in roles:
        current_vals.append(current.get(role, (np.nan, ""))[0])
        external_vals.append(external.get(role, np.nan))

    current_arr = np.array(current_vals, dtype=float)
    external_arr = np.array(external_vals, dtype=float)

    ax.barh(
        y - width / 2,
        np.nan_to_num(current_arr, nan=0.0),
        height=width,
        label="Newcastle benchmark",
        color="tab:blue",
        alpha=0.78,
    )
    ax.barh(
        y + width / 2,
        np.nan_to_num(external_arr, nan=0.0),
        height=width,
        label=f"External ceiling ≤ €{price_cap:.0f}M" if price_cap is not None else "External ceiling",
        color="tab:green",
        alpha=0.78,
    )
    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        color="red",
        label=f"Review threshold {threshold:.1f}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(roles)
    ax.invert_yaxis()
    ax.set_xlabel("Role score / fit score")
    ax.set_title(
        "Newcastle Tactical Role Coverage vs External Market",
        fontsize=16,
        pad=14,
    )
    ax.grid(True, axis="x", alpha=0.23)

    for yi, role in enumerate(roles):
        cval = current_arr[yi]
        eval_ = external_arr[yi]

        if np.isfinite(cval):
            player = current.get(role, (np.nan, ""))[1]
            ax.text(
                cval + 0.35,
                yi - width / 2,
                f"{cval:.1f} — {player}",
                va="center",
                fontsize=8.5,
            )
        else:
            manual = {
                "Sweeper / Build-Up GK": "Horníček: manually trusted / GK model not built",
                "Holding Controller": "Nico reference; dedicated controller score not built",
                "Hybrid Winger": "No Newcastle benchmark",
            }.get(role, "No current benchmark")
            ax.text(
                1.0,
                yi - width / 2,
                manual,
                va="center",
                fontsize=8.2,
                color="dimgray",
                style="italic",
            )

        if np.isfinite(eval_):
            ax.text(
                eval_ + 0.35,
                yi + width / 2,
                f"{eval_:.1f}",
                va="center",
                fontsize=8.5,
            )
        else:
            ax.text(
                1.0,
                yi + width / 2,
                "External model not available",
                va="center",
                fontsize=8.2,
                color="dimgray",
                style="italic",
            )

    ax.legend(loc="lower right")
    ax.set_xlim(0, max(72, np.nanmax(external_arr) + 5 if np.isfinite(external_arr).any() else 72))

    ax.text(
        0.01,
        -0.13,
        "Fixed tactical roles prevent duplicate labels. Right-back uses Livramento as the upgrade benchmark. "
        "GK/controller missing scores are shown as model gaps rather than false zeros. "
        "Dedicated Jaissle scores are used where available; controller/RB fallback pools use role fit until their own analyzers are built.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"),
    )

    outpath = visualization_dir(project_root, "squad") / "newcastle_role_gap_summary.png"
    fig.subplots_adjust(left=0.25, right=0.97, top=0.92, bottom=0.17)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")
    return outpath


# -----------------------------------------------------------------------------
# Extended portfolio visualizations
# -----------------------------------------------------------------------------

def create_goalkeeper_visualization(project_root: Path) -> Optional[Path]:
    path = shortlist_root(project_root) / "goalkeeper_jaissle_detailed.csv"
    if not path.exists():
        print("[skip] Goalkeeper: run analyze_goalkeeper.py first")
        return None
    df = pd.read_csv(path)
    modeled = df[df["Jaissle GK Score"].notna()].copy()
    if modeled.empty:
        return None

    comps = ["Shot Stopping", "Sweeper", "Distribution", "Box Command"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [3.2, 1.5]})
    x = np.arange(len(comps)); width = 0.36
    for i, (_, row) in enumerate(modeled.iterrows()):
        vals = [row[c] for c in comps]
        ax.bar(x + (i - (len(modeled)-1)/2)*width, vals, width=width, label=row["Player"], alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels(comps, rotation=18, ha="right")
    ax.set_ylabel("Component score")
    ax.set_title("Jaissle Goalkeeper Profile — Horníček vs Mendy Reference")
    ax.set_ylim(35, 85); ax.grid(True, axis="y", alpha=.25); ax.legend()

    bx.axis("off")
    bx.set_title("Newcastle goalkeeper context", pad=12)
    y=.90
    for _, r in df[df["Club"].eq("Newcastle United")].iterrows():
        score = r.get("Jaissle GK Score", np.nan)
        score_txt = f" | score {score:.1f}" if pd.notna(score) else " | not modeled"
        bx.text(.02,y,f"{r['Player']}\n{r['Role']} | trust {r['Trust']}{score_txt}\n{r['Future']}",va="top",fontsize=10,
                bbox=dict(boxstyle="round,pad=.35",fc="#f5f5f5",ec="#cccccc"))
        y-=.21
    bx.text(.02,.05,"Interpretation: Horníček's current profile is strongest as an aggressive sweeper/shot-stopper; distribution and high-claim output trail the Mendy reference.",fontsize=9,wrap=True)
    out=visualization_dir(project_root,"goalkeeper")/"goalkeeper_jaissle_profile.png"
    fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.14,wspace=.25)
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig); print(f"[saved] {out}"); return out


def create_squad_depth_visualization(project_root: Path) -> Optional[Path]:
    path=shortlist_root(project_root)/"squad_depth_analysis.csv"
    if not path.exists():
        print("[skip] Squad depth: run build_squad_depth_analysis.py first"); return None
    df=pd.read_csv(path)
    roles=df["Role"].tolist(); y=np.arange(len(df)); h=.22
    fig,ax=plt.subplots(figsize=(14,max(8,.75*len(df)+2)))
    for i,label in enumerate(["Starter","2nd choice","3rd choice"]):
        vals=pd.to_numeric(df[f"Score {i+1}"],errors="coerce")
        ax.barh(y+(i-1)*h, vals.fillna(0), height=h, label=label, alpha=.78)
        for yi,v in enumerate(vals):
            if pd.notna(v): ax.text(v+.25,yi+(i-1)*h,f"{v:.1f}",va="center",fontsize=8)
    ax.axvline(52.5,color="red",linestyle="--",lw=1.1,label="52.5 threshold")
    ax.set_yticks(y); ax.set_yticklabels(roles); ax.invert_yaxis(); ax.set_xlabel("Role score")
    ax.set_title("Newcastle Tactical Depth by Role"); ax.grid(True,axis="x",alpha=.22); ax.legend(loc="lower right")
    out=visualization_dir(project_root,"squad")/"newcastle_tactical_role_depth.png"
    fig.subplots_adjust(left=.24,right=.97,top=.92,bottom=.10); fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig); print(f"[saved] {out}"); return out


def create_value_efficiency_visualization(project_root: Path) -> Optional[Path]:
    path=shortlist_root(project_root)/"recruitment_value_analysis.csv"
    if not path.exists():
        print("[skip] Value analysis: run build_recruitment_value_analysis.py first"); return None
    df=pd.read_csv(path)
    df["Upgrade per €M"]=pd.to_numeric(df["Upgrade per €M"],errors="coerce")
    df["Upgrade"]=pd.to_numeric(df["Upgrade"],errors="coerce")
    df=df[df["Upgrade"].gt(0)&df["Upgrade per €M"].notna()].sort_values("Upgrade per €M",ascending=False).head(25)
    if df.empty:return None
    labels=(df["Player"].astype(str)+" — "+df["Role"].astype(str)).tolist(); y=np.arange(len(df))
    fig,ax=plt.subplots(figsize=(13,max(8,.38*len(df)+2)))
    ax.barh(y,df["Upgrade per €M"],alpha=.8); ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Projected score upgrade per €1M market value"); ax.set_title("Recruitment Value Efficiency — Positive Upgrades")
    ax.grid(True,axis="x",alpha=.22)
    out=visualization_dir(project_root,"value")/"upgrade_per_euro.png"
    fig.subplots_adjust(left=.35,right=.97,top=.92,bottom=.10); fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig); print(f"[saved] {out}"); return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create Jaissle recruitment visualization pack.")
    parser.add_argument("--top-n", type=int, default=40, help="How many market players to plot per role (default: 40)")
    parser.add_argument("--annotate-n", type=int, default=20, help="How many market names to annotate per chart before extra outlier labels (default: 18)")
    parser.add_argument("--price-cap", type=float, default=70.0, help="External target price cap in €M (default: 70). Use negative to disable.")
    parser.add_argument("--threshold", type=float, default=52.5, help="Role-fit threshold for benchmark context (default: 52.5)")
    args = parser.parse_args()

    project_root = find_project_root()
    out_root = visualizations_root(project_root)
    price_cap = None if args.price_cap is not None and args.price_cap < 0 else args.price_cap

    print("=" * 110)
    print("JAISSLE IMAGE CREATION")
    print("=" * 110)
    print(f"Project root: {project_root}")
    print(f"Output dir   : {out_root}")

    benchmarks = None
    try:
        benchmarks = load_newcastle_benchmarks(project_root)
        print(f"Loaded Newcastle benchmark rows: {len(benchmarks)}")
    except Exception as exc:
        print(f"[warn] Could not load Newcastle benchmarks: {exc}")

    saved = []
    for spec in ROLE_SPECS:
        path = create_role_chart(
            project_root=project_root,
            spec=spec,
            top_n=args.top_n,
            annotate_n=args.annotate_n,
            price_cap=price_cap,
            threshold=args.threshold,
            benchmarks=benchmarks,
        )
        if path is not None:
            saved.append(path)

    gk = create_goalkeeper_visualization(project_root)
    if gk is not None:
        saved.append(gk)

    depth_chart = create_squad_depth_visualization(project_root)
    if depth_chart is not None:
        saved.append(depth_chart)

    value_chart = create_value_efficiency_visualization(project_root)
    if value_chart is not None:
        saved.append(value_chart)

    if benchmarks is not None:
        cov = create_newcastle_coverage_heatmap(project_root, benchmarks)
        if cov is not None:
            saved.append(cov)
        gap = create_newcastle_role_gap_chart(project_root, benchmarks, threshold=args.threshold, price_cap=price_cap)
        if gap is not None:
            saved.append(gap)

    print("\n" + "=" * 110)
    print("SAVED FILES")
    print("=" * 110)
    if not saved:
        print("No images were generated.")
    else:
        for p in saved:
            print(p)


if __name__ == "__main__":
    main()
