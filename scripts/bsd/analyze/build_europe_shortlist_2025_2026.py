
from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import re

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[3]

HISTORICAL_RAW = ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_season_metrics.parquet"
TRANSFERMARKT_ENRICHED = (
    ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_percentiles_enriched.parquet"
)
CURRENT_SUMMARY = ROOT / "data" / "processed" / "bsd" / "newcastle_complete_role_fit_summary.parquet"
CURRENT_LONG = ROOT / "data" / "processed" / "bsd" / "newcastle_complete_role_fit_long.parquet"

ROLE_MODEL_SCRIPT = Path(__file__).resolve().with_name(
    "analyze_newcastle_2025_2026_historical_roles.py"
)

OUT_ALL = ROOT / "data" / "processed" / "bsd" / "europe_2025_26_full_role_shortlist.parquet"
OUT_TOP = ROOT / "data" / "processed" / "bsd" / "europe_2025_26_top_role_shortlist.csv"


# ======================================================================================
# SETTINGS
# ======================================================================================

MIN_MINUTES = 900
TOP_N_PER_ROLE = 20

# We are not using this to exclude older players completely; it is only used
# for the "value target" flag below.
PREFERRED_MAX_AGE = 27

# Soft value bands for depth/value shopping. These do NOT exclude expensive stars.
DEPTH_VALUE_MAX_EUR_M = 30.0
STARTER_VALUE_MAX_EUR_M = 55.0

ROLE_ORDER = [
    "center_back",
    "vertical_wingback",
    "holding_pivot",
    "advanced_pivot",
    "creative_cam",
    "vertical_winger",
    "inside_creator",
    "focal_striker",
    "second_striker",
]

ROLE_POSITION_FILTERS = {
    "center_back": {"CB"},
    "vertical_wingback": {"FB/WB"},
    "holding_pivot": {"DM/CM"},
    "advanced_pivot": {"DM/CM", "AM"},
    "creative_cam": {"AM", "DM/CM", "W"},
    "vertical_winger": {"W", "AM"},
    "inside_creator": {"W", "AM"},
    "focal_striker": {"ST"},
    "second_striker": {"ST", "AM", "W"},
}


# ======================================================================================
# LOAD ROLE MODEL
# ======================================================================================

def load_role_module():
    if not ROLE_MODEL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find role model script:\n{ROLE_MODEL_SCRIPT}\n"
            "Place this shortlist script in the same scripts/bsd/analyze folder."
        )

    spec = importlib.util.spec_from_file_location("newcastle_role_model", ROLE_MODEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# GENERAL HELPERS
# ======================================================================================

def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).lower().strip()
    s = (
        s.replace("á", "a").replace("à", "a").replace("ä", "a").replace("â", "a")
         .replace("ã", "a").replace("å", "a")
         .replace("é", "e").replace("è", "e").replace("ë", "e").replace("ê", "e")
         .replace("í", "i").replace("ì", "i").replace("ï", "i").replace("î", "i")
         .replace("ó", "o").replace("ò", "o").replace("ö", "o").replace("ô", "o").replace("õ", "o")
         .replace("ú", "u").replace("ù", "u").replace("ü", "u").replace("û", "u")
         .replace("ç", "c").replace("ñ", "n")
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_existing(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_series(df: pd.DataFrame, col):
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def infer_value_millions(series: pd.Series) -> pd.Series:
    """
    Transfermarkt snapshots can store value in euros, millions, or formatted text.
    Normalize to EUR millions using conservative heuristics.
    """
    s = series.copy()

    def parse_one(v):
        if pd.isna(v):
            return np.nan

        if isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            if x > 100_000:       # likely raw EUR
                return x / 1_000_000.0
            return x              # likely already millions

        txt = str(v).strip().lower().replace("€", "").replace("$", "").replace(",", "")
        mult = 1.0
        if txt.endswith("m"):
            txt = txt[:-1]
            mult = 1.0
        elif txt.endswith("k"):
            txt = txt[:-1]
            mult = 0.001

        try:
            x = float(txt)
        except ValueError:
            return np.nan

        if x > 100_000:
            return x / 1_000_000.0
        return x * mult

    return s.map(parse_one)


def ft_in(height_cm):
    if pd.isna(height_cm):
        return ""
    try:
        inches = round(float(height_cm) / 2.54)
    except Exception:
        return ""
    return f"{inches // 12}'{inches % 12}\""


# ======================================================================================
# ENRICHMENT MERGE
# ======================================================================================

def prepare_transfermarkt(tm: pd.DataFrame) -> pd.DataFrame:
    out = tm.copy()

    name_col = first_existing(
        out,
        ["player_name", "name", "tm_player_name", "transfermarkt_name"],
    )
    league_col = first_existing(
        out,
        ["league_slug", "league", "competition_slug", "league_name"],
    )
    age_col = first_existing(out, ["age", "player_age"])
    value_col = first_existing(
        out,
        ["market_value_eur", "market_value", "market_value_m", "value_eur", "value"],
    )
    height_col = first_existing(out, ["height_cm", "height", "player_height_cm"])
    club_col = first_existing(
        out,
        ["club_name", "team_name", "transfermarkt_club", "club"],
    )
    player_id_col = first_existing(
        out,
        ["player_id", "bsd_player_id"],
    )

    result = pd.DataFrame(index=out.index)

    if player_id_col:
        result["player_id"] = out[player_id_col]

    result["tm_player_name"] = out[name_col] if name_col else pd.NA
    result["tm_name_norm"] = result["tm_player_name"].map(norm_text)
    result["tm_league"] = out[league_col] if league_col else pd.NA
    result["tm_league_norm"] = result["tm_league"].map(norm_text)
    result["age"] = numeric_series(out, age_col)
    result["market_value_eur_m"] = infer_value_millions(out[value_col]) if value_col else np.nan
    result["height_cm"] = numeric_series(out, height_col)
    result["club_name"] = out[club_col] if club_col else pd.NA

    return result


def merge_transfermarkt(hist: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    out = hist.copy()
    tm2 = prepare_transfermarkt(tm)

    hist_name_col = first_existing(out, ["player_name", "name"])
    if hist_name_col is None:
        raise RuntimeError("Historical file has no player_name/name column.")

    out["_name_norm"] = out[hist_name_col].map(norm_text)

    hist_league_col = first_existing(out, ["league_slug", "league_name", "league"])
    if hist_league_col:
        out["_league_norm"] = out[hist_league_col].map(norm_text)
    else:
        out["_league_norm"] = ""

    # Prefer player_id when available and populated.
    if "player_id" in out.columns and "player_id" in tm2.columns:
        left_nonnull = out["player_id"].notna().sum()
        right_nonnull = tm2["player_id"].notna().sum()

        if left_nonnull and right_nonnull:
            tm_one = (
                tm2.sort_values("market_value_eur_m", ascending=False)
                   .drop_duplicates("player_id")
            )
            merged = out.merge(tm_one, on="player_id", how="left")

            missing = merged["tm_player_name"].isna()
            if not missing.any():
                return merged

            # Fill unmatched rows with name+league matching.
            fallback = name_league_merge(out.loc[missing].copy(), tm2)
            for col in [
                "tm_player_name", "age", "market_value_eur_m",
                "height_cm", "club_name", "tm_league",
            ]:
                if col in fallback.columns:
                    merged.loc[missing, col] = fallback[col].values
            return merged

    return name_league_merge(out, tm2)


def name_league_merge(hist: pd.DataFrame, tm2: pd.DataFrame) -> pd.DataFrame:
    left = hist.copy()

    # A one-row-per-name/league table is more robust than fuzzy matching here.
    right = (
        tm2.sort_values("market_value_eur_m", ascending=False)
           .drop_duplicates(["tm_name_norm", "tm_league_norm"])
    )

    merged = left.merge(
        right,
        left_on=["_name_norm", "_league_norm"],
        right_on=["tm_name_norm", "tm_league_norm"],
        how="left",
    )

    # Second pass: exact normalized name regardless of league.
    missing = merged["tm_player_name"].isna()
    if missing.any():
        name_only = (
            tm2.sort_values("market_value_eur_m", ascending=False)
               .drop_duplicates("tm_name_norm")
               .set_index("tm_name_norm")
        )
        for idx in merged.index[missing]:
            key = merged.at[idx, "_name_norm"]
            if key in name_only.index:
                row = name_only.loc[key]
                for col in [
                    "tm_player_name", "age", "market_value_eur_m",
                    "height_cm", "club_name", "tm_league",
                ]:
                    if col in row.index:
                        merged.at[idx, col] = row[col]

    return merged


# ======================================================================================
# ROLE SCORING
# ======================================================================================

def score_all_players(hist: pd.DataFrame, role_module) -> pd.DataFrame:
    hist = role_module.ensure_common_columns(hist.copy())

    required = ["player_name", "minutes", "position_group", "league_slug"]
    missing = [c for c in required if c not in hist.columns]
    if missing:
        raise RuntimeError(f"Historical file missing required columns: {missing}")

    hist["minutes"] = pd.to_numeric(hist["minutes"], errors="coerce")
    eligible = hist[hist["minutes"] >= MIN_MINUTES].copy()

    rows = []

    for idx, player in eligible.iterrows():
        player_pos = str(player.get("position_group", ""))

        for role_key in ROLE_ORDER:
            if role_key not in role_module.ROLE_MODELS:
                continue

            if player_pos not in ROLE_POSITION_FILTERS[role_key]:
                continue

            role_fit, coverage, metric_z = role_module.role_fit_for_row(
                player,
                role_key,
                hist,
            )

            if pd.isna(role_fit):
                continue

            min_cov = role_module.minimum_role_coverage(role_key)
            if coverage < min_cov:
                continue

            model = role_module.ROLE_MODELS[role_key]

            strengths = []
            weaknesses = []

            for label, metric in model["display"]:
                z = metric_z.get(metric, np.nan)
                if pd.isna(z):
                    continue
                strengths.append((label, float(z)))
                weaknesses.append((label, float(z)))

            strengths = sorted(strengths, key=lambda x: x[1], reverse=True)[:3]
            weaknesses = sorted(weaknesses, key=lambda x: x[1])[:3]

            rows.append({
                "_source_index": idx,
                "player_name": player["player_name"],
                "league_slug": player["league_slug"],
                "position_group": player["position_group"],
                "minutes": player["minutes"],
                "role_key": role_key,
                "role": model["label"],
                "role_fit": float(role_fit),
                "metric_coverage_pct": round(float(coverage) * 100.0, 1),
                "league_strength": role_module.LEAGUE_STRENGTH.get(
                    str(player["league_slug"]), np.nan
                ),
                "strengths": "; ".join(f"{label} {z:+.2f}z" for label, z in strengths),
                "weaknesses": "; ".join(f"{label} {z:+.2f}z" for label, z in weaknesses),
            })

    return pd.DataFrame(rows)


# ======================================================================================
# VALUE / MARKET LAYER
# ======================================================================================

def add_market_layer(scored: pd.DataFrame, enriched_hist: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()

    metadata_cols = [
        "age", "market_value_eur_m", "height_cm", "club_name",
    ]

    meta = enriched_hist.copy()
    meta = meta.reset_index().rename(columns={"index": "_source_index"})

    keep = ["_source_index"] + [c for c in metadata_cols if c in meta.columns]
    meta = meta[keep].drop_duplicates("_source_index")

    out = out.merge(meta, on="_source_index", how="left")

    out["height"] = out["height_cm"].map(ft_in)

    # Value efficiency is intentionally simple and transparent.
    # 50 = average role fit. We value points above/below 50 relative to price.
    # A small floor prevents division by nearly-zero values.
    value_floor = out["market_value_eur_m"].clip(lower=3.0)
    out["fit_above_avg"] = out["role_fit"] - 50.0
    out["fit_per_10m"] = out["fit_above_avg"] / value_floor * 10.0

    # Labels meant for triage, not automatic transfer decisions.
    out["market_bucket"] = np.select(
        [
            out["market_value_eur_m"] <= 15,
            out["market_value_eur_m"] <= DEPTH_VALUE_MAX_EUR_M,
            out["market_value_eur_m"] <= STARTER_VALUE_MAX_EUR_M,
        ],
        [
            "Low-cost",
            "Depth/value",
            "Starter-range",
        ],
        default="Premium",
    )

    out["value_target"] = (
        (out["role_fit"] >= 55.0)
        & (out["market_value_eur_m"] <= DEPTH_VALUE_MAX_EUR_M)
        & (out["age"] <= PREFERRED_MAX_AGE)
    )

    out["starter_target"] = (
        (out["role_fit"] >= 58.0)
        & (out["market_value_eur_m"] <= STARTER_VALUE_MAX_EUR_M)
        & (out["age"] <= 29)
    )

    return out


# ======================================================================================
# NEWCASTLE REFERENCES
# ======================================================================================

def load_newcastle_refs() -> pd.DataFrame:
    if not CURRENT_LONG.exists():
        return pd.DataFrame()

    current = pd.read_parquet(CURRENT_LONG).copy()

    rename_map = {}
    if "player_name" not in current.columns:
        for c in ["Newcastle Player", "Player"]:
            if c in current.columns:
                rename_map[c] = "player_name"
                break
    if "role_key" not in current.columns and "Role Key" in current.columns:
        rename_map["Role Key"] = "role_key"
    if "role_fit" not in current.columns:
        for c in ["Role Fit", "role_fit_score"]:
            if c in current.columns:
                rename_map[c] = "role_fit"
                break

    current = current.rename(columns=rename_map)

    required = {"player_name", "role_key", "role_fit"}
    if not required.issubset(current.columns):
        return pd.DataFrame()

    refs = current[
        current["role_key"].isin(ROLE_ORDER)
        & pd.to_numeric(current["role_fit"], errors="coerce").notna()
    ].copy()

    refs["role_fit"] = pd.to_numeric(refs["role_fit"], errors="coerce")
    return refs


# ======================================================================================
# PRINTING
# ======================================================================================

def format_money(v):
    if pd.isna(v):
        return ""
    return f"€{float(v):.1f}M"


def print_role_shortlist(df: pd.DataFrame, refs: pd.DataFrame, role_module):
    print()
    print("=" * 180)
    print("EUROPE 2025/26 — FULL JAISSLE ROLE SHORTLIST")
    print("=" * 180)
    print(
        "Role Fit uses the same raw-metric z-score models as the Newcastle historical analysis. "
        "League Strength is context only; it does not change Role Fit."
    )
    print(
        f"Eligibility: {MIN_MINUTES}+ minutes. Top {TOP_N_PER_ROLE} external candidates per role are shown."
    )

    for role_key in ROLE_ORDER:
        model = role_module.ROLE_MODELS[role_key]
        role_name = model["label"]

        print()
        print("=" * 180)
        print(role_name.upper())
        print("=" * 180)

        if not refs.empty:
            role_refs = (
                refs[refs["role_key"].eq(role_key)]
                .sort_values("role_fit", ascending=False)
                .head(5)
            )

            if not role_refs.empty:
                print("NEWCASTLE REFERENCES")
                for _, r in role_refs.iterrows():
                    print(f"  {str(r['player_name']):<28} {float(r['role_fit']):>5.1f}")

        role_df = (
            df[df["role_key"].eq(role_key)]
            .sort_values(
                ["role_fit", "fit_per_10m", "market_value_eur_m"],
                ascending=[False, False, True],
            )
            .head(TOP_N_PER_ROLE)
            .copy()
        )

        if role_df.empty:
            print("No eligible candidates.")
            continue

        display = role_df[
            [
                "player_name", "club_name", "league_slug", "age", "height",
                "minutes", "role_fit", "metric_coverage_pct",
                "market_value_eur_m", "fit_per_10m", "market_bucket",
                "value_target", "starter_target", "strengths", "weaknesses",
            ]
        ].copy()

        display["market_value_eur_m"] = display["market_value_eur_m"].map(format_money)
        display["age"] = display["age"].round(0).astype("Int64")
        display["minutes"] = display["minutes"].round(0).astype("Int64")
        display["role_fit"] = display["role_fit"].round(1)
        display["metric_coverage_pct"] = display["metric_coverage_pct"].round(0)
        display["fit_per_10m"] = display["fit_per_10m"].round(2)

        display = display.rename(columns={
            "player_name": "Player",
            "club_name": "Club",
            "league_slug": "League",
            "age": "Age",
            "height": "Ht",
            "minutes": "Min",
            "role_fit": "Fit",
            "metric_coverage_pct": "Coverage",
            "market_value_eur_m": "Value",
            "fit_per_10m": "Fit/€10m",
            "market_bucket": "Market",
            "value_target": "Value Target",
            "starter_target": "Starter Target",
            "strengths": "Strengths",
            "weaknesses": "Weaknesses",
        })

        print(display.to_string(index=False))


def print_value_targets(df: pd.DataFrame):
    print()
    print("=" * 180)
    print("CROSS-ROLE VALUE TARGETS")
    print("=" * 180)

    targets = (
        df[df["value_target"]]
        .sort_values(["fit_per_10m", "role_fit"], ascending=False)
        .drop_duplicates(["player_name", "role_key"])
        .head(40)
        .copy()
    )

    if targets.empty:
        print("No players met the value-target rule.")
        return

    display = targets[
        [
            "player_name", "role", "club_name", "league_slug", "age",
            "role_fit", "market_value_eur_m", "fit_per_10m", "strengths",
        ]
    ].copy()

    display["market_value_eur_m"] = display["market_value_eur_m"].map(format_money)
    display["age"] = display["age"].round(0).astype("Int64")
    display["role_fit"] = display["role_fit"].round(1)
    display["fit_per_10m"] = display["fit_per_10m"].round(2)

    display = display.rename(columns={
        "player_name": "Player",
        "role": "Role",
        "club_name": "Club",
        "league_slug": "League",
        "age": "Age",
        "role_fit": "Fit",
        "market_value_eur_m": "Value",
        "fit_per_10m": "Fit/€10m",
        "strengths": "Strengths",
    })

    print(display.to_string(index=False))


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    for path in [HISTORICAL_RAW, TRANSFERMARKT_ENRICHED]:
        if not path.exists():
            raise FileNotFoundError(path)

    role_module = load_role_module()

    hist = pd.read_parquet(HISTORICAL_RAW)
    hist = role_module.ensure_common_columns(hist)

    tm = pd.read_parquet(TRANSFERMARKT_ENRICHED)
    enriched_hist = merge_transfermarkt(hist, tm)

    scored = score_all_players(hist, role_module)
    final = add_market_layer(scored, enriched_hist)

    # Exclude current Newcastle players from the external market results where possible.
    refs = load_newcastle_refs()
    if not refs.empty:
        current_names = set(refs["player_name"].map(norm_text))
        final = final[~final["player_name"].map(norm_text).isin(current_names)].copy()

    # Stable rank within each role.
    final["role_rank"] = (
        final.groupby("role_key")["role_fit"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    final = final.sort_values(
        ["role_key", "role_fit", "fit_per_10m"],
        ascending=[True, False, False],
    )

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_ALL, index=False)

    top_export = (
        final.sort_values(["role_key", "role_fit"], ascending=[True, False])
        .groupby("role_key", group_keys=False)
        .head(TOP_N_PER_ROLE)
        .copy()
    )
    top_export.to_csv(OUT_TOP, index=False)

    print_role_shortlist(final, refs, role_module)
    print_value_targets(final)

    print()
    print("Saved:")
    print(OUT_ALL)
    print(OUT_TOP)


if __name__ == "__main__":
    main()
