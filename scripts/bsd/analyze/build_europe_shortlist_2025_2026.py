
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

SHORTLIST_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"

OUT_ALL = SHORTLIST_DIR / "europe_2025_26_full_role_shortlist.parquet"
OUT_UNIQUE = SHORTLIST_DIR / "europe_2025_26_unique_shortlist.csv"
OUT_HTML = SHORTLIST_DIR / "europe_2025_26_shortlist_report.html"


# ======================================================================================
# SETTINGS
# ======================================================================================

MIN_MINUTES = 900
TOP_N_PER_ROLE = 20
FULFILLED_ROLE_FIT = 55.0
UNIQUE_REPORT_LIMIT = 150

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

ROLE_EXPORT_FILES = {
    "center_back": "center_back.csv",
    "vertical_wingback": "wingback.csv",
    "holding_pivot": "holding_pivot.csv",
    "advanced_pivot": "advanced_pivot.csv",
    "creative_cam": "creative_10.csv",
    "vertical_winger": "vertical_winger.csv",
    "inside_creator": "inside_creator.csv",
    "focal_striker": "focal_striker.csv",
    "second_striker": "second_striker.csv",
}

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


def first_nonempty_existing(df: pd.DataFrame, candidates):
    existing = []

    for col in candidates:
        if col not in df.columns:
            continue

        existing.append(col)

        series = df[col]

        if series.notna().sum() == 0:
            continue

        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            nonblank = (
                series.dropna()
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )
            if nonblank > 0:
                return col
        else:
            return col

    return existing[0] if existing else None


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
    age_col = first_nonempty_existing(
        out,
        [
            "age",
            "player_age",
            "tm_age",
            "age_2025",
            "age_2026",
            "age_at_season_start",
            "age_at_season_end",
        ],
    )

    dob_col = first_nonempty_existing(
        out,
        [
            "date_of_birth",
            "birth_date",
            "dob",
            "player_date_of_birth",
            "tm_date_of_birth",
        ],
    )

    value_col = first_nonempty_existing(
        out,
        [
            "market_value_eur",
            "market_value",
            "market_value_m",
            "value_eur",
            "value",
            "tm_market_value_eur",
        ],
    )

    height_col = first_nonempty_existing(
        out,
        [
            "height_cm",
            "height",
            "player_height_cm",
            "tm_height_cm",
            "height_in_cm",
        ],
    )

    club_col = first_nonempty_existing(
        out,
        [
            "club_name",
            "team_name",
            "transfermarkt_club",
            "club",
            "current_club_name",
            "tm_club_name",
            "squad_name",
            "current_team_name",
            "current_club",
            "club",
            "team",
            "tm_team_name",
            "transfermarkt_team_name",
        ],
    )
    tm_position_col = first_nonempty_existing(
        out,
        ["tm_position", "position", "player_position", "main_position"],
    )
    tm_sub_position_col = first_nonempty_existing(
        out,
        ["tm_sub_position", "sub_position", "player_sub_position", "detailed_position"],
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
    if age_col:
        result["age"] = numeric_series(out, age_col)
    else:
        result["age"] = np.nan

    # If age is unavailable but DOB exists, derive age at 30 June 2026,
    # approximately the end of the 2025/26 European season.
    if dob_col:
        dob = pd.to_datetime(out[dob_col], errors="coerce")
        season_end = pd.Timestamp("2026-06-30")
        derived_age = (
            season_end.year
            - dob.dt.year
            - (
                (season_end.month < dob.dt.month)
                | (
                    (season_end.month == dob.dt.month)
                    & (season_end.day < dob.dt.day)
                )
            ).astype("Int64")
        )
        current_age = pd.to_numeric(
            result["age"], errors="coerce"
        ).astype("float64")

        derived_age_numeric = pd.to_numeric(
            derived_age, errors="coerce"
        ).astype("float64")

        result["age"] = current_age.fillna(
            derived_age_numeric
        )

    result["market_value_eur_m"] = infer_value_millions(out[value_col]) if value_col else np.nan
    result["height_cm"] = numeric_series(out, height_col)
    result["club_name"] = out[club_col] if club_col else pd.NA
    result["tm_position"] = out[tm_position_col] if tm_position_col else pd.NA
    result["tm_sub_position"] = out[tm_sub_position_col] if tm_sub_position_col else pd.NA

    result.attrs["detected_age_col"] = age_col
    result.attrs["detected_dob_col"] = dob_col
    result.attrs["detected_club_col"] = club_col
    result.attrs["detected_value_col"] = value_col
    result.attrs["detected_height_col"] = height_col
    result.attrs["detected_tm_position_col"] = tm_position_col
    result.attrs["detected_tm_sub_position_col"] = tm_sub_position_col

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
                "height_cm", "club_name", "tm_league", "tm_position", "tm_sub_position",
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
                    "height_cm", "club_name", "tm_league", "tm_position", "tm_sub_position",
                ]:
                    if col in row.index:
                        merged.at[idx, col] = row[col]

    return merged



# ======================================================================================
# POSITION / ROLE ELIGIBILITY
# ======================================================================================

def natural_position_code(row) -> str:
    sub = norm_text(row.get("tm_sub_position", ""))
    pos = norm_text(row.get("tm_position", ""))

    value = f"{sub} {pos}".strip()

    if any(x in value for x in ["centre back", "center back", "central defender"]):
        return "CB"

    if any(
        x in value
        for x in [
            "left back", "right back", "full back", "fullback",
            "wing back", "wingback",
        ]
    ):
        return "FB/WB"

    if "defensive midfield" in value:
        return "DM"

    if "central midfield" in value or "centre midfield" in value:
        return "CM"

    if "attacking midfield" in value:
        return "AM"

    if any(x in value for x in ["left winger", "right winger", "winger"]):
        return "W"

    if "second striker" in value:
        return "SS"

    if any(x in value for x in ["centre forward", "center forward", "striker"]):
        return "ST"

    fallback = str(row.get("position_group", "")).strip()
    if fallback == "DM/CM":
        return "CM"
    return fallback


def natural_position_label(row) -> str:
    sub = row.get("tm_sub_position", pd.NA)
    pos = row.get("tm_position", pd.NA)

    if pd.notna(sub) and str(sub).strip():
        return str(sub).strip()

    if pd.notna(pos) and str(pos).strip():
        return str(pos).strip()

    return str(row.get("position_group", "")).strip()


ROLE_NATURAL_POSITION_FILTERS = {
    "center_back": {"CB"},
    "vertical_wingback": {"FB/WB"},
    "holding_pivot": {"DM", "CM"},
    "advanced_pivot": {"DM", "CM", "AM"},
    "creative_cam": {"CM", "AM", "W", "SS"},
    "vertical_winger": {"W", "AM", "SS"},
    "inside_creator": {"W", "AM", "SS"},
    "focal_striker": {"ST"},
    "second_striker": {"ST", "SS", "AM", "W"},
}


def role_position_eligible(row, role_key: str) -> bool:
    code = natural_position_code(row)
    return code in ROLE_NATURAL_POSITION_FILTERS.get(role_key, set())


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
        for role_key in ROLE_ORDER:
            if role_key not in role_module.ROLE_MODELS:
                continue

            # Use Transfermarkt's natural/detailed position when available.
            # This prevents a productive LWB such as Dimarco from being treated
            # as a CAM merely because his creation metrics are excellent.
            if not role_position_eligible(player, role_key):
                continue

            role_fit, coverage, metric_z = role_module.role_fit_for_row(
                player,
                hist,
                role_key,
                str(player["league_slug"]),
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
                "natural_position": natural_position_label(player),
                "natural_position_code": natural_position_code(player),
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
        "tm_position", "tm_sub_position",
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

    # Separate fit/price qualification from age qualification.
    # Missing age should not silently turn an otherwise valid target into False.
    out["age_known"] = out["age"].notna()

    out["value_fit_price_ok"] = (
        (out["role_fit"] >= 55.0)
        & (out["market_value_eur_m"] <= DEPTH_VALUE_MAX_EUR_M)
    )

    out["starter_fit_price_ok"] = (
        (out["role_fit"] >= 58.0)
        & (out["market_value_eur_m"] <= STARTER_VALUE_MAX_EUR_M)
    )

    out["value_target"] = np.where(
        ~out["age_known"],
        pd.NA,
        out["value_fit_price_ok"] & (out["age"] <= PREFERRED_MAX_AGE),
    )

    out["starter_target"] = np.where(
        ~out["age_known"],
        pd.NA,
        out["starter_fit_price_ok"] & (out["age"] <= 29),
    )

    out["needs_age_review"] = (
        ~out["age_known"]
        & (out["value_fit_price_ok"] | out["starter_fit_price_ok"])
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

    # Attach each current Newcastle player's actual current tactical position
    # from the summary so CB tables do not show Nico/Hall/etc as references.
    if CURRENT_SUMMARY.exists():
        summary = pd.read_parquet(CURRENT_SUMMARY).copy()

        name_col = first_existing(
            summary,
            ["player_name", "Newcastle Player", "Player"],
        )
        pos_col = first_existing(
            summary,
            ["current_position_group", "position_group", "26/27 Role", "current_role"],
        )

        if name_col and pos_col:
            pos_map = (
                summary[[name_col, pos_col]]
                .dropna(subset=[name_col])
                .drop_duplicates(name_col)
                .rename(columns={name_col: "player_name", pos_col: "current_position_group"})
            )
            current = current.merge(pos_map, on="player_name", how="left")

    current["role_fit"] = pd.to_numeric(current["role_fit"], errors="coerce")

    def current_role_ok(row):
        p = str(row.get("current_position_group", "")).strip()
        allowed = ROLE_POSITION_FILTERS.get(row["role_key"], set())
        return (not p) or (p in allowed)

    refs = current[
        current["role_key"].isin(ROLE_ORDER)
        & current["role_fit"].notna()
    ].copy()

    refs = refs[refs.apply(current_role_ok, axis=1)].copy()
    return refs


# ======================================================================================
# UNIQUE PLAYER SHORTLIST
# ======================================================================================

def build_unique_player_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["player_name", "club_name", "league_slug"]

    for _, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("role_fit", ascending=False).copy()
        best = group.iloc[0]

        all_role_fits = [
            (str(r["role"]), round(float(r["role_fit"]), 1))
            for _, r in group.iterrows()
        ]

        fulfilled = [
            (str(r["role"]), round(float(r["role_fit"]), 1))
            for _, r in group.iterrows()
            if float(r["role_fit"]) >= FULFILLED_ROLE_FIT
        ]

        starter_roles = [
            (str(r["role"]), round(float(r["role_fit"]), 1))
            for _, r in group.iterrows()
            if pd.notna(r["starter_target"]) and bool(r["starter_target"])
        ]

        value_roles = [
            (str(r["role"]), round(float(r["role_fit"]), 1))
            for _, r in group.iterrows()
            if pd.notna(r["value_target"]) and bool(r["value_target"])
        ]

        if starter_roles:
            target_tier = "Starter target"
        elif value_roles:
            target_tier = "Value target"
        elif fulfilled:
            target_tier = "Role fit"
        else:
            target_tier = "Watch"

        rows.append({
            "player_name": best["player_name"],
            "club_name": best["club_name"],
            "league_slug": best["league_slug"],
            "natural_position": best.get("natural_position", ""),
            "natural_position_code": best.get("natural_position_code", ""),
            "age": best["age"],
            "height": best["height"],
            "minutes": best["minutes"],
            "market_value_eur_m": best["market_value_eur_m"],
            "market_bucket": best["market_bucket"],
            "best_role": best["role"],
            "best_fit": float(best["role_fit"]),
            "best_fit_per_10m": float(best["fit_per_10m"]) if pd.notna(best["fit_per_10m"]) else np.nan,
            "target_tier": target_tier,
            "fulfilled_roles": fulfilled,
            "role_fits": all_role_fits,
            "starter_roles": starter_roles,
            "value_roles": value_roles,
            "best_role_strengths": best["strengths"],
            "best_role_weaknesses": best["weaknesses"],
            "max_coverage_pct": float(group["metric_coverage_pct"].max()),
            "role_count": len(group),
        })

    out = pd.DataFrame(rows)

    tier_order = {
        "Starter target": 0,
        "Value target": 1,
        "Role fit": 2,
        "Watch": 3,
    }

    out["_tier_sort"] = out["target_tier"].map(tier_order).fillna(9)
    out = out.sort_values(
        ["_tier_sort", "best_fit", "best_fit_per_10m"],
        ascending=[True, False, False],
    ).drop(columns="_tier_sort")

    return out.reset_index(drop=True)


def tuple_list_text(items) -> str:
    if not items:
        return ""
    return ", ".join(f"({role}, {score:.1f})" for role, score in items)


def html_escape(value) -> str:
    import html
    return html.escape("" if pd.isna(value) else str(value))


def save_html_report(unique_df: pd.DataFrame, role_module):
    report = unique_df.head(UNIQUE_REPORT_LIMIT).copy()

    def money(v):
        return "" if pd.isna(v) else f"€{float(v):.1f}M"

    rows_html = []

    for _, r in report.iterrows():
        tier_class = {
            "Starter target": "starter",
            "Value target": "value",
            "Role fit": "fit",
            "Watch": "watch",
        }.get(r["target_tier"], "watch")

        rows_html.append(
            "<tr>"
            f"<td><strong>{html_escape(r['player_name'])}</strong></td>"
            f"<td>{html_escape(r['natural_position'])}</td>"
            f"<td>{html_escape(r['club_name'])}</td>"
            f"<td>{html_escape(r['league_slug'])}</td>"
            f"<td class='num'>{'' if pd.isna(r['age']) else int(round(r['age']))}</td>"
            f"<td>{html_escape(r['height'])}</td>"
            f"<td class='num'>{int(round(r['minutes']))}</td>"
            f"<td class='num'>{money(r['market_value_eur_m'])}</td>"
            f"<td><span class='badge {tier_class}'>{html_escape(r['target_tier'])}</span></td>"
            f"<td><strong>{html_escape(r['best_role'])}</strong><br><span class='score'>{r['best_fit']:.1f}</span></td>"
            f"<td>{html_escape(tuple_list_text(r['role_fits']))}</td>"
            f"<td>{html_escape(tuple_list_text(r['fulfilled_roles']))}</td>"
            f"<td>{html_escape(r['best_role_strengths'])}</td>"
            f"<td>{html_escape(r['best_role_weaknesses'])}</td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Newcastle United - Jaissle Recruitment Shortlist 2025/26</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 28px;
        color: #17202a;
        background: #f5f7f9;
    }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #5d6d7e; margin-bottom: 18px; }}
    .note {{
        background: white;
        border-left: 4px solid #5d6d7e;
        padding: 12px 14px;
        margin-bottom: 18px;
        line-height: 1.45;
    }}
    .table-wrap {{
        overflow-x: auto;
        background: white;
        border: 1px solid #d5d8dc;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 12px;
        min-width: 1850px;
    }}
    th {{
        position: sticky;
        top: 0;
        background: #17202a;
        color: white;
        text-align: left;
        padding: 9px;
        white-space: nowrap;
    }}
    td {{
        padding: 8px 9px;
        border-bottom: 1px solid #e5e7e9;
        vertical-align: top;
        line-height: 1.35;
    }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .num {{ text-align: right; white-space: nowrap; }}
    .badge {{
        border-radius: 12px;
        padding: 3px 8px;
        font-size: 11px;
        white-space: nowrap;
        display: inline-block;
    }}
    .starter {{ background: #d5f5e3; }}
    .value {{ background: #d6eaf8; }}
    .fit {{ background: #fcf3cf; }}
    .watch {{ background: #ebedef; }}
    .score {{
        display: inline-block;
        margin-top: 3px;
        font-size: 16px;
        font-weight: bold;
    }}
    .legend {{
        margin: 12px 0 18px 0;
        color: #566573;
        font-size: 12px;
    }}
    @media print {{
        body {{ margin: 8mm; background: white; }}
        .table-wrap {{ overflow: visible; border: none; }}
        table {{ font-size: 8px; min-width: 0; }}
        th {{ position: static; }}
        td, th {{ padding: 4px; }}
    }}
</style>
</head>
<body>
<h1>Newcastle United - 2025/26 Jaissle Recruitment Shortlist</h1>
<div class="subtitle">Unique-player market screen across all tactical roles</div>
<div class="note">
<strong>Method:</strong> role fit uses the same league-role standardized metrics as the Newcastle historical analysis.
Transfermarkt natural position is used as a hard eligibility guard, so players are only scored for tactically plausible roles.
Each player appears once. <strong>Role Fits</strong> contains every compatible role and score; <strong>Roles >=55</strong> shows roles
that clear the model's useful-fit threshold.
</div>
<div class="legend">
Starter target = at least one role clears starter fit/price rules.
Value target = at least one role clears value fit/price rules.
Role fit = role score >= {FULFILLED_ROLE_FIT:.0f} without meeting the price/age target rule.
</div>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>Player</th><th>Natural Position</th><th>Club</th><th>League</th><th>Age</th><th>Ht</th>
<th>Min</th><th>Value</th><th>Target</th><th>Best Role</th><th>Role Fits</th><th>Roles >=55</th>
<th>Best-role strengths</th><th>Best-role weaknesses</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</div>
</body>
</html>
"""

    OUT_HTML.write_text(html_doc, encoding="utf-8")


def print_unique_targets(unique_df: pd.DataFrame):
    print()
    print("=" * 180)
    print("UNIQUE PLAYER SHORTLIST - ONE ROW PER PLAYER")
    print("=" * 180)

    display = unique_df.head(60).copy()

    display["Role Fits"] = display["role_fits"].map(tuple_list_text)
    display["Roles >=55"] = display["fulfilled_roles"].map(tuple_list_text)
    display["Value"] = display["market_value_eur_m"].map(format_money)

    display = display[
        [
            "player_name", "natural_position", "club_name", "league_slug",
            "age", "height", "Value", "target_tier", "best_role", "best_fit",
            "Role Fits", "Roles >=55",
        ]
    ].copy()

    display["age"] = display["age"].round(0).astype("Int64")
    display["best_fit"] = display["best_fit"].round(1)

    display = display.rename(columns={
        "player_name": "Player",
        "natural_position": "Position",
        "club_name": "Club",
        "league_slug": "League",
        "age": "Age",
        "height": "Ht",
        "target_tier": "Target",
        "best_role": "Best Role",
        "best_fit": "Best Fit",
    })

    print(display.to_string(index=False))



# ======================================================================================
# PER-ROLE EXPORTS
# ======================================================================================

def export_role_csvs(df: pd.DataFrame):
    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "player_name",
        "natural_position",
        "club_name",
        "league_slug",
        "age",
        "height",
        "minutes",
        "role_fit",
        "metric_coverage_pct",
        "market_value_eur_m",
        "fit_per_10m",
        "market_bucket",
        "value_target",
        "starter_target",
        "strengths",
        "weaknesses",
    ]

    for role_key in ROLE_ORDER:
        role_df = (
            df[df["role_key"].eq(role_key)]
            .sort_values(
                ["role_fit", "fit_per_10m", "market_value_eur_m"],
                ascending=[False, False, True],
            )
            .copy()
        )

        if role_df.empty:
            continue

        keep = [c for c in export_cols if c in role_df.columns]
        export = role_df[keep].copy()

        export = export.rename(columns={
            "player_name": "Player",
            "natural_position": "Natural Position",
            "club_name": "Club",
            "league_slug": "League",
            "age": "Age",
            "height": "Height",
            "minutes": "Minutes",
            "role_fit": "Role Fit",
            "metric_coverage_pct": "Coverage %",
            "market_value_eur_m": "Market Value €M",
            "fit_per_10m": "Fit per €10M",
            "market_bucket": "Market Bucket",
            "value_target": "Value Target",
            "starter_target": "Starter Target",
            "strengths": "Strengths",
            "weaknesses": "Weaknesses",
        })

        filename = ROLE_EXPORT_FILES[role_key]
        export.to_csv(SHORTLIST_DIR / filename, index=False)


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

    target_mask = (
        df["value_fit_price_ok"]
        & (
            (df["age"].notna() & (df["age"] <= PREFERRED_MAX_AGE))
            | df["age"].isna()
        )
    )

    targets = (
        df[target_mask]
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

    # Helpful schema diagnostic for Transfermarkt enrichment.
    tm_prepared = prepare_transfermarkt(tm)
    print("TRANSFERMARKT METADATA DETECTION")
    print("-" * 120)
    print("Age column:", tm_prepared.attrs.get("detected_age_col"))
    print("DOB column:", tm_prepared.attrs.get("detected_dob_col"))
    print("Club column:", tm_prepared.attrs.get("detected_club_col"))
    print("Value column:", tm_prepared.attrs.get("detected_value_col"))
    print("Height column:", tm_prepared.attrs.get("detected_height_col"))
    print("TM position column:", tm_prepared.attrs.get("detected_tm_position_col"))
    print("TM sub-position column:", tm_prepared.attrs.get("detected_tm_sub_position_col"))
    print("Rows with age:", int(tm_prepared["age"].notna().sum()), "/", len(tm_prepared))
    print("Rows with club:", int(tm_prepared["club_name"].notna().sum()), "/", len(tm_prepared))

    club_like_cols = [
        c for c in tm.columns
        if any(token in c.lower() for token in ["club", "team", "squad"])
    ]

    if club_like_cols:
        print("Club/team candidate columns:")
        for c in club_like_cols:
            nonnull = int(tm[c].notna().sum())
            nonblank = nonnull
            if tm[c].dtype == "object" or pd.api.types.is_string_dtype(tm[c]):
                nonblank = int(
                    tm[c].dropna().astype(str).str.strip().ne("").sum()
                )
            print(f"  {c:<35} non-null={nonnull:<5} non-blank={nonblank}")

    print()

    enriched_hist = merge_transfermarkt(hist, tm)

    scored = score_all_players(enriched_hist, role_module)
    final = add_market_layer(scored, enriched_hist)

    # Exclude current Newcastle players from the external market results where possible.
    refs = load_newcastle_refs()
    if not refs.empty:
        current_names = set(refs["player_name"].map(norm_text))
        final = final[~final["player_name"].map(norm_text).isin(current_names)].copy()

    # Name abbreviations can prevent exact current-squad matching (e.g. S. Botman
    # versus Sven Botman), so also remove anyone whose current Transfermarkt club
    # is Newcastle United.
    if "club_name" in final.columns:
        current_club_norm = final["club_name"].fillna("").map(norm_text)
        final = final[~current_club_norm.str.contains("newcastle united", regex=False)].copy()

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

    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_ALL, index=False)

    export_role_csvs(final)

    unique_df = build_unique_player_shortlist(final)

    unique_export = unique_df.copy()
    unique_export["role_fits"] = unique_export["role_fits"].map(tuple_list_text)
    unique_export["fulfilled_roles"] = unique_export["fulfilled_roles"].map(tuple_list_text)
    unique_export["starter_roles"] = unique_export["starter_roles"].map(tuple_list_text)
    unique_export["value_roles"] = unique_export["value_roles"].map(tuple_list_text)

    unique_export = unique_export.rename(columns={
        "player_name": "Player",
        "club_name": "Club",
        "league_slug": "League",
        "natural_position": "Natural Position",
        "natural_position_code": "Position Code",
        "age": "Age",
        "height": "Height",
        "minutes": "Minutes",
        "market_value_eur_m": "Market Value €M",
        "market_bucket": "Market Bucket",
        "best_role": "Best Role",
        "best_fit": "Best Fit",
        "best_fit_per_10m": "Best Fit per €10M",
        "target_tier": "Target Tier",
        "fulfilled_roles": "Roles >=55",
        "role_fits": "All Role Fits",
        "starter_roles": "Starter Roles",
        "value_roles": "Value Roles",
        "best_role_strengths": "Best Role Strengths",
        "best_role_weaknesses": "Best Role Weaknesses",
        "max_coverage_pct": "Max Coverage %",
        "role_count": "Role Count",
    })

    unique_export.to_csv(OUT_UNIQUE, index=False)

    save_html_report(unique_df, role_module)

    print_role_shortlist(final, refs, role_module)
    print_unique_targets(unique_df)

    print()
    print("Saved shortlist outputs:")
    print(SHORTLIST_DIR)
    print()
    print("Core files:")
    print(" ", OUT_ALL)
    print(" ", OUT_UNIQUE)
    print(" ", OUT_HTML)
    print()
    print("Per-role CSVs:")
    for role_key in ROLE_ORDER:
        print(" ", SHORTLIST_DIR / ROLE_EXPORT_FILES[role_key])


if __name__ == "__main__":
    main()
