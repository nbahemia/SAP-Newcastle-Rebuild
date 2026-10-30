from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed" / "bsd"

# Completed 2025/26 six-league pool. Must contain raw season/per90 metrics.
HISTORICAL = PROCESSED / "europe_2025_26_player_season_metrics.parquet"

# Current EPL data.
CURRENT_TEAM = PROCESSED / "player_team_season_metrics.parquet"
CURRENT_ENRICHED = PROCESSED / "player_season_metrics_enriched.parquet"
CURRENT_DIAGNOSTIC = PROCESSED / "newcastle_player_diagnostic.parquet"

OUT_SUMMARY = PROCESSED / "newcastle_complete_role_fit_summary.parquet"
OUT_LONG = PROCESSED / "newcastle_complete_role_fit_long.parquet"

HIST_MIN_MINUTES = 450
CURRENT_MIN_MINUTES_FOR_CONFIDENCE = 180
MIN_ROLE_COVERAGE = 0.60

# Standardized score:
# Role Fit = 50 + 10 * weighted z-score.
# 50 ~= role-cohort average; 60 ~= +1 SD; 40 ~= -1 SD.
ROLE_FIT_CENTER = 50.0
ROLE_FIT_SCALE = 10.0
ROLE_FIT_MIN = 20.0
ROLE_FIT_MAX = 80.0


# ======================================================================================
# LEAGUE CONTEXT — DISPLAY ONLY, NEVER MULTIPLIED INTO ROLE FIT
# ======================================================================================

UEFA_COEFFICIENTS_2025 = {
    "premier_league": 103.658,
    "serie_a": 92.124,
    "la_liga": 85.953,
    "bundesliga": 82.902,
    "ligue_1": 75.534,
    "eredivisie": 65.762,
}

UEFA_MAX = UEFA_COEFFICIENTS_2025["premier_league"]

LEAGUE_STRENGTH = {
    league: round(value / UEFA_MAX * 100.0, 1)
    for league, value in UEFA_COEFFICIENTS_2025.items()
}

# External supplement: Amar Dedić, Benfica 2025/26 (FotMob, user supplied).
# Portugal is outside the six-league comparison pool, so these metrics are
# shown as a tactical benchmark only and are not assigned a fabricated
# league-normalized Role Fit score.
DEDIC_2025_26_FOTMOB = {
    "minutes": 1903,
    "xg_per90": 0.05,
    "shots_per90": 0.71,
    "shots_on_target_per90": 0.09,
    "assists_per90": 0.19,
    "xa_per90": 0.17,
    "successful_passes_per90": 37.55,
    "pass_accuracy_pct": 84.6,
    "accurate_long_balls_per90": 1.14,
    "long_ball_accuracy_pct": 44.4,
    "line_breaking_passes_per90": 9.60,
    "chances_created_per90": 1.14,
    "big_chances_created_per90": 0.24,
    "successful_crosses_per90": 0.43,
    "cross_accuracy_pct": 23.1,
    "successful_dribbles_per90": 1.61,
    "dribble_success_pct": 53.1,
    "duels_won_per90": 4.78,
    "duel_win_pct": 51.5,
    "aerials_won_per90": 0.57,
    "aerial_win_pct": 46.2,
    "touches_per90": 68.29,
    "box_touches_per90": 2.88,
    "defensive_contributions_per90": 4.26,
    "tackles_per90": 1.66,
    "interceptions_per90": 0.76,
    "recoveries_per90": 4.07,
    "possession_won_final_third_per90": 0.66,
    "dribbled_past_per90": 0.66,
}



# ======================================================================================
# ROLE MODELS
#
# Weights reflect the Jaissle/Newcastle system we've been building:
# - CB: defend transitions + aerial/duel competence + secure buildup
# - WB: vertical progression + final-third access + defensive recovery
# - Controller: secure circulation + progression + receiving volume
# - B2B/Workhorse: defensive activity + carries + involvement + forward access
# - CAM: chance creation + final-third involvement + progression + pressing contribution
# - Winger: vertical threat + creation + shooting + carrying
# - Striker: Toney-like focal profile: box threat + physical duels + link play
#
# No percentile inputs are used.
# ======================================================================================

ROLE_MODELS = {
    "center_back": {
        "label": "High-Line Center Back",
        "cohort_positions": {"CB"},
        "weights": {
            "interceptions_per90": 0.16, "recoveries_per90": 0.15,
            "aerials_won_per90": 0.14, "duels_won_per90": 0.13,
            "tackles_won_per90": 0.10, "pass_accuracy_pct": 0.10,
            "total_long_balls_per90": 0.08, "successful_passes_per90": 0.08,
            "progressive_carries_per90": 0.06,
        },
        "display": [
            ("Interceptions","interceptions_per90"), ("Recoveries","recoveries_per90"),
            ("Aerials Won","aerials_won_per90"), ("Duels Won","duels_won_per90"),
            ("Tackles Won","tackles_won_per90"), ("Pass Acc","pass_accuracy_pct"),
            ("Long Balls","total_long_balls_per90"), ("Passes","successful_passes_per90"),
            ("Prog Carries","progressive_carries_per90"),
        ],
    },
    "vertical_wingback": {
        "label": "Vertical Full-Back / Wingback",
        "cohort_positions": {"FB/WB"},
        "weights": {
            "progressive_carries_per90":0.20, "accurate_opposition_half_passes_per90":0.15,
            "total_crosses_per90":0.12, "xa_per90":0.10, "recoveries_per90":0.12,
            "tackles_won_per90":0.08, "interceptions_per90":0.07,
            "duels_won_per90":0.06, "successful_passes_per90":0.05, "touches_per90":0.05,
        },
        "display": [
            ("Prog Carries","progressive_carries_per90"), ("Opp Half Passes","accurate_opposition_half_passes_per90"),
            ("Crosses","total_crosses_per90"), ("xA","xa_per90"), ("Recoveries","recoveries_per90"),
            ("Tackles Won","tackles_won_per90"), ("Interceptions","interceptions_per90"),
            ("Duels Won","duels_won_per90"), ("Passes","successful_passes_per90"), ("Touches","touches_per90"),
        ],
    },
    "holding_pivot": {
        "label": "Holding Pivot / Controller",
        "cohort_positions": {"DM/CM"},
        "weights": {
            "pass_accuracy_pct":0.15, "successful_passes_per90":0.14,
            "accurate_opposition_half_passes_per90":0.12, "progressive_carries_per90":0.10,
            "touches_per90":0.10, "recoveries_per90":0.12, "interceptions_per90":0.10,
            "duels_won_per90":0.08, "total_long_balls_per90":0.06, "xa_per90":0.03,
        },
        "display": [
            ("Pass Acc","pass_accuracy_pct"), ("Passes","successful_passes_per90"),
            ("Opp Half Passes","accurate_opposition_half_passes_per90"), ("Prog Carries","progressive_carries_per90"),
            ("Touches","touches_per90"), ("Recoveries","recoveries_per90"), ("Interceptions","interceptions_per90"),
            ("Duels Won","duels_won_per90"), ("Long Balls","total_long_balls_per90"), ("xA","xa_per90"),
        ],
    },
    "advanced_pivot": {
        "label": "Advanced Pivot / Workhorse",
        "cohort_positions": {"DM/CM","AM"},
        "weights": {
            "recoveries_per90":0.18, "duels_won_per90":0.15, "tackles_won_per90":0.12,
            "interceptions_per90":0.12, "progressive_carries_per90":0.14,
            "accurate_opposition_half_passes_per90":0.08, "touches_per90":0.06,
            "successful_passes_per90":0.05, "xg_per90":0.04, "shots_per90":0.03, "xa_per90":0.03,
        },
        "display": [
            ("Recoveries","recoveries_per90"), ("Duels Won","duels_won_per90"),
            ("Tackles Won","tackles_won_per90"), ("Interceptions","interceptions_per90"),
            ("Prog Carries","progressive_carries_per90"), ("Opp Half Passes","accurate_opposition_half_passes_per90"),
            ("Touches","touches_per90"), ("Passes","successful_passes_per90"), ("xG","xg_per90"), ("Shots","shots_per90"),
        ],
    },
    "creative_cam": {
        "label": "Creative 10",
        "cohort_positions": {"AM","DM/CM","W"},
        "weights": {
            "xa_per90":0.22, "key_passes_per90":0.20, "accurate_opposition_half_passes_per90":0.15,
            "progressive_carries_per90":0.12, "successful_passes_per90":0.08, "touches_per90":0.08,
            "recoveries_per90":0.05, "interceptions_per90":0.03, "xg_per90":0.04, "shots_per90":0.03,
        },
        "display": [
            ("xA","xa_per90"), ("Key Passes","key_passes_per90"), ("Opp Half Passes","accurate_opposition_half_passes_per90"),
            ("Prog Carries","progressive_carries_per90"), ("Passes","successful_passes_per90"), ("Touches","touches_per90"),
            ("Recoveries","recoveries_per90"), ("Interceptions","interceptions_per90"), ("xG","xg_per90"), ("Shots","shots_per90"),
        ],
    },
    "vertical_winger": {
        "label": "Vertical Transition Winger",
        "cohort_positions": {"W","AM"},
        "weights": {
            "progressive_carries_per90":0.22, "xg_per90":0.15, "shots_per90":0.12,
            "shots_on_target_per90":0.08, "accurate_opposition_half_passes_per90":0.10,
            "xa_per90":0.07, "key_passes_per90":0.05, "total_crosses_per90":0.05,
            "recoveries_per90":0.07, "duels_won_per90":0.05, "touches_per90":0.04,
        },
        "display": [
            ("Prog Carries","progressive_carries_per90"), ("xG","xg_per90"), ("Shots","shots_per90"),
            ("Shots on Target","shots_on_target_per90"), ("Opp Half Passes","accurate_opposition_half_passes_per90"),
            ("xA","xa_per90"), ("Key Passes","key_passes_per90"), ("Crosses","total_crosses_per90"),
            ("Recoveries","recoveries_per90"), ("Duels Won","duels_won_per90"),
        ],
    },
    "inside_creator": {
        "label": "Inside Creator",
        "cohort_positions": {"W","AM"},
        "weights": {
            "xa_per90":0.21, "key_passes_per90":0.18, "progressive_carries_per90":0.16,
            "accurate_opposition_half_passes_per90":0.14, "successful_passes_per90":0.08,
            "touches_per90":0.07, "xg_per90":0.07, "shots_per90":0.04,
            "recoveries_per90":0.03, "duels_won_per90":0.02,
        },
        "display": [
            ("xA","xa_per90"), ("Key Passes","key_passes_per90"), ("Prog Carries","progressive_carries_per90"),
            ("Opp Half Passes","accurate_opposition_half_passes_per90"), ("Passes","successful_passes_per90"),
            ("Touches","touches_per90"), ("xG","xg_per90"), ("Shots","shots_per90"), ("Recoveries","recoveries_per90"),
        ],
    },
    "focal_striker": {
        "label": "Pressing Focal Striker",
        "cohort_positions": {"ST"},
        "weights": {
            "xg_per90":0.17, "shots_per90":0.10, "shots_on_target_per90":0.08,
            "aerials_won_per90":0.15, "duels_won_per90":0.13, "successful_passes_per90":0.09,
            "touches_per90":0.08, "key_passes_per90":0.06, "xa_per90":0.05,
            "progressive_carries_per90":0.04, "recoveries_per90":0.05,
        },
        "display": [
            ("xG","xg_per90"), ("Shots","shots_per90"), ("Shots on Target","shots_on_target_per90"),
            ("Aerials Won","aerials_won_per90"), ("Duels Won","duels_won_per90"),
            ("Passes","successful_passes_per90"), ("Touches","touches_per90"),
            ("Key Passes","key_passes_per90"), ("xA","xa_per90"), ("Recoveries","recoveries_per90"),
        ],
    },
    "second_striker": {
        "label": "Second Striker / Shadow 10",
        "cohort_positions": {"ST","AM","W"},
        "weights": {
            "xg_per90":0.20, "xa_per90":0.14, "shots_per90":0.13, "key_passes_per90":0.11,
            "progressive_carries_per90":0.10, "touches_per90":0.09, "successful_passes_per90":0.08,
            "duels_won_per90":0.06, "recoveries_per90":0.05, "accurate_opposition_half_passes_per90":0.04,
        },
        "display": [
            ("xG","xg_per90"), ("xA","xa_per90"), ("Shots","shots_per90"), ("Key Passes","key_passes_per90"),
            ("Prog Carries","progressive_carries_per90"), ("Touches","touches_per90"),
            ("Passes","successful_passes_per90"), ("Duels Won","duels_won_per90"), ("Recoveries","recoveries_per90"),
        ],
    },
}


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

# ======================================================================================
# HISTORICAL IDENTITY OVERRIDES
# ======================================================================================

HISTORICAL_OVERRIDES = {
    # IMPORTANT: force our Bazoumana Touré to his Hoffenheim/Bundesliga winger row.
    "bazoumana toure": {
        "league_slug": "bundesliga",
        "position_group": "W",
        "min_minutes": 1500,
        "preferred_name_terms": ["bazoumana", "toure"],
    },

    "matias fernandez pardo": {
        "league_slug": "ligue_1",
        "position_group": "W",
        "min_minutes": 1500,
        "preferred_name_terms": ["fernandez", "pardo"],
    },

    "nick woltemade": {
        "league_slug": "premier_league",
        "position_group": "ST",
        "min_minutes": 1500,
        "preferred_name_terms": ["woltemade"],
    },

    "tino livramento": {
        "league_slug": "premier_league",
        "position_group": "FB/WB",
        "min_minutes": 450,
        "preferred_name_terms": ["livramento"],
    },

    "aladji bamba": {
        "league_slug": "ligue_1",
        "position_group": "DM/CM",
        "min_minutes": 450,
        "preferred_name_terms": ["bamba"],
    },

    "jacob ramsey": {
        "league_slug": "premier_league",
        "position_group": "DM/CM",
        "min_minutes": 450,
        "preferred_name_terms": ["ramsey"],
    },

    # Dedić played in Portugal in 2025/26, outside this six-league historical pool.
    "amar dedic": {
        "outside_pool": True,
        "reason": "Benfica/Portugal 2025/26 is outside the six-league model; see external FotMob full-back benchmark below.",
    },
}


# ======================================================================================
# HELPERS
# ======================================================================================

def norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text.lower(),
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def infer_slug_from_league_name(value):
    s = norm_text(value)

    mapping = {
        "premier league": "premier_league",
        "bundesliga": "bundesliga",
        "la liga": "la_liga",
        "serie a": "serie_a",
        "ligue 1": "ligue_1",
        "eredivisie": "eredivisie",
    }

    for text, slug in mapping.items():
        if text in s:
            return slug

    return None


def ensure_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize schema and reconstruct per-90 metrics from season totals when needed.
    This is important because older Europe parquet builds used slightly different
    BSD field names and some per90 columns may be absent even when raw totals exist.
    """
    out = df.copy()

    aliases = {
        "name": "player_name",
        "mins": "minutes",
        "role": "position_group",
        "accurate_passes": "successful_passes",
        "accurate_pass": "successful_passes",
        "total_pass": "total_passes",
        "key_pass": "key_passes",
        "expected_goals": "xg",
        "expected_assists": "xa",
        "total_shots": "shots",
        "total_cross": "total_crosses",
        "accurate_cross": "accurate_crosses",
        "ball_recovery": "recoveries",
        "interception": "interceptions",
        "won_tackle": "tackles_won",
        "total_tackle": "total_tackles",
        "duel_won": "duels_won",
        "aerial_won": "aerials_won",
        "progressive_ball_carries_count": "progressive_carries",
        "progressive_carries_count": "progressive_carries",
        "ball_carries_progressive": "progressive_carries",
        "total_progressive_ball_carries_distance": "progressive_carry_distance",
        "progressive_ball_carries_distance": "progressive_carry_distance",
        "progressive_carry_distance": "progressive_carry_distance",
        "opposition_half_accurate_passes": "accurate_opposition_half_passes",
        "accurate_passes_opposition_half": "accurate_opposition_half_passes",
        "accurate_opposition_half_pass": "accurate_opposition_half_passes",
    }

    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "league_slug" not in out.columns and "league_name" in out.columns:
        out["league_slug"] = out["league_name"].apply(infer_slug_from_league_name)

    minutes = (
        pd.to_numeric(out["minutes"], errors="coerce")
        if "minutes" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    # Reconstruct common per90s from raw season totals if the per90 column
    # is missing OR is entirely null.
    total_to_per90 = {
        "xg": "xg_per90",
        "xa": "xa_per90",
        "shots": "shots_per90",
        "shots_on_target": "shots_on_target_per90",
        "key_passes": "key_passes_per90",
        "successful_passes": "successful_passes_per90",
        "total_passes": "total_passes_per90",
        "accurate_opposition_half_passes": "accurate_opposition_half_passes_per90",
        "total_long_balls": "total_long_balls_per90",
        "accurate_long_balls": "accurate_long_balls_per90",
        "total_crosses": "total_crosses_per90",
        "accurate_crosses": "accurate_crosses_per90",
        "touches": "touches_per90",
        "progressive_carries": "progressive_carries_per90",
        "progressive_carry_distance": "progressive_carry_distance_per90",
        "recoveries": "recoveries_per90",
        "interceptions": "interceptions_per90",
        "duels_won": "duels_won_per90",
        "aerials_won": "aerials_won_per90",
        "tackles_won": "tackles_won_per90",
    }

    # Recover progression totals from any alternate historical column names.
    fallback_candidates = {
        "progressive_carries": [
            "progressive_ball_carries_count",
            "progressive_carries_count",
            "ball_carries_progressive",
        ],
        "progressive_carry_distance": [
            "total_progressive_ball_carries_distance",
            "progressive_ball_carries_distance",
        ],
        "accurate_opposition_half_passes": [
            "accurate_opposition_half_pass",
            "opposition_half_accurate_passes",
            "accurate_passes_opposition_half",
        ],
    }

    for canonical, candidates in fallback_candidates.items():
        if canonical in out.columns and pd.to_numeric(
            out[canonical], errors="coerce"
        ).notna().sum() > 0:
            continue

        for candidate in candidates:
            if candidate in out.columns:
                out[canonical] = pd.to_numeric(
                    out[candidate], errors="coerce"
                )
                break

    valid_minutes = minutes.where(minutes > 0)

    for total_col, per90_col in total_to_per90.items():
        if total_col not in out.columns:
            continue

        needs_build = (
            per90_col not in out.columns
            or pd.to_numeric(out[per90_col], errors="coerce").notna().sum() == 0
        )

        if needs_build:
            out[per90_col] = (
                pd.to_numeric(out[total_col], errors="coerce")
                / valid_minutes
                * 90.0
            )

    # Pass accuracy.
    if (
        "successful_passes" in out.columns
        and "total_passes" in out.columns
        and (
            "pass_accuracy_pct" not in out.columns
            or pd.to_numeric(out["pass_accuracy_pct"], errors="coerce").notna().sum() == 0
        )
    ):
        denom = pd.to_numeric(out["total_passes"], errors="coerce").replace(0, np.nan)
        out["pass_accuracy_pct"] = (
            pd.to_numeric(out["successful_passes"], errors="coerce")
            / denom
            * 100.0
        )

    return out


def robust_z(value, cohort: pd.Series) -> float:
    """
    Raw-metric standardization with winsorization.
    No percentile/rank inputs.

    We clip the cohort at the 5th/95th percentiles only to reduce extreme
    small-sample distortion, then compute a conventional z-score.
    """
    value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    s = pd.to_numeric(
        cohort,
        errors="coerce",
    ).dropna()

    if pd.isna(value) or len(s) < 8:
        return math.nan

    lower = s.quantile(0.05)
    upper = s.quantile(0.95)

    clipped = s.clip(
        lower=lower,
        upper=upper,
    )

    mean = clipped.mean()
    std = clipped.std(ddof=0)

    if pd.isna(std) or std <= 1e-9:
        return math.nan

    value_clipped = min(
        max(value, lower),
        upper,
    )

    return float(
        (value_clipped - mean) / std
    )


def z_to_fit(z: float) -> float:
    if pd.isna(z):
        return math.nan

    return float(
        np.clip(
            ROLE_FIT_CENTER + ROLE_FIT_SCALE * z,
            ROLE_FIT_MIN,
            ROLE_FIT_MAX,
        )
    )


def name_match_score(a, b) -> float:
    aa = norm_text(a)
    bb = norm_text(b)

    if not aa or not bb:
        return 0.0

    if aa == bb:
        return 100.0

    at = aa.split()
    bt = bb.split()

    score = 0.0

    if at[-1] == bt[-1]:
        score += 60.0

    if at[0][0] == bt[0][0]:
        score += 25.0

    overlap = len(set(at) & set(bt)) / max(
        1,
        len(set(at) | set(bt)),
    )

    score += 15.0 * overlap

    return score


def find_historical_player(
    current_name,
    historical: pd.DataFrame,
):
    normalized = norm_text(
        current_name
    )

    override = HISTORICAL_OVERRIDES.get(
        normalized
    )

    if override and override.get(
        "outside_pool"
    ):
        return None, override.get(
            "reason"
        )

    candidates = historical.copy()

    if override:
        if override.get("league_slug"):
            candidates = candidates[
                candidates["league_slug"].eq(
                    override["league_slug"]
                )
            ]

        if override.get("position_group"):
            candidates = candidates[
                candidates["position_group"].eq(
                    override["position_group"]
                )
            ]

        if override.get("min_minutes"):
            candidates = candidates[
                pd.to_numeric(
                    candidates["minutes"],
                    errors="coerce",
                ).fillna(0)
                >= override["min_minutes"]
            ]

        terms = [
            norm_text(x)
            for x in override.get(
                "preferred_name_terms",
                []
            )
        ]

        if terms:
            normalized_names = (
                candidates["player_name"]
                .astype(str)
                .map(norm_text)
            )

            mask = normalized_names.apply(
                lambda name: all(
                    term in name
                    for term in terms
                )
            )

            preferred = candidates[
                mask
            ].copy()

            if not preferred.empty:
                candidates = preferred

    if candidates.empty:
        return None, "No historical candidate after identity filters."

    candidates = candidates.copy()

    candidates["_name_score"] = (
        candidates["player_name"]
        .apply(
            lambda x: name_match_score(
                current_name,
                x,
            )
        )
    )

    # Explicit identity overrides are authoritative after league/position/minute filters.
    threshold = 0.0 if override else 75.0

    candidates = candidates.sort_values(
        [
            "_name_score",
            "minutes",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = candidates.iloc[0]

    if best["_name_score"] < threshold:
        return None, (
            "Historical identity match below confidence threshold."
        )

    return best, None


def role_fit_for_row(
    row: pd.Series,
    source: pd.DataFrame,
    role_key: str,
    league_slug: str,
):
    model = ROLE_MODELS[
        role_key
    ]

    cohort = source[
        source["league_slug"].eq(
            league_slug
        )
        & source["position_group"].isin(
            model["cohort_positions"]
        )
        & (
            pd.to_numeric(
                source["minutes"],
                errors="coerce",
            ).fillna(0)
            >= HIST_MIN_MINUTES
        )
    ].copy()

    weighted_z = 0.0
    used_weight = 0.0
    metric_detail = {}

    for metric, weight in model[
        "weights"
    ].items():

        if metric not in source.columns:
            metric_detail[
                metric
            ] = math.nan
            continue

        z = robust_z(
            row.get(metric),
            cohort[metric],
        )

        metric_detail[
            metric
        ] = z

        if not pd.isna(z):
            weighted_z += z * weight
            used_weight += weight

    total_weight = sum(
        model["weights"].values()
    )

    coverage = (
        used_weight / total_weight
        if total_weight > 0
        else 0.0
    )

    if used_weight <= 0:
        return math.nan, coverage, metric_detail

    normalized_z = (
        weighted_z / used_weight
    )

    return (
        z_to_fit(normalized_z),
        coverage,
        metric_detail,
    )


def get_complete_current_newcastle(
    diagnostic: pd.DataFrame,
    team_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use the Newcastle-specific diagnostic table as the authoritative readable roster.
    Do NOT assume player_team_season_metrics contains player_name.

    We retain all diagnostic players and then, when possible, enrich from the team-level
    table by player_id.
    """
    diagnostic = ensure_common_columns(
        diagnostic
    )
    team_metrics = ensure_common_columns(
        team_metrics
    )

    if "player_name" not in diagnostic.columns:
        raise RuntimeError(
            "newcastle_player_diagnostic.parquet does not contain player_name."
        )

    squad = diagnostic.copy()

    if "player_id" not in squad.columns:
        # Historical matching is name-based, so player_id is helpful but not mandatory.
        squad["player_id"] = pd.NA

    if "minutes" not in squad.columns:
        squad["minutes"] = pd.NA

    if "position_group" not in squad.columns:
        squad["position_group"] = pd.NA

    # If team-level metrics include player_id, enrich the diagnostic roster from them.
    if (
        "player_id" in squad.columns
        and "player_id" in team_metrics.columns
    ):
        team_unique = (
            team_metrics
            .sort_values(
                "minutes",
                ascending=False,
                na_position="last",
            )
            .drop_duplicates(
                "player_id"
            )
        )

        add_cols = [
            c for c in team_unique.columns
            if c not in squad.columns
            and c != "player_name"
        ]

        if add_cols:
            squad = squad.merge(
                team_unique[
                    ["player_id"] + add_cols
                ],
                on="player_id",
                how="left",
            )

    return squad


def merge_current_raw(
    squad: pd.DataFrame,
    current_raw: pd.DataFrame,
) -> pd.DataFrame:
    squad = ensure_common_columns(
        squad
    )
    raw = ensure_common_columns(
        current_raw
    )

    # Prefer player_id. Avoid duplicate columns from the team table.
    if (
        "player_id" in squad.columns
        and "player_id" in raw.columns
    ):
        raw_cols = [
            c for c in raw.columns
            if c not in {
                "player_name",
                "minutes",
                "position_group",
                "team_id",
            }
        ]

        raw_keep = (
            ["player_id"]
            + [
                c for c in raw_cols
                if c != "player_id"
            ]
        )

        raw_one = (
            raw[raw_keep]
            .drop_duplicates(
                "player_id"
            )
        )

        return squad.merge(
            raw_one,
            on="player_id",
            how="left",
            suffixes=(
                "",
                "_raw",
            ),
        )

    return squad


def plausible_roles(position_group):
    """Jaissle 4-2-3-1 role families."""
    pos = str(position_group)
    mapping = {
        "CB": ["center_back"],
        "FB/WB": ["vertical_wingback"],
        "DM/CM": ["holding_pivot", "advanced_pivot", "creative_cam"],
        "AM": ["creative_cam", "advanced_pivot", "vertical_winger", "inside_creator", "second_striker"],
        "W": ["vertical_winger", "inside_creator", "creative_cam", "second_striker", "focal_striker"],
        "ST": ["focal_striker", "second_striker", "creative_cam", "vertical_winger"],
    }
    return mapping.get(pos, [])


def minimum_role_coverage(role_key: str) -> float:
    # WB and progression-heavy roles can be evaluated at slightly lower coverage
    # if historical BSD did not expose every carry metric.
    if role_key == "vertical_wingback":
        return 0.45
    if role_key in {
        "holding_pivot",
        "advanced_pivot",
        "creative_cam",
        "vertical_winger",
        "inside_creator",
        "second_striker",
    }:
        return 0.55
    return MIN_ROLE_COVERAGE


def build_historical_summary(
    current_squad: pd.DataFrame,
    historical: pd.DataFrame,
):
    rows = []
    long_rows = []

    for _, current in current_squad.iterrows():
        player_name = current.get(
            "player_name"
        )

        hist, note = find_historical_player(
            player_name,
            historical,
        )

        base = {
            "player_name": player_name,
            "current_minutes": current.get(
                "minutes"
            ),
            "current_position": current.get(
                "position_group"
            ),
            "historical_found": hist is not None,
            "historical_note": note,
        }

        if hist is None:
            rows.append(
                base
            )
            continue

        league_slug = hist.get(
            "league_slug"
        )

        base.update({
            "historical_player_name": hist.get(
                "player_name"
            ),
            "historical_league": hist.get(
                "league_name"
            ),
            "historical_league_slug": league_slug,
            "league_strength_index": LEAGUE_STRENGTH.get(
                league_slug
            ),
            "historical_minutes": hist.get(
                "minutes"
            ),
            "historical_position": hist.get(
                "position_group"
            ),
        })

        role_results = []

        for role_key in ROLE_ORDER:
            fit, coverage, detail = role_fit_for_row(
                hist,
                historical,
                role_key,
                league_slug,
            )

            plausible = (
                role_key
                in plausible_roles(
                    hist.get(
                        "position_group"
                    )
                )
            )

            long_row = {
                "player_name": player_name,
                "historical_player_name": hist.get(
                    "player_name"
                ),
                "historical_league": hist.get(
                    "league_name"
                ),
                "historical_league_slug": league_slug,
                "league_strength_index": LEAGUE_STRENGTH.get(
                    league_slug
                ),
                "historical_minutes": hist.get(
                    "minutes"
                ),
                "historical_position": hist.get(
                    "position_group"
                ),
                "role_key": role_key,
                "role": ROLE_MODELS[
                    role_key
                ]["label"],
                "role_fit": fit,
                "metric_coverage": coverage * 100.0,
                "plausible_role": plausible,
            }

            for metric, z in detail.items():
                long_row[
                    f"{metric}_z"
                ] = z

            long_rows.append(
                long_row
            )

            role_min_coverage = minimum_role_coverage(role_key)

            if (
                plausible
                and not pd.isna(fit)
                and coverage >= role_min_coverage
            ):
                role_results.append(
                    (
                        role_key,
                        fit,
                        coverage,
                    )
                )

        role_results.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        if role_results:
            first = role_results[
                0
            ]
            base[
                "best_2025_26_role"
            ] = ROLE_MODELS[
                first[0]
            ]["label"]
            base[
                "best_2025_26_fit"
            ] = first[1]

        if len(role_results) > 1:
            second = role_results[
                1
            ]
            base[
                "second_2025_26_role"
            ] = ROLE_MODELS[
                second[0]
            ]["label"]
            base[
                "second_2025_26_fit"
            ] = second[1]

        rows.append(
            base
        )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            long_rows
        ),
    )


def role_strengths_weaknesses(long_df, player_name, role_key, top_n=3):
    rows = long_df[
        long_df["player_name"].astype(str).eq(str(player_name))
        & long_df["role_key"].eq(role_key)
    ]
    if rows.empty:
        return "", ""

    row = rows.iloc[0]
    scored = []
    for label, metric in ROLE_MODELS[role_key]["display"]:
        col = f"{metric}_z"
        if col in row.index and pd.notna(row.get(col)):
            scored.append((label, float(row[col])))

    strengths = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    weaknesses = sorted(scored, key=lambda x: x[1])[:top_n]

    return (
        "; ".join(f"{label} {z:+.2f}z" for label, z in strengths),
        "; ".join(f"{label} {z:+.2f}z" for label, z in weaknesses),
    )


def print_summary(
    summary: pd.DataFrame,
    long_df: pd.DataFrame,
):
    summary = summary.copy()

    strengths_col = []
    weaknesses_col = []

    label_to_key = {model["label"]: key for key, model in ROLE_MODELS.items()}

    for _, row in summary.iterrows():
        role_key = label_to_key.get(row.get("best_2025_26_role"))
        if role_key is None:
            strengths_col.append("")
            weaknesses_col.append("")
            continue

        strengths, weaknesses = role_strengths_weaknesses(
            long_df, row.get("player_name"), role_key
        )
        strengths_col.append(strengths)
        weaknesses_col.append(weaknesses)

    summary["best_role_strengths"] = strengths_col
    summary["best_role_weaknesses"] = weaknesses_col
    cols = [
        "player_name",
        "current_minutes",
        "current_position",
        "historical_found",
        "historical_league",
        "league_strength_index",
        "historical_minutes",
        "historical_position",
        "best_2025_26_role",
        "best_2025_26_fit",
        "second_2025_26_role",
        "second_2025_26_fit",
        "best_role_strengths",
        "best_role_weaknesses",
        "historical_note",
    ]

    cols = [
        c for c in cols
        if c in summary.columns
    ]

    shown = summary[
        cols
    ].copy()

    shown = shown.rename(
        columns={
            "player_name": "Newcastle Player",
            "current_minutes": "26/27 Min",
            "current_position": "26/27 Role",
            "historical_found": "25/26 Found",
            "historical_league": "25/26 League",
            "league_strength_index": "League Strength",
            "historical_minutes": "25/26 Min",
            "historical_position": "25/26 Pos",
            "best_2025_26_role": "Best 25/26 Role",
            "best_2025_26_fit": "Best Fit",
            "second_2025_26_role": "2nd Role",
            "second_2025_26_fit": "2nd Fit",
            "best_role_strengths": "Best Role Strengths",
            "best_role_weaknesses": "Best Role Weaknesses",
            "historical_note": "Note",
        }
    )

    for col in [
        "Best Fit",
        "2nd Fit",
        "League Strength",
    ]:
        if col in shown.columns:
            shown[col] = pd.to_numeric(
                shown[col],
                errors="coerce",
            ).round(1)

    print()
    print(
        "ALL NEWCASTLE OUTFIELD PLAYERS — 2025/26 ROLE FIT"
    )
    print(
        "-" * 200
    )
    print(
        shown.to_string(
            index=False
        )
    )


def print_role_tables(
    long_df: pd.DataFrame,
):
    for role_key in ROLE_ORDER:
        role_label = ROLE_MODELS[
            role_key
        ]["label"]

        role_rows = long_df[
            long_df["role_key"].eq(
                role_key
            )
            & long_df[
                "plausible_role"
            ].eq(True)
            & (
                long_df["metric_coverage"]
                >= minimum_role_coverage(role_key) * 100.0
            )
        ].copy()

        role_rows = role_rows.sort_values(
            "role_fit",
            ascending=False,
        )

        cols = [
            "player_name",
            "historical_league",
            "league_strength_index",
            "historical_minutes",
            "historical_position",
            "role_fit",
            "metric_coverage",
        ]

        # Print z-score context for the most important metrics.
        metric_labels = []

        for label, metric in ROLE_MODELS[
            role_key
        ]["display"]:

            col = f"{metric}_z"

            if col in role_rows.columns:
                cols.append(
                    col
                )
                metric_labels.append(
                    (
                        col,
                        f"{label} Z",
                    )
                )

        shown = role_rows[
            cols
        ].head(
            15
        ).copy()

        rename = {
            "player_name": "Player",
            "historical_league": "25/26 League",
            "league_strength_index": "League Strength",
            "historical_minutes": "25/26 Min",
            "historical_position": "25/26 Pos",
            "role_fit": "Role Fit",
            "metric_coverage": "Coverage %",
        }

        rename.update(
            dict(
                metric_labels
            )
        )

        shown = shown.rename(
            columns=rename
        )

        for col in shown.columns:
            if (
                col.endswith(" Z")
                or col
                in {
                    "Role Fit",
                    "Coverage %",
                    "League Strength",
                }
            ):
                shown[col] = pd.to_numeric(
                    shown[col],
                    errors="coerce",
                ).round(2)

        print()
        print(
            "=" * 200
        )
        print(
            role_label.upper().center(
                200
            )
        )
        print(
            "=" * 200
        )
        print(
            "Role Fit uses raw per-90 metrics standardized within the player's 2025/26 league role cohort. "
            "Z = standard deviations above/below cohort average; no percentile inputs."
        )
        print(
            shown.to_string(
                index=False
            )
        )



def print_dedic_external_benchmark():
    d = DEDIC_2025_26_FOTMOB

    print()
    print("=" * 200)
    print("AMAR DEDIĆ — EXTERNAL 2025/26 BENFICA FULL-BACK BENCHMARK")
    print("=" * 200)
    print(
        "Portugal is outside the six-league historical model, so these FotMob values "
        "are shown qualitatively and are not converted into a comparable Role Fit score."
    )
    print()

    rows = [
        ("Minutes", d["minutes"]),
        ("Successful passes /90", d["successful_passes_per90"]),
        ("Pass accuracy", f'{d["pass_accuracy_pct"]:.1f}%'),
        ("Line-breaking passes /90", d["line_breaking_passes_per90"]),
        ("Accurate long balls /90", d["accurate_long_balls_per90"]),
        ("Long-ball accuracy", f'{d["long_ball_accuracy_pct"]:.1f}%'),
        ("Successful dribbles /90", d["successful_dribbles_per90"]),
        ("Dribble success", f'{d["dribble_success_pct"]:.1f}%'),
        ("xA /90", d["xa_per90"]),
        ("Chances created /90", d["chances_created_per90"]),
        ("Successful crosses /90", d["successful_crosses_per90"]),
        ("Cross accuracy", f'{d["cross_accuracy_pct"]:.1f}%'),
        ("Touches /90", d["touches_per90"]),
        ("Touches in box /90", d["box_touches_per90"]),
        ("Recoveries /90", d["recoveries_per90"]),
        ("Tackles /90", d["tackles_per90"]),
        ("Interceptions /90", d["interceptions_per90"]),
        ("Duels won /90", d["duels_won_per90"]),
        ("Duel win rate", f'{d["duel_win_pct"]:.1f}%'),
        ("Possession won final 3rd /90", d["possession_won_final_third_per90"]),
        ("Dribbled past /90", d["dribbled_past_per90"]),
    ]

    for label, value in rows:
        print(f"{label:<34} {value}")

    print()
    print("TACTICAL READ")
    print("-" * 120)
    print(
        "Strong Jaissle fit: high passing involvement, line-breaking distribution, "
        "ball-carrying, useful chance creation and active defensive recovery."
    )
    print(
        "Main caution: crossing efficiency and duel percentages are solid rather than dominant; "
        "he profiles more as a progressive/build-up full-back than a shutdown defender."
    )
    print(
        "Squad implication: Hall + Dedić can be treated as the likely starting full-back pair, "
        "with Livramento as rotation/competition rather than an urgent recruitment need."
    )


def print_target_checks(
    summary: pd.DataFrame,
    long_df: pd.DataFrame,
):
    targets = [
        "Matías Fernández-Pardo",
        "Bazoumana Touré",
        "Amar Dedić",
        "Tino Livramento",
        "Aladji Bamba",
        "Jacob Ramsey",
        "Harvey Barnes",
        "Nick Woltemade",
        "Yoane Wissa",
        "Sean Steur",
    ]

    print()
    print(
        "=" * 200
    )
    print(
        "TARGETED ROLE CHECKS".center(
            200
        )
    )
    print(
        "=" * 200
    )

    for target in targets:
        target_norm = norm_text(
            target
        )

        s = summary[
            summary["player_name"]
            .astype(str)
            .map(norm_text)
            .eq(
                target_norm
            )
        ]

        print()
        print(
            target
        )

        if s.empty:
            print(
                "  Not found in inferred current Newcastle squad."
            )
            continue

        row = s.iloc[
            0
        ]

        if not bool(
            row.get(
                "historical_found",
                False,
            )
        ):
            print(
                f"  Historical profile unavailable: "
                f"{row.get('historical_note')}"
            )
            continue

        roles = long_df[
            long_df["player_name"]
            .astype(str)
            .map(norm_text)
            .eq(
                target_norm
            )
            & long_df[
                "plausible_role"
            ].eq(True)
            & (
                long_df["metric_coverage"]
                >= long_df["role_key"].map(
                    lambda rk: minimum_role_coverage(rk) * 100.0
                )
            )
        ].sort_values(
            "role_fit",
            ascending=False,
        )

        text = ", ".join(
            f"{r['role']}: {r['role_fit']:.1f}"
            for _, r in roles.iterrows()
            if pd.notna(
                r["role_fit"]
            )
        )

        print(
            f"  2025/26 match: {row.get('historical_player_name')} | "
            f"{row.get('historical_league')} | "
            f"{row.get('historical_minutes'):.0f} min | "
            f"listed {row.get('historical_position')}"
        )
        print(
            f"  Plausible role fits: {text}"
        )


def main():
    for path in [
        HISTORICAL,
        CURRENT_TEAM,
        CURRENT_DIAGNOSTIC,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    historical = ensure_common_columns(
        pd.read_parquet(
            HISTORICAL
        )
    )

    team_metrics = ensure_common_columns(
        pd.read_parquet(
            CURRENT_TEAM
        )
    )

    diagnostic = ensure_common_columns(
        pd.read_parquet(
            CURRENT_DIAGNOSTIC
        )
    )

    if CURRENT_ENRICHED.exists():
        current_raw = ensure_common_columns(
            pd.read_parquet(
                CURRENT_ENRICHED
            )
        )
    else:
        current_raw = team_metrics.copy()

    current_squad = get_complete_current_newcastle(
        diagnostic,
        team_metrics,
    )

    current_squad = merge_current_raw(
        current_squad,
        current_raw,
    )

    # Explicit current-squad additions that may be omitted from the diagnostic
    # because of minutes/eligibility thresholds.
    explicit_additions = [
        {"player_name": "Nick Woltemade", "position_group": "ST", "minutes": pd.NA},
        {"player_name": "Tino Livramento", "position_group": "FB/WB", "minutes": pd.NA},
        {"player_name": "Aladji Bamba", "position_group": "DM/CM", "minutes": pd.NA},
        {"player_name": "Jacob Ramsey", "position_group": "DM/CM", "minutes": pd.NA},
    ]

    existing = set(
        current_squad["player_name"]
        .astype(str)
        .map(norm_text)
    )

    additions = []

    for player in explicit_additions:
        if norm_text(player["player_name"]) not in existing:
            additions.append(player)

    if additions:
        add_df = pd.DataFrame(additions)

        all_cols = list(dict.fromkeys(
            list(current_squad.columns) + list(add_df.columns)
        ))

        current_squad = current_squad.reindex(columns=all_cols)
        add_df = add_df.reindex(columns=all_cols)

        current_squad = pd.concat(
            [current_squad, add_df],
            ignore_index=True,
        )

    # GK intentionally excluded from replacement analysis.
    if "position_group" in current_squad.columns:
        outfield = current_squad[
            ~current_squad[
                "position_group"
            ].astype(str).eq("GK")
        ].copy()
    else:
        outfield = current_squad.copy()

    summary, long_df = build_historical_summary(
        outfield,
        historical,
    )

    summary.to_parquet(
        OUT_SUMMARY,
        index=False,
    )

    long_df.to_parquet(
        OUT_LONG,
        index=False,
    )

    print(
        "=" * 200
    )
    print(
        "NEWCASTLE COMPLETE OUTFIELD ROLE MODEL — RAW METRICS, NO PERCENTILE INPUTS".center(
            200
        )
    )
    print(
        "=" * 200
    )
    print(
        "Role Fit = 50 + 10 × weighted standardized score. "
        "50 is roughly league-role average; 60 is roughly one standard deviation above average."
    )
    print(
        "League Strength is contextual only and never changes Role Fit."
    )
    print(
        "Tactical roles: High-Line CB | Vertical FB/WB | Holding Pivot | Advanced Pivot | "
        "Creative 10 | Vertical Winger | Inside Creator | Focal Striker | Second Striker."
    )
    print(
        "Goalkeeper replacement is intentionally omitted; Horníček remains outside this recruitment model."
    )

    important_metrics = sorted({
        metric
        for model in ROLE_MODELS.values()
        for metric in model["weights"]
    })

    available = []
    unavailable = []

    for metric in important_metrics:
        if metric in historical.columns and pd.to_numeric(
            historical[metric], errors="coerce"
        ).notna().sum() > 0:
            available.append(metric)
        else:
            unavailable.append(metric)

    print()
    print("HISTORICAL METRIC AVAILABILITY")
    print("-" * 120)
    print(f"Available role metrics: {len(available)}/{len(important_metrics)}")
    if unavailable:
        print("Unavailable / all-null metrics:")
        print("  " + ", ".join(unavailable))

    print_summary(
        summary,
        long_df,
    )

    print_role_tables(
        long_df
    )

    print_dedic_external_benchmark()

    print_target_checks(
        summary,
        long_df,
    )

    print()
    print("=" * 200)
    print("HISTORICAL ROLE ANALYSIS — RECRUITMENT HANDOFF")
    print("=" * 200)
    print("Priority roles for the external shortlist:")
    print("  1. Starting Creative 10")
    print("  2. Starting Pressing Focal Striker")
    print("  3. High-Line Center Back with passing + tackling/duel quality")
    print(
        "Lower priority after internal depth review: full-back and central midfield. "
        "Hall, Dedić, Livramento, Nico, Bamba, Miley, Steur, Joelinton and Ramsey "
        "provide enough internal coverage to avoid forcing a purchase."
    )
    print(
        "Winger recruitment is value-driven rather than urgent. Use the 2025/26 historical "
        "profiles to avoid overreacting to suppressed 2026/27 involvement in Jaissle's system."
    )

    print()
    print(
        "Saved:"
    )
    print(
        OUT_SUMMARY
    )
    print(
        OUT_LONG
    )


if __name__ == "__main__":
    main()
