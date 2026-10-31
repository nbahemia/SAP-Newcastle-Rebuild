from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# Place this file in:
# scripts/bsd/analyze/role_scouting/build_newcastle_historical_benchmarks.py
ROOT = Path(__file__).resolve().parents[4]
PROCESSED = ROOT / "data" / "processed" / "bsd"
SHORTLIST_DIR = PROCESSED / "shortlist"

HISTORICAL = PROCESSED / "europe_2025_26_player_season_metrics.parquet"
ENRICHED_HISTORICAL = (
    PROCESSED / "europe_2025_26_player_percentiles_enriched.parquet"
)
CURRENT_TEAM = PROCESSED / "player_team_season_metrics.parquet"
CURRENT_DIAGNOSTIC = PROCESSED / "newcastle_player_diagnostic.parquet"

OUT_LONG = PROCESSED / "newcastle_jaissle_role_benchmarks.parquet"
OUT_CSV = SHORTLIST_DIR / "newcastle_jaissle_role_benchmarks.csv"
OUT_OPTIMAL = PROCESSED / "newcastle_jaissle_optimal_roles.parquet"
OUT_OPTIMAL_CSV = SHORTLIST_DIR / "newcastle_jaissle_optimal_roles.csv"
OUT_TM_AUDIT = SHORTLIST_DIR / "newcastle_transfermarkt_match_audit.csv"

HIST_MIN_MINUTES = 450
ROLE_CENTER = 50.0
ROLE_SCALE = 10.0

# ---------------------------------------------------------------------
# Current Newcastle roster / role families.
# These explicit rows make the bridge independent of current-minute
# filters in the Newcastle diagnostic parquet.
# ---------------------------------------------------------------------
CURRENT_ROSTER = {
    "Lewis Hall": "FB/WB",
    "Tino Livramento": "FB/WB",
    "Sven Botman": "CB",
    "Malick Thiaw": "CB",
    "Fabian Schar": "CB",
    "Dan Burn": "CB",
    "Nico Gonzalez": "DM/CM",
    "Aladji Bamba": "DM/CM",
    "Joelinton": "DM/CM",
    "Jacob Ramsey": "DM/CM",
    "Joe Willock": "DM/CM",
    "Lewis Miley": "DM/CM",
    "Sean Steur": "DM/CM",
    "Anthony Elanga": "W",
    "Jacob Murphy": "W",
    "Harvey Barnes": "W",
    "Bazoumana Toure": "W",
    "Matias Fernandez-Pardo": "W",
    "Yoane Wissa": "ST",
    "William Osula": "ST",
    "Nick Woltemade": "ST",
}

# Historical identity hints. Only used to narrow the full 2025/26 pool.
HISTORICAL_OVERRIDES = {
    "bazoumana toure": {
        "league_slug": "bundesliga",
        "position_group": "W",
        "terms": ["toure"],
        "aliases": ["b toure", "bazoumana toure"],
    },
    "matias fernandez pardo": {
        "league_slug": "ligue_1",
        "position_group": "W",
        "terms": ["fernandez"],
        "aliases": ["m fernandez", "matias fernandez pardo"],
    },
    "nick woltemade": {
        "league_slug": "premier_league",
        "position_group": "ST",
        "terms": ["woltemade"],
    },
    "tino livramento": {
        "league_slug": "premier_league",
        "position_group": "FB/WB",
        "terms": ["livramento"],
    },
    "aladji bamba": {
        "league_slug": "ligue_1",
        "position_group": "DM/CM",
        "terms": ["bamba"],
    },
    "jacob ramsey": {
        "league_slug": "premier_league",
        "position_group": "DM/CM",
        "terms": ["ramsey"],
    },
    "malick thiaw": {
        "position_group": "CB",
        "terms": ["thiaw"],
        "aliases": ["m thiaw", "malick thiaw"],
    },
    "nico gonzalez": {
        "league_slug": "premier_league",
        "position_group": "DM/CM",
        "terms": ["gonzalez"],
    },
    "sean steur": {
        "league_slug": "eredivisie",
        "position_group": "DM/CM",
        "terms": ["steur"],
    },
}

ROLE_MODELS = {
    "center_back": {
        "label": "High-Line Center Back",
        "positions": {"CB"},
        "weights": {
            "interceptions_per90": .16,
            "recoveries_per90": .15,
            "aerials_won_per90": .14,
            "duels_won_per90": .13,
            "tackles_won_per90": .10,
            "pass_accuracy_pct": .10,
            "total_long_balls_per90": .08,
            "successful_passes_per90": .08,
            "progressive_carries_per90": .06,
        },
    },
    "vertical_wingback": {
        "label": "Vertical Full-Back / Wingback",
        "positions": {"FB/WB"},
        "weights": {
            "progressive_carries_per90": .20,
            "accurate_opposition_half_passes_per90": .15,
            "total_crosses_per90": .12,
            "xa_per90": .10,
            "recoveries_per90": .12,
            "tackles_won_per90": .08,
            "interceptions_per90": .07,
            "duels_won_per90": .06,
            "successful_passes_per90": .05,
            "touches_per90": .05,
        },
    },
    "advanced_pivot": {
        "label": "Advanced Pivot / Workhorse",
        "positions": {"DM/CM", "AM"},
        "weights": {
            "recoveries_per90": .18,
            "duels_won_per90": .15,
            "tackles_won_per90": .12,
            "interceptions_per90": .12,
            "progressive_carries_per90": .14,
            "accurate_opposition_half_passes_per90": .08,
            "touches_per90": .06,
            "successful_passes_per90": .05,
            "xg_per90": .04,
            "shots_per90": .03,
            "xa_per90": .03,
        },
    },
    "creative_cam": {
        "label": "Creative 10",
        "positions": {"DM/CM", "AM", "W"},
        "weights": {
            "xa_per90": .22,
            "key_passes_per90": .20,
            "accurate_opposition_half_passes_per90": .15,
            "progressive_carries_per90": .12,
            "successful_passes_per90": .08,
            "touches_per90": .08,
            "recoveries_per90": .05,
            "interceptions_per90": .03,
            "xg_per90": .04,
            "shots_per90": .03,
        },
    },
    "vertical_winger": {
        "label": "Vertical Transition Winger",
        "positions": {"W", "AM"},
        "weights": {
            "progressive_carries_per90": .22,
            "xg_per90": .15,
            "shots_per90": .12,
            "shots_on_target_per90": .08,
            "accurate_opposition_half_passes_per90": .10,
            "xa_per90": .07,
            "key_passes_per90": .05,
            "total_crosses_per90": .05,
            "recoveries_per90": .07,
            "duels_won_per90": .05,
            "touches_per90": .04,
        },
    },
    "focal_striker": {
        "label": "Pressing Focal Striker",
        "positions": {"ST"},
        "weights": {
            "xg_per90": .17,
            "shots_per90": .10,
            "shots_on_target_per90": .08,
            "aerials_won_per90": .15,
            "duels_won_per90": .13,
            "successful_passes_per90": .09,
            "touches_per90": .08,
            "key_passes_per90": .06,
            "xa_per90": .05,
            "progressive_carries_per90": .04,
            "recoveries_per90": .05,
        },
    },
}

ROLE_DETAIL = {
    "creative_cam": {
        "csv": "creative_10_jaissle_detailed.csv",
        "score_col": "jaissle_cam_score",
        "components": {
            "creation_link_index": {
                "xa_per90": .30,
                "key_passes_per90": .30,
                "successful_passes_per90": .18,
                "total_passes_per90": .12,
                "touches_per90": .10,
            },
            "attacking_threat_index": {
                "xg_per90": .45,
                "shots_per90": .30,
                "shots_on_target_per90": .25,
            },
            "pressing_activity_index": {
                "recoveries_per90": .35,
                "tackles_won_per90": .25,
                "interceptions_per90": .20,
                "duels_won_per90": .20,
            },
        },
        "formula": {
            "role_fit": .40,
            "creation_link_index": .35,
            "attacking_threat_index": .10,
            "pressing_activity_index": .15,
        },
    },
    "focal_striker": {
        "csv": "focal_striker_jaissle_detailed.csv",
        "score_col": "jaissle_striker_score",
        "components": {
            "box_threat_index": {
                "xg_per90": .45,
                "shots_per90": .30,
                "shots_on_target_per90": .25,
            },
            "focal_presence_index": {
                "duels_won_per90": .35,
                "aerials_won_per90": .30,
                "touches_per90": .20,
                "total_passes_per90": .15,
            },
            "link_play_index": {
                "successful_passes_per90": .35,
                "total_passes_per90": .20,
                "key_passes_per90": .20,
                "xa_per90": .15,
                "touches_per90": .10,
            },
            "pressing_activity_index": {
                "recoveries_per90": .40,
                "tackles_won_per90": .25,
                "interceptions_per90": .15,
                "duels_won_per90": .20,
            },
        },
        "formula": {
            "role_fit": .30,
            "box_threat_index": .25,
            "focal_presence_index": .20,
            "link_play_index": .10,
            "pressing_activity_index": .15,
        },
    },
    "center_back": {
        "csv": "center_back_jaissle_detailed.csv",
        "score_col": "jaissle_cb_score",
        "components": {
            "build_up_index": {
                "pass_accuracy_pct": .30,
                "successful_passes_per90": .25,
                "total_passes_per90": .20,
                "accurate_long_balls_per90": .15,
                "touches_per90": .10,
            },
            "front_foot_defending_index": {
                "recoveries_per90": .30,
                "interceptions_per90": .30,
                "tackles_won_per90": .20,
                "duels_won_per90": .20,
            },
            "physical_security_index": {
                "aerials_won_per90": .40,
                "duels_won_per90": .35,
                "total_aerials_per90": .15,
                "total_duels_per90": .10,
            },
        },
        "formula": {
            "role_fit": .40,
            "build_up_index": .25,
            "front_foot_defending_index": .20,
            "physical_security_index": .15,
        },
    },
    "advanced_pivot": {
        "csv": "workhorse_cm_jaissle_detailed.csv",
        "score_col": "jaissle_workhorse_score",
        "components": {
            "workhorse_activity_index": {
                "recoveries_per90": .30,
                "tackles_won_per90": .25,
                "interceptions_per90": .20,
                "duels_won_per90": .20,
                "aerials_won_per90": .05,
            },
            "ball_security_link_index": {
                "pass_accuracy_pct": .30,
                "successful_passes_per90": .25,
                "total_passes_per90": .20,
                "touches_per90": .15,
                "xa_per90": .10,
            },
            "attacking_support_index": {
                "xa_per90": .30,
                "key_passes_per90": .25,
                "xg_per90": .25,
                "shots_per90": .20,
            },
        },
        "formula": {
            "role_fit": .40,
            "workhorse_activity_index": .30,
            "ball_security_link_index": .20,
            "attacking_support_index": .10,
        },
    },
    "vertical_winger": {
        "csv": "winger_jaissle_archetypes.csv",
        "score_col": "best_winger_score",
        "archetype_mode": True,
        "components": {
            "wide_creator_score": {
                "xa_per90": .20,
                "key_passes_per90": .18,
                "big_chances_created_per90": .12,
                "line_breaking_passes_per90": .15,
                "accurate_crosses_per90": .10,
                "successful_passes_per90": .10,
                "pass_accuracy_pct": .05,
                "touches_per90": .05,
                "recoveries_per90": .05,
            },
            "vertical_box_score": {
                "xg_per90": .18,
                "shots_per90": .12,
                "shots_on_target_per90": .10,
                "box_touches_per90": .15,
                "successful_dribbles_per90": .12,
                "duels_won_per90": .08,
                "fouls_won_per90": .05,
                "recoveries_per90": .07,
                "possession_won_final_third_per90": .06,
                "xa_per90": .07,
            },
            "hybrid_score": {
                "xa_per90": .13,
                "key_passes_per90": .10,
                "xg_per90": .12,
                "shots_per90": .08,
                "shots_on_target_per90": .06,
                "box_touches_per90": .08,
                "successful_dribbles_per90": .08,
                "accurate_crosses_per90": .06,
                "successful_passes_per90": .07,
                "touches_per90": .05,
                "recoveries_per90": .07,
                "duels_won_per90": .05,
                "possession_won_final_third_per90": .05,
            },
        },
    },
    "vertical_wingback": {
        "csv": "left_wingback_recruitment_progressive.csv",
        "score_col": "jaissle_fb_score",
        "components": {
            "attacking_creation_index": {
                "xa_per90": .30,
                "key_passes_per90": .25,
                "accurate_crosses_per90": .20,
                "successful_passes_per90": .15,
                "touches_per90": .10,
            },
            "physical_activity_index": {
                "duels_won_per90": .30,
                "aerials_won_per90": .15,
                "recoveries_per90": .25,
                "interceptions_per90": .15,
                "tackles_won_per90": .15,
            },
        },
        "formula": {
            "role_fit": .40,
            "attacking_creation_index": .30,
            "physical_activity_index": .30,
        },
    },
}


def norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def ensure_common_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        "big_chance_created": "big_chances_created",
        "line_breaking_passes": "line_breaking_passes",
        "won_contest": "successful_dribbles",
        "fouls_won": "fouls_won",
        "possession_won_final_third": "possession_won_final_third",
        "ball_recovery": "recoveries",
        "interception": "interceptions",
        "won_tackle": "tackles_won",
        "total_tackle": "total_tackles",
        "duel_won": "duels_won",
        "duel_lost": "duels_lost",
        "aerial_won": "aerials_won",
        "aerial_lost": "aerials_lost",
        "progressive_ball_carries_count": "progressive_carries",
        "accurate_opposition_half_pass": "accurate_opposition_half_passes",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    minutes = pd.to_numeric(
        out["minutes"], errors="coerce"
    ) if "minutes" in out.columns else pd.Series(np.nan, index=out.index)

    total_to_p90 = {
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
        "big_chances_created": "big_chances_created_per90",
        "line_breaking_passes": "line_breaking_passes_per90",
        "successful_dribbles": "successful_dribbles_per90",
        "fouls_won": "fouls_won_per90",
        "possession_won_final_third": "possession_won_final_third_per90",
        "box_touches": "box_touches_per90",
        "touches": "touches_per90",
        "progressive_carries": "progressive_carries_per90",
        "recoveries": "recoveries_per90",
        "interceptions": "interceptions_per90",
        "duels_won": "duels_won_per90",
        "aerials_won": "aerials_won_per90",
        "tackles_won": "tackles_won_per90",
    }
    valid_minutes = minutes.where(minutes > 0)
    for total_col, p90_col in total_to_p90.items():
        if total_col not in out.columns:
            continue
        if p90_col not in out.columns or pd.to_numeric(
            out[p90_col], errors="coerce"
        ).notna().sum() == 0:
            out[p90_col] = (
                pd.to_numeric(out[total_col], errors="coerce")
                / valid_minutes * 90.0
            )

    if "pass_accuracy_pct" not in out.columns and {
        "successful_passes", "total_passes"
    }.issubset(out.columns):
        denom = pd.to_numeric(
            out["total_passes"], errors="coerce"
        ).replace(0, np.nan)
        out["pass_accuracy_pct"] = (
            pd.to_numeric(out["successful_passes"], errors="coerce")
            / denom * 100
        )

    if "total_duels_per90" not in out.columns:
        if {"duels_won", "duels_lost"}.issubset(out.columns):
            out["total_duels_per90"] = (
                pd.to_numeric(out["duels_won"], errors="coerce")
                + pd.to_numeric(out["duels_lost"], errors="coerce")
            ) / valid_minutes * 90

    if "total_aerials_per90" not in out.columns:
        if {"aerials_won", "aerials_lost"}.issubset(out.columns):
            out["total_aerials_per90"] = (
                pd.to_numeric(out["aerials_won"], errors="coerce")
                + pd.to_numeric(out["aerials_lost"], errors="coerce")
            ) / valid_minutes * 90

    return out


def robust_z(value, cohort: pd.Series) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    s = pd.to_numeric(cohort, errors="coerce").dropna()
    if pd.isna(value) or len(s) < 8:
        return math.nan
    lo, hi = s.quantile(.05), s.quantile(.95)
    clipped = s.clip(lo, hi)
    sd = clipped.std(ddof=0)
    if pd.isna(sd) or sd <= 1e-9:
        return math.nan
    value = min(max(value, lo), hi)
    return float((value - clipped.mean()) / sd)


def z_to_index(z: float) -> float:
    if pd.isna(z):
        return math.nan
    return float(np.clip(50 + 10 * z, 20, 80))


def name_match_score(a, b) -> float:
    aa, bb = norm_text(a), norm_text(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 100.0
    at, bt = aa.split(), bb.split()
    score = 0.0
    if at[-1] == bt[-1]:
        score += 60
    if at[0][0] == bt[0][0]:
        score += 25
    overlap = len(set(at) & set(bt)) / max(1, len(set(at) | set(bt)))
    score += 15 * overlap
    return score


def find_historical_player(current_name: str, historical: pd.DataFrame):
    normalized = norm_text(current_name)
    override = HISTORICAL_OVERRIDES.get(normalized)
    candidates = historical.copy()

    if override:
        if override.get("league_slug"):
            candidates = candidates[
                candidates["league_slug"].astype(str).eq(
                    override["league_slug"]
                )
            ]
        if override.get("position_group"):
            candidates = candidates[
                candidates["position_group"].astype(str).eq(
                    override["position_group"]
                )
            ]
        terms = [norm_text(x) for x in override.get("terms", [])]
        aliases = [norm_text(x) for x in override.get("aliases", [])]

        if terms or aliases:
            names = candidates["player_name"].astype(str).map(norm_text)

            alias_mask = (
                names.apply(
                    lambda x: any(
                        x == alias
                        or x.startswith(alias + " ")
                        or alias.startswith(x + " ")
                        for alias in aliases
                    )
                )
                if aliases
                else pd.Series(False, index=candidates.index)
            )

            term_mask = (
                names.apply(lambda x: all(t in x for t in terms))
                if terms
                else pd.Series(False, index=candidates.index)
            )

            preferred = candidates[alias_mask | term_mask].copy()

            if preferred.empty:
                return None, (
                    "Historical override matched broad filters but no player "
                    f"matched aliases={aliases} or terms={terms}"
                )

            candidates = preferred

    if candidates.empty:
        return None, "No candidate after identity filters"

    candidates = candidates.copy()
    candidates["_name_score"] = candidates["player_name"].apply(
        lambda x: name_match_score(current_name, x)
    )
    candidates["_minutes_num"] = pd.to_numeric(
        candidates.get("minutes"), errors="coerce"
    ).fillna(0)
    candidates = candidates.sort_values(
        ["_name_score", "_minutes_num"],
        ascending=[False, False],
    )
    best = candidates.iloc[0]
    threshold = 0 if override else 75
    if best["_name_score"] < threshold:
        return None, "Historical identity match below confidence threshold"
    return best, None


def base_role_fit(row: pd.Series, historical: pd.DataFrame, role_key: str):
    model = ROLE_MODELS[role_key]
    league = row.get("league_slug")
    cohort = historical[
        historical["league_slug"].astype(str).eq(str(league))
        & historical["position_group"].isin(model["positions"])
        & (
            pd.to_numeric(historical["minutes"], errors="coerce")
            .fillna(0) >= HIST_MIN_MINUTES
        )
    ].copy()

    weighted = 0.0
    used = 0.0
    for metric, weight in model["weights"].items():
        if metric not in historical.columns:
            continue
        z = robust_z(row.get(metric), cohort[metric])
        if not pd.isna(z):
            weighted += z * weight
            used += weight

    total = sum(model["weights"].values())
    coverage = used / total if total else 0
    if used == 0:
        return math.nan, coverage
    return z_to_index(weighted / used), coverage


def normalize_role_pool_schema(pool: pd.DataFrame) -> pd.DataFrame:
    """
    Role scripts export convenient *_p90 column names, while the historical
    benchmark universe uses canonical *_per90 names. Normalize the role pool
    so Newcastle detailed component scores are calculated on the exact same
    external-player distributions.
    """
    if pool.empty:
        return pool

    out = pool.copy()

    rename = {
        "xa_p90": "xa_per90",
        "xg_p90": "xg_per90",
        "shots_p90": "shots_per90",
        "shots_on_target_p90": "shots_on_target_per90",
        "key_passes_p90": "key_passes_per90",
        "successful_passes_p90": "successful_passes_per90",
        "total_passes_p90": "total_passes_per90",
        "touches_p90": "touches_per90",
        "recoveries_p90": "recoveries_per90",
        "interceptions_p90": "interceptions_per90",
        "tackles_won_p90": "tackles_won_per90",
        "duels_won_p90": "duels_won_per90",
        "aerials_won_p90": "aerials_won_per90",
        "accurate_crosses_p90": "accurate_crosses_per90",
        "accurate_long_balls_p90": "accurate_long_balls_per90",
        "big_chances_created_p90": "big_chances_created_per90",
        "line_breaking_passes_p90": "line_breaking_passes_per90",
        "successful_dribbles_p90": "successful_dribbles_per90",
        "fouls_won_p90": "fouls_won_per90",
        "possession_won_final_third_p90": "possession_won_final_third_per90",
        "box_touches_p90": "box_touches_per90",
        "total_duels_p90": "total_duels_per90",
        "total_aerials_p90": "total_aerials_per90",
        "pass_accuracy_p90": "pass_accuracy_pct",
    }

    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    return out


def detailed_component(
    row: pd.Series,
    pool: pd.DataFrame,
    metric_weights: dict[str, float],
):
    weighted = 0.0
    used = 0.0
    for metric, weight in metric_weights.items():
        if metric not in pool.columns:
            continue
        z = robust_z(row.get(metric), pool[metric])
        if not pd.isna(z):
            weighted += z * weight
            used += weight
    if used == 0:
        return math.nan
    return z_to_index(weighted / used)


def enrich_metadata(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach Transfermarkt metadata to the full 2025/26 historical universe.

    Primary match:
        normalized player name + league_slug

    Fallback:
        normalized player name only, taking the strongest/most complete
        metadata row when league-level matching is unavailable.

    Transfermarkt fields are context only; they never affect the tactical
    role score.
    """
    if not ENRICHED_HISTORICAL.exists():
        audit = pd.DataFrame(columns=[
            "historical_player_name",
            "historical_league",
            "tm_player_name",
            "tm_match_type",
            "age",
            "market_value_eur_m",
            "tm_club_name",
        ])
        return hist.copy(), audit

    meta = pd.read_parquet(ENRICHED_HISTORICAL).copy()

    if "player_name" not in meta.columns:
        return hist.copy(), pd.DataFrame()

    hist = hist.copy()

    hist["_name_norm"] = hist["player_name"].astype(str).map(norm_text)
    meta["_name_norm"] = meta["player_name"].astype(str).map(norm_text)

    if "league_slug" in hist.columns:
        hist["_league_norm"] = hist["league_slug"].astype(str)
    else:
        hist["_league_norm"] = ""

    if "league_slug" in meta.columns:
        meta["_league_norm"] = meta["league_slug"].astype(str)
    else:
        meta["_league_norm"] = ""

    # Standardize useful Transfermarkt fields.
    field_map = {}

    for candidate in ["age"]:
        if candidate in meta.columns:
            field_map[candidate] = "age"
            break

    # Keep raw EUR and EUR-millions distinct so we do not accidentally
    # compare 32,000,000 against a recruitment value of 35.0.
    if "market_value_eur_m" in meta.columns:
        field_map["market_value_eur_m"] = "market_value_eur_m"
    elif "market_value_eur" in meta.columns:
        meta["market_value_eur_m"] = (
            pd.to_numeric(meta["market_value_eur"], errors="coerce")
            / 1_000_000.0
        )
    elif "market_value" in meta.columns:
        raw_value = pd.to_numeric(meta["market_value"], errors="coerce")
        # Transfermarkt raw snapshots are normally in euros.
        meta["market_value_eur_m"] = np.where(
            raw_value.abs() > 1000,
            raw_value / 1_000_000.0,
            raw_value,
        )

    for candidate in [
        "club_name",
        "current_club_name",
        "team_name",
    ]:
        if candidate in meta.columns:
            field_map[candidate] = "tm_club_name"
            break

    for candidate in [
        "current_club_id",
        "club_id",
    ]:
        if candidate in meta.columns:
            field_map[candidate] = "tm_club_id"
            break

    for old, new in field_map.items():
        if old != new and new not in meta.columns:
            meta = meta.rename(columns={old: new})

    if "market_value_eur_m" in meta.columns:
        meta["market_value_eur_m"] = pd.to_numeric(
            meta["market_value_eur_m"], errors="coerce"
        )

    useful = [
        c for c in [
            "player_name",
            "_name_norm",
            "_league_norm",
            "age",
            "market_value_eur_m",
            "tm_club_name",
            "tm_club_id",
        ]
        if c in meta.columns
    ]

    meta_use = meta[useful].copy()

    # Prefer rows with the most complete TM metadata.
    completeness_cols = [
        c for c in [
            "age",
            "market_value_eur_m",
            "tm_club_name",
            "tm_club_id",
        ]
        if c in meta_use.columns
    ]

    if completeness_cols:
        meta_use["_completeness"] = (
            meta_use[completeness_cols]
            .notna()
            .sum(axis=1)
        )
    else:
        meta_use["_completeness"] = 0

    meta_use = meta_use.sort_values(
        "_completeness",
        ascending=False,
    )

    exact = meta_use.drop_duplicates(
        ["_name_norm", "_league_norm"],
        keep="first",
    )

    fallback = meta_use.drop_duplicates(
        "_name_norm",
        keep="first",
    )

    # First merge exact normalized name + historical league.
    exact_cols = [
        c for c in [
            "_name_norm",
            "_league_norm",
            "player_name",
            "age",
            "market_value_eur_m",
            "tm_club_name",
            "tm_club_id",
        ]
        if c in exact.columns
    ]

    exact_renamed = exact[exact_cols].rename(
        columns={"player_name": "_tm_player_name_exact"}
    )

    out = hist.merge(
        exact_renamed,
        on=["_name_norm", "_league_norm"],
        how="left",
        suffixes=("", "_tm"),
    )

    # Name-only fallback for unresolved metadata.
    fallback_cols = [
        c for c in [
            "_name_norm",
            "player_name",
            "age",
            "market_value_eur_m",
            "tm_club_name",
            "tm_club_id",
        ]
        if c in fallback.columns
    ]

    fallback_renamed = fallback[fallback_cols].rename(
        columns={
            "player_name": "_tm_player_name_fallback",
            "age": "_age_fallback",
            "market_value_eur_m": "_value_fallback",
            "tm_club_name": "_club_fallback",
            "tm_club_id": "_club_id_fallback",
        }
    )

    out = out.merge(
        fallback_renamed,
        on="_name_norm",
        how="left",
    )

    # Coalesce metadata.
    if "age" not in out.columns:
        out["age"] = pd.NA
    if "market_value_eur_m" not in out.columns:
        out["market_value_eur_m"] = pd.NA
    if "tm_club_name" not in out.columns:
        out["tm_club_name"] = pd.NA
    if "tm_club_id" not in out.columns:
        out["tm_club_id"] = pd.NA

    out["age"] = out["age"].where(
        pd.to_numeric(out["age"], errors="coerce").notna(),
        out.get("_age_fallback"),
    )

    out["market_value_eur_m"] = out["market_value_eur_m"].where(
        pd.to_numeric(out["market_value_eur_m"], errors="coerce").notna(),
        out.get("_value_fallback"),
    )

    out["tm_club_name"] = out["tm_club_name"].where(
        out["tm_club_name"].notna(),
        out.get("_club_fallback"),
    )

    out["tm_club_id"] = out["tm_club_id"].where(
        out["tm_club_id"].notna(),
        out.get("_club_id_fallback"),
    )

    exact_match = out.get("_tm_player_name_exact").notna()
    fallback_match = (
        ~exact_match
        & out.get("_tm_player_name_fallback").notna()
    )

    out["tm_match_type"] = np.select(
        [exact_match, fallback_match],
        ["name+league", "name-only fallback"],
        default="unmatched",
    )

    out["tm_player_name"] = out.get(
        "_tm_player_name_exact"
    ).where(
        exact_match,
        out.get("_tm_player_name_fallback"),
    )

    audit_cols = [
        "player_name",
        "league_slug",
        "tm_player_name",
        "tm_match_type",
        "age",
        "market_value_eur_m",
        "tm_club_name",
        "tm_club_id",
    ]
    audit_cols = [c for c in audit_cols if c in out.columns]

    audit = out[audit_cols].copy().rename(columns={
        "player_name": "historical_player_name",
        "league_slug": "historical_league",
    })

    # Drop temporary matching columns from the enriched universe.
    drop_cols = [
        c for c in [
            "_name_norm",
            "_league_norm",
            "_tm_player_name_exact",
            "_tm_player_name_fallback",
            "_age_fallback",
            "_value_fallback",
            "_club_fallback",
            "_club_id_fallback",
        ]
        if c in out.columns
    ]
    out = out.drop(columns=drop_cols)

    return out, audit


def current_roster() -> pd.DataFrame:
    rows = [{"player_name": n, "position_group": p} for n, p in CURRENT_ROSTER.items()]
    if CURRENT_DIAGNOSTIC.exists():
        diag = pd.read_parquet(CURRENT_DIAGNOSTIC)
        if "player_name" in diag.columns:
            for _, r in diag.iterrows():
                n = str(r["player_name"])
                pos = r.get("position_group", pd.NA)
                if (
                    str(pos) != "GK"
                    and norm_text(n) not in {norm_text(x["player_name"]) for x in rows}
                ):
                    rows.append({
                        "player_name": n,
                        "position_group": pos,
                    })
    return pd.DataFrame(rows)



def build_optimal_roles(out: pd.DataFrame) -> pd.DataFrame:
    """
    One retention row per Newcastle player.

    The player's optimal role is the plausible role with the highest
    detailed Jaissle score when available, otherwise the highest base
    role-fit score. Secondary-role failures are not used to recommend
    selling a player.
    """
    if out.empty:
        return pd.DataFrame()

    resolved = out[
        out.get("historical_found", False).eq(True)
    ].copy()

    if resolved.empty:
        return pd.DataFrame()

    resolved["_decision_score"] = pd.to_numeric(
        resolved.get("jaissle_role_score"),
        errors="coerce",
    )

    fallback_fit = pd.to_numeric(
        resolved.get("role_fit"),
        errors="coerce",
    )

    resolved["_decision_score"] = (
        resolved["_decision_score"]
        .where(resolved["_decision_score"].notna(), fallback_fit)
    )

    # If a player somehow has duplicated rows for the same role, take
    # the most complete / highest-scoring version first.
    resolved["_coverage_num"] = pd.to_numeric(
        resolved.get("metric_coverage"),
        errors="coerce",
    ).fillna(0)

    resolved = resolved.sort_values(
        [
            "player_name",
            "_decision_score",
            "_coverage_num",
        ],
        ascending=[True, False, False],
        na_position="last",
    )

    optimal = (
        resolved
        .drop_duplicates("player_name", keep="first")
        .copy()
    )

    optimal = optimal.rename(columns={
        "role_key": "optimal_role_key",
        "role": "optimal_role",
        "role_fit": "optimal_role_fit",
        "jaissle_role_score": "optimal_jaissle_score",
        "metric_coverage": "optimal_metric_coverage",
    })

    keep = [
        "player_name",
        "current_position",
        "historical_player_name",
        "historical_league",
        "historical_position",
        "historical_minutes",
        "age",
        "market_value_eur_m",
        "tm_club_name",
        "tm_club_id",
        "tm_match_type",
        "tm_player_name",
        "optimal_role_key",
        "optimal_role",
        "optimal_role_fit",
        "optimal_jaissle_score",
        "optimal_metric_coverage",
        "best_winger_archetype",
    ]

    keep = [c for c in keep if c in optimal.columns]

    return optimal[keep].sort_values(
        "player_name"
    ).reset_index(drop=True)


def main():
    if not HISTORICAL.exists():
        raise FileNotFoundError(HISTORICAL)

    historical = ensure_common_columns(pd.read_parquet(HISTORICAL))
    historical, tm_audit = enrich_metadata(historical)
    roster = current_roster()

    role_pools = {}
    for role_key, detail in ROLE_DETAIL.items():
        path = SHORTLIST_DIR / detail["csv"]
        role_pools[role_key] = (
            normalize_role_pool_schema(pd.read_csv(path))
            if path.exists()
            else pd.DataFrame()
        )

    rows = []

    for _, current in roster.iterrows():
        current_name = current["player_name"]
        current_pos = current.get("position_group")
        hist, note = find_historical_player(current_name, historical)

        if hist is None:
            rows.append({
                "player_name": current_name,
                "current_position": current_pos,
                "historical_found": False,
                "historical_note": note,
            })
            continue

        for role_key, model in ROLE_MODELS.items():
            # Only evaluate plausible current/historical positions.
            historical_pos = hist.get("position_group")
            if (
                str(current_pos) not in model["positions"]
                and str(historical_pos) not in model["positions"]
            ):
                continue

            role_fit, coverage = base_role_fit(
                hist, historical, role_key
            )

            detail = ROLE_DETAIL.get(role_key)
            component_values = {}
            detailed_score = math.nan
            best_winger_archetype = pd.NA

            if detail:
                pool = role_pools.get(role_key, pd.DataFrame())
                if not pool.empty:
                    for component, weights in detail["components"].items():
                        component_values[component] = detailed_component(
                            hist, pool, weights
                        )

                    if detail.get("archetype_mode"):
                        available_archetypes = {
                            "Wide Creator": component_values.get("wide_creator_score"),
                            "Vertical / Box Winger": component_values.get("vertical_box_score"),
                            "Hybrid Winger": component_values.get("hybrid_score"),
                        }

                        available_archetypes = {
                            name: score
                            for name, score in available_archetypes.items()
                            if not pd.isna(score)
                        }

                        if available_archetypes:
                            best_winger_archetype = max(
                                available_archetypes,
                                key=available_archetypes.get,
                            )
                            detailed_score = available_archetypes[
                                best_winger_archetype
                            ]
                        else:
                            best_winger_archetype = pd.NA
                    else:
                        terms = []
                        for key, weight in detail["formula"].items():
                            value = (
                                role_fit
                                if key == "role_fit"
                                else component_values.get(key)
                            )
                            if not pd.isna(value):
                                terms.append((value, weight))

                        if terms:
                            used_weight = sum(w for _, w in terms)
                            detailed_score = sum(v * w for v, w in terms) / used_weight

            row = {
                "player_name": current_name,
                "current_position": current_pos,
                "historical_found": True,
                "historical_player_name": hist.get("player_name"),
                "historical_league": hist.get("league_slug"),
                "historical_position": hist.get("position_group"),
                "historical_minutes": hist.get("minutes"),
                "age": hist.get("age"),
                "market_value_eur_m": hist.get("market_value_eur_m"),
                "tm_club_name": hist.get("tm_club_name"),
                "tm_club_id": hist.get("tm_club_id"),
                "tm_match_type": hist.get("tm_match_type"),
                "tm_player_name": hist.get("tm_player_name"),
                "role_key": role_key,
                "role": model["label"],
                "role_fit": role_fit,
                "metric_coverage": coverage * 100,
                "jaissle_role_score": detailed_score,
                "best_winger_archetype": (
                    best_winger_archetype
                    if role_key == "vertical_winger"
                    else pd.NA
                ),
                "historical_note": note,
            }
            row.update(component_values)
            rows.append(row)

    out = pd.DataFrame(rows)
    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_LONG, index=False)
    out.to_csv(OUT_CSV, index=False)

    optimal = build_optimal_roles(out)
    optimal.to_parquet(OUT_OPTIMAL, index=False)
    optimal.to_csv(OUT_OPTIMAL_CSV, index=False)

    if not tm_audit.empty:
        tm_audit.to_csv(OUT_TM_AUDIT, index=False)

    print("=" * 150)
    print("NEWCASTLE 2025/26 HISTORICAL BENCHMARK BRIDGE")
    print("=" * 150)

    found = out[out.get("historical_found", False) == True].copy()
    show = [
        c for c in [
            "player_name", "historical_player_name",
            "historical_league", "historical_minutes",
            "role_key", "role_fit", "jaissle_role_score",
            "metric_coverage", "market_value_eur_m",
            "tm_club_name", "tm_match_type"
        ]
        if c in found.columns
    ]
    if show:
        print(
            found[show]
            .sort_values(["role_key", "jaissle_role_score"], ascending=[True, False])
            .to_string(index=False)
        )

    missing_names = (
        out[out.get("historical_found", False) == False]["player_name"]
        .dropna().unique().tolist()
        if "historical_found" in out.columns
        else []
    )
    if missing_names:
        print()
        print("UNRESOLVED CURRENT NEWCASTLE PLAYERS")
        print("-" * 80)
        for name in missing_names:
            print(" ", name)

    print()
    print("Saved:")
    print(" ", OUT_LONG)
    print(" ", OUT_CSV)
    print(" ", OUT_OPTIMAL)
    print(" ", OUT_OPTIMAL_CSV)
    print(" ", OUT_TM_AUDIT)


if __name__ == "__main__":
    main()
