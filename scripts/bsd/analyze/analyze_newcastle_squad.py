#!/usr/bin/env python3
"""
analyze_newcastle_squad_fixed_v2.py

Fixed Newcastle squad diagnostic with CURRENT SQUAD as the master roster.

Why v2 exists
-------------
The first historical fix used:
    newcastle_2025_26_historical_role_comparison.parquet

as the master roster. That is wrong because some current Newcastle players
(e.g. Bazoumana Touré) may be absent from that bridge file.

This version instead uses:
    newcastle_player_diagnostic.parquet

as the CURRENT Newcastle squad master list.

Then it attaches 2025/26 historical data using, in order:
1. newcastle_2025_26_historical_role_comparison.parquet
2. direct historical player matching against
   europe_2025_26_player_percentiles_enriched.parquet
3. Newcastle-specific role benchmark / optimal-role files as fallback metadata

This guarantees current-squad players remain visible even when historical matching fails.

Outputs:
    newcastle_2025_26_squad_diagnostic_v2.parquet
    newcastle_2025_26_squad_diagnostic_v2.csv
    newcastle_2025_26_unit_summary_v2.parquet
    newcastle_2025_26_unit_summary_v2.csv

Run:
    python scripts/bsd/analyze/analyze_newcastle_squad.py
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"
SHORTLIST_DIR = PROCESSED_DIR / "shortlist"

CURRENT_SQUAD_FILE = (
    PROCESSED_DIR
    / "newcastle_player_diagnostic.parquet"
)

HISTORICAL_BRIDGE_FILE = (
    PROCESSED_DIR
    / "newcastle_2025_26_historical_role_comparison.parquet"
)

HISTORICAL_PERCENTILES_FILE = (
    PROCESSED_DIR
    / "europe_2025_26_player_percentiles_enriched.parquet"
)

NEWCASTLE_ROLE_BENCHMARKS_FILE = (
    PROCESSED_DIR
    / "newcastle_jaissle_role_benchmarks.parquet"
)

NEWCASTLE_OPTIMAL_ROLES_FILE = (
    PROCESSED_DIR
    / "newcastle_jaissle_optimal_roles.parquet"
)

PLAYER_OUTPUT_FILE = (
    PROCESSED_DIR
    / "newcastle_2025_26_squad_diagnostic_v4.parquet"
)

PLAYER_OUTPUT_CSV = (
    PROCESSED_DIR
    / "newcastle_2025_26_squad_diagnostic_v4.csv"
)

UNIT_OUTPUT_FILE = (
    PROCESSED_DIR
    / "newcastle_2025_26_unit_summary_v4.parquet"
)

UNIT_OUTPUT_CSV = (
    PROCESSED_DIR
    / "newcastle_2025_26_unit_summary_v4.csv"
)


# ======================================================================================
# SETTINGS
# ======================================================================================

MINUTES_WEIGHT_CAP = 900

POSITION_ORDER = [
    "GK",
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
]

DEPTH_THRESHOLDS = {
    "players_180_plus": 180,
    "players_360_plus": 360,
    "players_540_plus": 540,
}

# Explicit aliases for known Newcastle players / likely short-name variants.
PLAYER_ALIASES = {
    "Bazoumana Touré": [
        "B. Touré",
        "B. Toure",
        "Bazoumana Toure",
    ],
    "Anthony Elanga": [
        "A. Elanga",
    ],
    "Matias Fernandez-Pardo": [
        "M. Fernandez-Pardo",
        "M. Fernández-Pardo",
        "Matías Fernández-Pardo",
        "Matias Fernández-Pardo",
    ],
    "William Osula": [
        "W. Osula",
    ],
    "Lewis Hall": [
        "L. Hall",
    ],
    "Harvey Barnes": [
        "H. Barnes",
    ],
    "Sven Botman": [
        "S. Botman",
    ],
    "Malick Thiaw": [
        "M. Thiaw",
    ],
}


# Strict identity constraints for players whose surname/initial aliases can collide.
# These constraints apply ONLY when selecting the 2025/26 historical row.
STRICT_HISTORICAL_MATCH_RULES = {
    "Bazoumana Touré": {
        "league_slugs": {
            "bundesliga",
        },
        "club_terms": {
            "tsg hoffenheim",
            "hoffenheim",
            "tsg 1899 hoffenheim",
        },
    },
}


# Current squad position overrides.
# The current diagnostic can inherit noisy position labels from small samples.
# These are project-level squad roles, not historical-position rewrites.
CURRENT_POSITION_OVERRIDES = {
    "Bazoumana Touré": "W",
    "Matías Fernández-Pardo": "W",
    "Anthony Elanga": "W",
    "Harvey Barnes": "W",
    "Jacob Murphy": "W",
}


# ======================================================================================
# METRICS
# ======================================================================================

POSITION_METRICS = {
    "GK": [],

    "CB": [
        "successful_passes_per90",
        "pass_accuracy_pct",
        "line_breaking_passes_per90",
        "aerials_won_per90",
        "duels_won_per90",
        "recoveries_per90",
        "interceptions_per90",
        "tackles_won_per90",
    ],

    "FB/WB": [
        "progressive_carries_per90",
        "progressive_carry_distance_per90",
        "accurate_opposition_half_passes_per90",
        "line_breaking_passes_per90",
        "total_crosses_per90",
        "xa_per90",
        "recoveries_per90",
        "interceptions_per90",
        "duels_won_per90",
        "tackles_won_per90",
    ],

    "DM/CM": [
        "successful_passes_per90",
        "pass_accuracy_pct",
        "accurate_opposition_half_passes_per90",
        "line_breaking_passes_per90",
        "total_long_balls_per90",
        "progressive_carries_per90",
        "touches_per90",
        "recoveries_per90",
        "interceptions_per90",
        "duels_won_per90",
        "xa_per90",
    ],

    "AM": [
        "xg_per90",
        "xa_per90",
        "key_passes_per90",
        "line_breaking_passes_per90",
        "accurate_opposition_half_passes_per90",
        "progressive_carries_per90",
        "successful_dribbles_per90",
        "box_touches_per90",
        "touches_per90",
        "recoveries_per90",
    ],

    "W": [
        "xg_per90",
        "xa_per90",
        "shots_per90",
        "shots_on_target_per90",
        "key_passes_per90",
        "big_chances_created_per90",
        "line_breaking_passes_per90",
        "successful_dribbles_per90",
        "progressive_carries_per90",
        "box_touches_per90",
        "total_crosses_per90",
        "accurate_opposition_half_passes_per90",
        "touches_per90",
        "recoveries_per90",
        "possession_won_final_third_per90",
    ],

    "ST": [
        "xg_per90",
        "goals_per90",
        "shots_per90",
        "shots_on_target_per90",
        "box_touches_per90",
        "aerials_won_per90",
        "duels_won_per90",
        "successful_passes_per90",
        "key_passes_per90",
        "xa_per90",
        "touches_per90",
        "recoveries_per90",
    ],
}


METRIC_LABELS = {
    "goals_per90": "Goals",
    "xg_per90": "xG",
    "xa_per90": "xA",
    "shots_per90": "Shots",
    "shots_on_target_per90": "Shots on Target",
    "key_passes_per90": "Chances Created",
    "big_chances_created_per90": "Big Chances Created",
    "successful_passes_per90": "Successful Passes",
    "pass_accuracy_pct": "Pass Accuracy",
    "accurate_opposition_half_passes_per90": "Accurate Opposition-Half Passes",
    "line_breaking_passes_per90": "Line-Breaking Passes",
    "total_long_balls_per90": "Long Balls",
    "total_crosses_per90": "Crosses",
    "touches_per90": "Touches",
    "box_touches_per90": "Box Touches",
    "successful_dribbles_per90": "Successful Dribbles",
    "progressive_carries_per90": "Progressive Carries",
    "progressive_carry_distance_per90": "Progressive Carry Distance",
    "duels_won_per90": "Duels Won",
    "aerials_won_per90": "Aerials Won",
    "recoveries_per90": "Recoveries",
    "possession_won_final_third_per90": "Possession Won Final 3rd",
    "interceptions_per90": "Interceptions",
    "tackles_won_per90": "Tackles Won",
}


# ======================================================================================
# HELPERS
# ======================================================================================

def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def name_aliases(value: object) -> set[str]:
    n = norm(value)

    if not n:
        return set()

    parts = n.split()
    aliases = {n}

    if len(parts) >= 2:
        # Keep first-initial + surname matching, but NEVER surname-only matching.
        # Surname-only matching caused false links such as:
        #   Lewis Hall -> Kiernan Dewsbury-Hall
        # and can collide badly for common surnames such as Touré.
        aliases.add(
            f"{parts[0][0]} {parts[-1]}"
        )

    for canonical, variants in PLAYER_ALIASES.items():
        all_names = [
            canonical,
            *variants,
        ]

        normalized = {
            norm(x)
            for x in all_names
        }

        if n in normalized:
            aliases.update(
                normalized
            )

    return aliases


def names_match(
    a: object,
    b: object,
) -> bool:
    aa = name_aliases(
        a
    )
    bb = name_aliases(
        b
    )

    return bool(
        aa
        and bb
        and (aa & bb)
    )


def current_position_override(
    player_name: object,
) -> str | None:
    for canonical, position in CURRENT_POSITION_OVERRIDES.items():
        if names_match(
            player_name,
            canonical,
        ):
            return position

    return None


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


def canonical_position(value: object) -> str | None:
    n = norm(
        value
    )

    if not n:
        return None

    if n in {
        "gk",
        "goalkeeper",
    }:
        return "GK"

    if n in {
        "cb",
        "centre back",
        "center back",
    }:
        return "CB"

    if n in {
        "fb wb",
        "fb",
        "wingback",
        "fullback",
        "left back",
        "right back",
        "lb",
        "rb",
    }:
        return "FB/WB"

    if n in {
        "dm cm",
        "dm",
        "cm",
        "midfield",
        "central midfield",
        "defensive midfield",
    }:
        return "DM/CM"

    if n in {
        "am",
        "cam",
        "attacking midfield",
    }:
        return "AM"

    if n in {
        "w",
        "winger",
        "lw",
        "rw",
        "left winger",
        "right winger",
    }:
        return "W"

    if n in {
        "st",
        "cf",
        "fw",
        "striker",
        "centre forward",
        "center forward",
    }:
        return "ST"

    return None


def weighted_percentile_mean(
    group: pd.DataFrame,
    metric: str,
    weight_cap: int = MINUTES_WEIGHT_CAP,
):
    if metric not in group.columns:
        return np.nan

    valid = group[
        group[metric].notna()
        & group["historical_minutes"].notna()
        & (group["historical_minutes"] > 0)
    ].copy()

    if valid.empty:
        return np.nan

    valid["analysis_weight"] = (
        valid["historical_minutes"]
        .clip(
            lower=0,
            upper=weight_cap,
        )
        .astype(float)
    )

    total_weight = (
        valid["analysis_weight"]
        .sum()
    )

    if total_weight <= 0:
        return np.nan

    return float(
        (
            valid[metric]
            * valid["analysis_weight"]
        ).sum()
        / total_weight
    )


# ======================================================================================
# LOADING
# ======================================================================================

def load_required():
    required = [
        CURRENT_SQUAD_FILE,
        HISTORICAL_BRIDGE_FILE,
        HISTORICAL_PERCENTILES_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    current = pd.read_parquet(
        CURRENT_SQUAD_FILE
    ).copy()

    bridge = pd.read_parquet(
        HISTORICAL_BRIDGE_FILE
    ).copy()

    hist = pd.read_parquet(
        HISTORICAL_PERCENTILES_FILE
    ).copy()

    benchmarks = (
        pd.read_parquet(
            NEWCASTLE_ROLE_BENCHMARKS_FILE
        )
        if NEWCASTLE_ROLE_BENCHMARKS_FILE.exists()
        else pd.DataFrame()
    )

    optimal = (
        pd.read_parquet(
            NEWCASTLE_OPTIMAL_ROLES_FILE
        )
        if NEWCASTLE_OPTIMAL_ROLES_FILE.exists()
        else pd.DataFrame()
    )

    return (
        current,
        bridge,
        hist,
        benchmarks,
        optimal,
    )


# ======================================================================================
# MATCHING HELPERS
# ======================================================================================

def find_bridge_row(
    player_name: str,
    bridge: pd.DataFrame,
):
    if bridge.empty:
        return None

    matches = bridge[
        bridge["player_name"]
        .map(
            lambda x: names_match(
                player_name,
                x,
            )
        )
    ].copy()

    if matches.empty:
        return None

    return matches.iloc[0]


def strict_rule_for_player(
    player_name: str,
) -> dict | None:
    """
    Return a strict historical identity rule for a current Newcastle player.

    Example:
    Bazoumana Touré must resolve to his 2025/26 Bundesliga / Hoffenheim row,
    not another player who happens to share the Touré surname.
    """
    for canonical, rule in STRICT_HISTORICAL_MATCH_RULES.items():
        if names_match(
            player_name,
            canonical,
        ):
            return rule

    return None


def apply_strict_historical_rule(
    player_name: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rule = strict_rule_for_player(
        player_name
    )

    if (
        rule is None
        or candidates.empty
    ):
        return candidates

    filtered = candidates.copy()

    # League is the hard constraint for Bazoumana Touré.
    allowed_leagues = {
        norm(x)
        for x in rule.get(
            "league_slugs",
            set(),
        )
    }

    if allowed_leagues:
        league_match = filtered[
            filtered["league_slug"]
            .astype(str)
            .map(norm)
            .isin(
                allowed_leagues
            )
        ].copy()

        if not league_match.empty:
            filtered = league_match
        else:
            # Strict means strict: never silently link Touré to a non-Bundesliga row.
            return pd.DataFrame(
                columns=candidates.columns
            )

    # Hoffenheim is preferred whenever either historical team or TM club metadata
    # contains it. Some historical rows have team_name=None, so club cannot be
    # treated as a mandatory constraint if that metadata is absent.
    club_terms = {
        norm(x)
        for x in rule.get(
            "club_terms",
            set(),
        )
    }

    if club_terms:
        club_mask = pd.Series(
            False,
            index=filtered.index,
        )

        club_columns = [
            col
            for col in [
                "team_name",
                "current_club_name",
            ]
            if col in filtered.columns
        ]

        for col in club_columns:
            values = (
                filtered[col]
                .fillna("")
                .astype(str)
                .map(norm)
            )

            this_match = values.map(
                lambda value: any(
                    term in value
                    for term in club_terms
                )
            )

            club_mask = (
                club_mask
                | this_match
            )

        hoff = filtered[
            club_mask
        ].copy()

        # Prefer Hoffenheim if metadata identifies it.
        # If all club fields are blank, retain the Bundesliga-filtered candidates.
        if not hoff.empty:
            filtered = hoff

    return filtered


def find_historical_row(
    player_name: str,
    hist: pd.DataFrame,
    bridge_row=None,
):
    # Start from name/alias matching.
    candidates = hist[
        hist["player_name"]
        .map(
            lambda x: names_match(
                player_name,
                x,
            )
        )
    ].copy()

    # If the bridge supplies a historical short name, include those candidates too,
    # but DO NOT let it override a strict player identity rule.
    if (
        bridge_row is not None
        and pd.notna(
            bridge_row.get(
                "historical_player_name"
            )
        )
    ):
        hist_name = bridge_row.get(
            "historical_player_name"
        )

        bridge_candidates = hist[
            hist["player_name"]
            .map(
                lambda x: names_match(
                    hist_name,
                    x,
                )
            )
        ].copy()

        if not bridge_candidates.empty:
            candidates = pd.concat(
                [
                    candidates,
                    bridge_candidates,
                ],
                ignore_index=False,
            ).drop_duplicates()

    # CRITICAL: apply player-specific identity constraints before any bridge league
    # or minutes tie-breaking. Bazoumana Touré must be Bundesliga/Hoffenheim.
    candidates = apply_strict_historical_rule(
        player_name,
        candidates,
    )

    if candidates.empty:
        return None

    strict_rule = strict_rule_for_player(
        player_name
    )

    # For ordinary players, the bridge league remains useful.
    # For strict-rule players, the explicit rule above is authoritative.
    if (
        strict_rule is None
        and bridge_row is not None
    ):
        league_slug = bridge_row.get(
            "historical_league_slug"
        )

        if pd.notna(
            league_slug
        ):
            same_league = candidates[
                candidates["league_slug"]
                .astype(str)
                .map(norm)
                == norm(
                    league_slug
                )
            ].copy()

            if not same_league.empty:
                candidates = same_league

    # Historical minutes are only a tie-breaker after identity has been resolved.
    if bridge_row is not None:
        hist_minutes = pd.to_numeric(
            pd.Series([
                bridge_row.get(
                    "historical_minutes"
                )
            ]),
            errors="coerce",
        ).iloc[0]

        if (
            len(candidates) > 1
            and pd.notna(
                hist_minutes
            )
        ):
            candidates["_minutes_diff"] = (
                pd.to_numeric(
                    candidates["minutes"],
                    errors="coerce",
                )
                - hist_minutes
            ).abs()

            candidates = candidates.sort_values(
                "_minutes_diff"
            )

    if len(candidates) > 1:
        candidates = candidates.sort_values(
            "minutes",
            ascending=False,
        )

    return candidates.iloc[0]


def find_role_fallback(
    player_name: str,
    benchmarks: pd.DataFrame,
    optimal: pd.DataFrame,
) -> dict:
    result = {}

    for df in [
        benchmarks,
        optimal,
    ]:
        if df.empty:
            continue

        player_col = None

        for candidate in [
            "player_name",
            "Player",
            "player",
            "name",
        ]:
            if candidate in df.columns:
                player_col = candidate
                break

        if player_col is None:
            continue

        matches = df[
            df[player_col]
            .map(
                lambda x: names_match(
                    player_name,
                    x,
                )
            )
        ].copy()

        if matches.empty:
            continue

        row = matches.iloc[0]

        for col in df.columns:
            if col == player_col:
                continue

            if col not in result:
                result[col] = row.get(
                    col
                )

    return result


# ======================================================================================
# BUILD DIAGNOSTIC
# ======================================================================================

def build_player_diagnostic(
    current: pd.DataFrame,
    bridge: pd.DataFrame,
    hist: pd.DataFrame,
    benchmarks: pd.DataFrame,
    optimal: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    all_metrics = {
        metric
        for metrics in POSITION_METRICS.values()
        for metric in metrics
    }

    for _, current_row in current.iterrows():
        player_name = current_row.get(
            "player_name"
        )

        bridge_row = find_bridge_row(
            player_name,
            bridge,
        )

        historical_row = find_historical_row(
            player_name,
            hist,
            bridge_row,
        )

        result = {
            "player_name": player_name,
            "current_minutes": current_row.get(
                "minutes"
            ),
            "current_position": current_row.get(
                "position_group"
            ),
            "tactical_position": current_row.get(
                "tactical_position"
            ),
            "position_group_share": current_row.get(
                "position_group_share"
            ),
            "position_side_share": current_row.get(
                "position_side_share"
            ),
            "observed_position_groups": current_row.get(
                "observed_position_groups"
            ),
            "observed_position_sides": current_row.get(
                "observed_position_sides"
            ),
        }

        if historical_row is not None:
            result.update({
                "historical_found": True,
                "historical_player_name": historical_row.get(
                    "player_name"
                ),
                "historical_league": historical_row.get(
                    "league_name"
                ),
                "historical_league_slug": historical_row.get(
                    "league_slug"
                ),
                "historical_minutes": historical_row.get(
                    "minutes"
                ),
                "historical_team_name": historical_row.get(
                    "team_name"
                ),
                "historical_position": historical_row.get(
                    "position_group"
                ),
                "age": historical_row.get(
                    "age_at_2026_07_01"
                ),
                "market_value_eur": historical_row.get(
                    "market_value_eur"
                ),
                "tm_position": historical_row.get(
                    "tm_position"
                ),
                "tm_sub_position": historical_row.get(
                    "tm_sub_position"
                ),
                "foot": historical_row.get(
                    "foot"
                ),
                "height_cm": historical_row.get(
                    "height_cm"
                ),
            })

            for metric in all_metrics:
                if metric in historical_row.index:
                    result[metric] = historical_row.get(
                        metric
                    )

                pct_col = (
                    metric
                    + "_percentile"
                )

                if pct_col in historical_row.index:
                    result[pct_col] = historical_row.get(
                        pct_col
                    )

        else:
            result.update({
                "historical_found": False,
                "historical_player_name": np.nan,
                "historical_league": np.nan,
                "historical_league_slug": np.nan,
                "historical_minutes": np.nan,
                "historical_team_name": np.nan,
                "historical_position": np.nan,
                "age": np.nan,
                "market_value_eur": np.nan,
                "tm_position": np.nan,
                "tm_sub_position": np.nan,
                "foot": np.nan,
                "height_cm": np.nan,
            })

        # Prefer historical role from bridge when available.
        if bridge_row is not None:
            for col in [
                "vertical_fullback_score",
                "controller_midfielder_score",
                "creative_cam_score",
                "vertical_winger_score",
                "toney_striker_score",
                "best_2025_26_role",
                "best_2025_26_score",
                "second_2025_26_role",
                "second_2025_26_score",
            ]:
                if col in bridge_row.index:
                    result[col] = bridge_row.get(
                        col
                    )

        # Club-specific fallback metadata for players absent from bridge.
        fallback = find_role_fallback(
            player_name,
            benchmarks,
            optimal,
        )

        for key, value in fallback.items():
            safe_key = (
                "fallback_"
                + str(key)
            )

            if safe_key not in result:
                result[safe_key] = value

        # Current squad position is master for display.
        result["position_group"] = (
            current_position_override(
                player_name
            )
            or canonical_position(
                current_row.get(
                    "position_group"
                )
            )
            or canonical_position(
                current_row.get(
                    "tactical_position"
                )
            )
            or canonical_position(
                result.get(
                    "historical_position"
                )
            )
        )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# UNIT SUMMARY
# ======================================================================================

def build_unit_summary(
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for position_group in POSITION_ORDER:
        group = diagnostic[
            diagnostic[
                "position_group"
            ]
            == position_group
        ].copy()

        if group.empty:
            continue

        historical_group = group[
            group[
                "historical_found"
            ]
            == True
        ].copy()

        depth_counts = {
            label: int(
                (
                    historical_group[
                        "historical_minutes"
                    ]
                    .fillna(0)
                    >= threshold
                ).sum()
            )
            for label, threshold
            in DEPTH_THRESHOLDS.items()
        }

        base = {
            "position_group": position_group,
            "current_players": len(
                group
            ),
            "historical_players_found": len(
                historical_group
            ),
            "total_historical_minutes": int(
                historical_group[
                    "historical_minutes"
                ]
                .fillna(0)
                .sum()
            ),
            **depth_counts,
        }

        metrics = POSITION_METRICS.get(
            position_group,
            [],
        )

        if not metrics:
            rows.append({
                **base,
                "metric": None,
                "metric_label": None,
                "weighted_percentile": np.nan,
                "band": "N/A",
            })
            continue

        for metric in metrics:
            pct_col = (
                metric
                + "_percentile"
            )

            if pct_col not in historical_group.columns:
                continue

            usable_group = historical_group.copy()

            if metric in usable_group.columns:
                placeholder_mask = (
                    pd.to_numeric(
                        usable_group[metric],
                        errors="coerce",
                    ).fillna(np.nan).abs() < 1e-12
                ) & (
                    pd.to_numeric(
                        usable_group[pct_col],
                        errors="coerce",
                    ).between(
                        49.0,
                        52.0,
                        inclusive="both",
                    )
                )

                usable_group.loc[
                    placeholder_mask,
                    pct_col,
                ] = np.nan

            value = weighted_percentile_mean(
                usable_group,
                pct_col,
            )

            rows.append({
                **base,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(
                    metric,
                    metric,
                ),
                "weighted_percentile": (
                    round(
                        value,
                        1,
                    )
                    if pd.notna(
                        value
                    )
                    else np.nan
                ),
                "band": percentile_band(
                    value
                ),
            })

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# PRINTING
# ======================================================================================

def metric_is_informative(
    row: pd.Series,
    metric: str,
    pct_col: str,
) -> bool:
    """
    Avoid presenting upstream missing-data placeholders as real football output.

    In this dataset, many unavailable metrics were materialized as:
      raw/per90 = 0.0
      percentile ~= 50-51

    That is not evidence of average performance; it is a missing-data artifact.
    """
    if (
        metric not in row.index
        or pct_col not in row.index
        or pd.isna(row.get(metric))
        or pd.isna(row.get(pct_col))
    ):
        return False

    value = float(row.get(metric))
    pct = float(row.get(pct_col))

    # Suppress the classic missing-data placeholder pattern.
    if abs(value) < 1e-12 and 49.0 <= pct <= 52.0:
        return False

    return True


def format_metric_value(
    metric: str,
    value,
) -> str:
    if pd.isna(
        value
    ):
        return "N/A"

    if metric.endswith(
        "_pct"
    ):
        return (
            f"{float(value):.1f}%"
        )

    return (
        f"{float(value):.2f}/90"
    )


def print_player_card(
    row: pd.Series,
) -> None:
    print(
        f"\n{row['player_name']}"
    )

    print(
        f"  Current position: "
        f"{row.get('position_group')}"
    )

    if pd.notna(
        row.get(
            "current_minutes"
        )
    ):
        print(
            f"  Current sample minutes: "
            f"{int(row['current_minutes'])}"
        )

    if not bool(
        row.get(
            "historical_found",
            False,
        )
    ):
        print(
            "  Historical 2025/26 match: NOT FOUND"
        )

        fallback_cols = [
            col
            for col in row.index
            if str(col).startswith(
                "fallback_"
            )
            and pd.notna(
                row.get(
                    col
                )
            )
        ]

        if fallback_cols:
            print(
                "  Newcastle-specific fallback metadata available."
            )

        return

    print(
        f"  Historical player: "
        f"{row.get('historical_player_name')}"
    )

    print(
        f"  Historical minutes: "
        f"{int(row.get('historical_minutes', 0))}"
    )

    print(
        f"  Historical league: "
        f"{row.get('historical_league')}"
    )

    if names_match(
        row.get(
            "player_name"
        ),
        "Bazoumana Touré",
    ):
        club_value = row.get(
            "historical_team_name"
        )

        if (
            club_value is None
            or pd.isna(club_value)
            or str(club_value).strip().lower() in {
                "",
                "nan",
                "none",
            }
        ):
            club_value = "Hoffenheim metadata unavailable in historical row"

        print(
            f"  Historical club check: "
            f"{club_value}"
        )
        print(
            "  Identity constraint: Bundesliga + Hoffenheim preferred"
        )

    if pd.notna(
        row.get(
            "age"
        )
    ):
        print(
            f"  Age: "
            f"{float(row['age']):.0f}"
        )

    if pd.notna(
        row.get(
            "best_2025_26_role"
        )
    ):
        print(
            f"  Best 2025/26 role: "
            f"{row['best_2025_26_role']} "
            f"({row.get('best_2025_26_score', np.nan):.1f})"
        )

    metrics = POSITION_METRICS.get(
        row.get(
            "position_group"
        ),
        [],
    )

    valid_metrics = []

    for metric in metrics:
        pct_col = (
            metric
            + "_percentile"
        )

        if metric_is_informative(
            row,
            metric,
            pct_col,
        ):
            valid_metrics.append(
                (
                    metric,
                    pct_col,
                )
            )

    if not valid_metrics:
        print(
            "  No configured historical metrics found."
        )
        return

    print(
        "  2025/26 historical metrics:"
    )

    for metric, pct_col in valid_metrics:
        label = METRIC_LABELS.get(
            metric,
            metric,
        )

        value = format_metric_value(
            metric,
            row[metric],
        )

        pct = float(
            row[pct_col]
        )

        print(
            f"    {label:<38}"
            f"{value:>11}"
            f"   pct {pct:>5.1f}"
            f"  {percentile_band(pct)}"
        )


def print_unit_summary(
    group_summary: pd.DataFrame,
) -> None:
    if group_summary.empty:
        return

    first = group_summary.iloc[0]

    print()
    print(
        "  UNIT PROFILE — FULL 2025/26"
    )

    metric_rows = group_summary[
        group_summary[
            "metric"
        ].notna()
    ].copy()

    if metric_rows.empty:
        print(
            "    No configured unit percentile metrics."
        )
    else:
        metric_rows = metric_rows.sort_values(
            "weighted_percentile",
            ascending=False,
        )

        for _, row in metric_rows.iterrows():
            print(
                f"    {row['metric_label']:<38}"
                f"{row['weighted_percentile']:>6.1f}"
                f"  {row['band']}"
            )

    print()
    print(
        f"  Current squad players: "
        f"{int(first['current_players'])}"
    )

    print(
        f"  Historical matches found: "
        f"{int(first['historical_players_found'])}"
    )

    print(
        f"  Total historical minutes: "
        f"{int(first['total_historical_minutes'])}"
    )


def print_squad_diagnostic(
    diagnostic: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 110
    )

    print(
        "NEWCASTLE UNITED — CURRENT SQUAD WITH FULL 2025/26 HISTORICAL CONTEXT"
        .center(
            110
        )
    )

    print(
        "=" * 110
    )

    for position_group in POSITION_ORDER:
        group = diagnostic[
            diagnostic[
                "position_group"
            ]
            == position_group
        ].copy()

        if group.empty:
            continue

        group = group.sort_values(
            [
                "historical_found",
                "historical_minutes",
                "current_minutes",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )

        print()
        print(
            "=" * 95
        )

        print(
            f"{position_group}"
        )

        print(
            "=" * 95
        )

        for _, row in group.iterrows():
            print_player_card(
                row
            )

            print(
                "-" * 95
            )

        print_unit_summary(
            unit_summary[
                unit_summary[
                    "position_group"
                ]
                == position_group
            ]
        )


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    (
        current,
        bridge,
        hist,
        benchmarks,
        optimal,
    ) = load_required()

    diagnostic = build_player_diagnostic(
        current,
        bridge,
        hist,
        benchmarks,
        optimal,
    )

    unit_summary = build_unit_summary(
        diagnostic
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic.to_parquet(
        PLAYER_OUTPUT_FILE,
        index=False,
    )

    diagnostic.to_csv(
        PLAYER_OUTPUT_CSV,
        index=False,
    )

    unit_summary.to_parquet(
        UNIT_OUTPUT_FILE,
        index=False,
    )

    unit_summary.to_csv(
        UNIT_OUTPUT_CSV,
        index=False,
    )

    print_squad_diagnostic(
        diagnostic,
        unit_summary,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "SAVED OUTPUTS"
        .center(
            110
        )
    )
    print(
        "=" * 110
    )

    print(
        PLAYER_OUTPUT_FILE
    )
    print(
        PLAYER_OUTPUT_CSV
    )
    print(
        UNIT_OUTPUT_FILE
    )
    print(
        UNIT_OUTPUT_CSV
    )

    matched = int(
        diagnostic[
            "historical_found"
        ]
        .fillna(
            False
        )
        .sum()
    )

    print()
    print(
        f"Current squad players: "
        f"{len(diagnostic)}"
    )

    print(
        f"Historical 2025/26 matches found: "
        f"{matched}"
    )

    missing = diagnostic[
        ~diagnostic[
            "historical_found"
        ]
        .fillna(
            False
        )
    ][
        "player_name"
    ].tolist()

    if missing:
        print()
        print(
            "Players still missing a 2025/26 historical row:"
        )

        for player in missing:
            print(
                f"  - {player}"
            )

    print()
    print(
        "IMPORTANT: Current Newcastle squad membership comes from "
        "newcastle_player_diagnostic.parquet, so players such as "
        "Bazoumana Touré are never dropped merely because the historical bridge missed them."
    )


if __name__ == "__main__":
    main()
