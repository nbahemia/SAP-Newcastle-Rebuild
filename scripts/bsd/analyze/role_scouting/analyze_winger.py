from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

FULL_SHORTLIST = (
    ROOT / "data" / "processed" / "bsd" / "shortlist"
    / "europe_2025_26_full_role_shortlist.parquet"
)

HISTORICAL_METRICS = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_season_metrics.parquet"
)

OUT_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"
IMAGE_DIR = OUT_DIR / "images"

OUT_CSV = OUT_DIR / "winger_jaissle_archetypes.csv"
OUT_HTML = OUT_DIR / "winger_jaissle_archetypes.html"
OUT_CHART = IMAGE_DIR / "winger_jaissle_archetypes_chart.png"

MIN_MINUTES = 450


# =============================================================================
# METRIC ALIASES
# Optional metrics are used when the historical parquet contains them.
# The scoring functions automatically re-normalize weights when a metric is
# unavailable, so the script remains usable with the current BSD history.
# =============================================================================

ALIASES = {
    # Shooting / box threat
    "xg": ["xg_per90", "xg", "expected_goals"],
    "shots": ["shots_per90", "shots", "total_shots_per90", "total_shots"],
    "shots_on_target": [
        "shots_on_target_per90", "shots_on_target"
    ],
    "box_touches": [
        "box_touches_per90",
        "touches_in_opposition_box_per90",
        "touches_in_box_per90",
        "touches_in_opposition_box",
        "box_touches",
    ],

    # Creation / progression
    "xa": ["xa_per90", "xa", "expected_assists"],
    "key_passes": [
        "key_passes_per90", "key_passes", "key_pass"
    ],
    "big_chances_created": [
        "big_chances_created_per90",
        "big_chance_created_per90",
        "big_chances_created",
        "big_chance_created",
    ],
    "line_breaking_passes": [
        "line_breaking_passes_per90",
        "line_breaking_passes",
        "line_break_passes_per90",
    ],
    "accurate_crosses": [
        "accurate_crosses_per90", "accurate_crosses",
        "successful_crosses_per90", "successful_crosses",
    ],
    "successful_passes": [
        "successful_passes_per90", "successful_passes",
        "accurate_passes_per90", "accurate_passes",
    ],
    "total_passes": [
        "total_passes_per90", "total_passes"
    ],
    "pass_accuracy": [
        "pass_accuracy_pct", "successful_passes_pct",
        "successful_passes_percentage",
    ],
    "touches": ["touches_per90", "touches"],

    # Carrying / 1v1
    "successful_dribbles": [
        "successful_dribbles_per90",
        "successful_dribbles",
        "won_contest_per90",
        "won_contest",
    ],
    "progressive_carries": [
        "progressive_carries_per90",
        "progressive_carries",
        "progressive_ball_carries_count_per90",
        "progressive_ball_carries_count",
    ],

    # Physical / defensive
    "duels_won": ["duels_won_per90", "duels_won", "duel_won"],
    "fouls_won": [
        "fouls_won_per90", "fouls_won"
    ],
    "recoveries": [
        "recoveries_per90", "recoveries", "ball_recovery"
    ],
    "possession_won_final_third": [
        "possession_won_final_third_per90",
        "possession_won_final_third",
        "final_third_wins_per90",
        "final_third_wins",
    ],
    "tackles_won": [
        "tackles_won_per90", "tackles_won", "won_tackle"
    ],
    "interceptions": [
        "interceptions_per90", "interceptions", "interception"
    ],
}


NEWCASTLE_BENCHMARK_ALIASES = {
    "Elanga": ["a elanga", "anthony elanga"],
    "Barnes": ["h barnes", "harvey barnes"],
    "Murphy": ["j murphy", "jacob murphy"],
    "Toure": [
        "b toure", "bazoumana toure",
        "b touré", "bazoumana touré",
    ],
    "Fernandez-Pardo": [
        "m fernandez",
        "m fernandez pardo",
        "matias fernandez pardo",
        "matias fernández-pardo",
    ],
}


# =============================================================================
# JAISSLE REFERENCE PROFILES
# User-supplied FotMob season profiles.
# These are descriptive anchors only; they are NOT inserted into the
# league-normalized candidate score.
# =============================================================================

JAISSLE_REFERENCES = {
    "Mahrez-type Wide Creator": {
        "minutes": 2311,
        "goals_per90": 0.16,
        "xg_per90": 0.14,
        "shots_per90": 1.48,
        "shots_on_target_per90": 0.39,
        "assists_per90": 0.31,
        "xa_per90": 0.37,
        "successful_passes_per90": 28.90,
        "pass_accuracy_pct": 85.0,
        "line_breaking_passes_per90": 7.01,
        "key_passes_per90": 2.96,
        "big_chances_created_per90": 0.78,
        "accurate_crosses_per90": 2.18,
        "successful_dribbles_per90": 1.01,
        "duels_won_per90": 3.23,
        "touches_per90": 52.81,
        "box_touches_per90": 3.70,
        "recoveries_per90": 3.78,
        "possession_won_final_third_per90": 0.90,
        "description": (
            "Wide creator: high xA, chance creation, crossing, "
            "line-breaking distribution and passing involvement."
        ),
    },
    "Galeno-type Vertical / Box Winger": {
        "minutes": 1821,
        "goals_per90": 0.05,
        "xg_per90": 0.13,
        "shots_per90": 2.03,
        "shots_on_target_per90": 0.74,
        "assists_per90": 0.35,
        "xa_per90": 0.34,
        "successful_passes_per90": 14.48,
        "pass_accuracy_pct": 70.6,
        "line_breaking_passes_per90": 4.84,
        "key_passes_per90": 1.24,
        "big_chances_created_per90": 0.40,
        "accurate_crosses_per90": 0.74,
        "successful_dribbles_per90": 0.79,
        "duels_won_per90": 4.05,
        "touches_per90": 44.04,
        "box_touches_per90": 5.34,
        "fouls_won_per90": 1.78,
        "recoveries_per90": 3.66,
        "possession_won_final_third_per90": 0.69,
        "interceptions_per90": 1.24,
        "description": (
            "Vertical/box winger: lower circulation volume, more box presence, "
            "shooting, duels and defensive activity."
        ),
    },
}


# =============================================================================
# ARCHETYPE WEIGHTS
# =============================================================================

WIDE_CREATOR_WEIGHTS = {
    "xa_p90": 0.20,
    "key_passes_p90": 0.18,
    "big_chances_created_p90": 0.12,
    "line_breaking_passes_p90": 0.15,
    "accurate_crosses_p90": 0.10,
    "successful_passes_p90": 0.10,
    "pass_accuracy": 0.05,
    "touches_p90": 0.05,
    "recoveries_p90": 0.05,
}

VERTICAL_BOX_WEIGHTS = {
    "xg_p90": 0.18,
    "shots_p90": 0.12,
    "shots_on_target_p90": 0.10,
    "box_touches_p90": 0.15,
    "successful_dribbles_p90": 0.12,
    "duels_won_p90": 0.08,
    "fouls_won_p90": 0.05,
    "recoveries_p90": 0.07,
    "possession_won_final_third_p90": 0.06,
    "xa_p90": 0.07,
}

HYBRID_WEIGHTS = {
    "xa_p90": 0.13,
    "key_passes_p90": 0.10,
    "xg_p90": 0.12,
    "shots_p90": 0.08,
    "shots_on_target_p90": 0.06,
    "box_touches_p90": 0.08,
    "successful_dribbles_p90": 0.08,
    "accurate_crosses_p90": 0.06,
    "successful_passes_p90": 0.07,
    "touches_p90": 0.05,
    "recoveries_p90": 0.07,
    "duels_won_p90": 0.05,
    "possession_won_final_third_p90": 0.05,
}


# =============================================================================
# HELPERS
# =============================================================================

def norm(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    ).strip()


def find_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(c).lower(),
        ).strip("_"): c
        for c in df.columns
    }

    for alias in aliases:
        key = re.sub(
            r"[^a-z0-9]+",
            "_",
            alias.lower(),
        ).strip("_")
        if key in lookup:
            return lookup[key]

    return None


def p90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    minutes = pd.to_numeric(minutes, errors="coerce")

    return pd.Series(
        np.where(
            minutes > 0,
            values * 90 / minutes,
            np.nan,
        ),
        index=values.index,
        dtype=float,
    )


def winsor_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if len(valid) < 8:
        return pd.Series(
            np.nan,
            index=s.index,
            dtype=float,
        )

    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)
    clipped = s.clip(lo, hi)

    sd = clipped.std(ddof=0)

    if pd.isna(sd) or sd <= 1e-9:
        return pd.Series(
            0.0,
            index=s.index,
            dtype=float,
        )

    return (
        clipped - clipped.mean()
    ) / sd


def league_adjusted_z(
    df: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """
    Standardize a winger against other winger candidates in the same league.

    If a league group is too small, fall back to the full winger pool.
    """
    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if metric not in df.columns:
        return result

    global_z = winsor_z(df[metric])

    if "league_slug" not in df.columns:
        return global_z

    for league, idx in df.groupby(
        "league_slug",
        dropna=False,
    ).groups.items():
        league_idx = list(idx)

        if len(league_idx) < 8:
            result.loc[league_idx] = global_z.loc[
                league_idx
            ]
            continue

        result.loc[league_idx] = winsor_z(
            df.loc[league_idx, metric]
        )

    return result


def score_index(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    """
    50 = league-winger average.
    60 ~= one weighted SD above league-winger average.

    Returns score and metric coverage.
    """
    weighted = pd.Series(
        0.0,
        index=df.index,
    )
    observed = pd.Series(
        0.0,
        index=df.index,
    )

    total_weight = sum(weights.values())

    for col, weight in weights.items():
        if col not in df.columns:
            continue

        z = league_adjusted_z(
            df,
            col,
        )

        valid = z.notna()

        weighted.loc[valid] += (
            z.loc[valid] * weight
        )

        observed.loc[valid] += weight

    normalized = pd.Series(
        np.where(
            observed > 0,
            weighted / observed,
            np.nan,
        ),
        index=df.index,
    )

    score = (
        50 + 10 * normalized
    ).clip(20, 80)

    coverage = (
        observed / total_weight * 100
        if total_weight > 0
        else 0
    )

    return score, coverage


def player_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["player_name"]
        .astype(str)
        .map(norm)
        + "|"
        + df["league_slug"]
        .astype(str)
    )


def is_winger(row) -> bool:
    text = " ".join(
        norm(row.get(c, ""))
        for c in [
            "natural_position",
            "tm_sub_position",
            "tm_position",
        ]
    )

    return any(
        term in text
        for term in [
            "left winger",
            "right winger",
            "winger",
            "left midfield",
            "right midfield",
            "attacking midfield",
        ]
    )


def benchmark_label(name: str) -> str | None:
    n = norm(name)

    for label, aliases in (
        NEWCASTLE_BENCHMARK_ALIASES.items()
    ):
        for alias in aliases:
            a = norm(alias)
            if (
                n == a
                or n.startswith(a + " ")
                or a.startswith(n + " ")
            ):
                return label

    return None


def top_with_benchmarks(
    df: pd.DataFrame,
    score_col: str,
    n: int = 30,
) -> pd.DataFrame:
    top = (
        df.sort_values(
            score_col,
            ascending=False,
        )
        .head(n)
    )

    benches = df[
        df["is_newcastle_benchmark"]
    ].copy()

    combined = pd.concat(
        [top, benches],
        ignore_index=False,
    )

    combined = combined[
        ~combined.index.duplicated(
            keep="first"
        )
    ]

    return combined.sort_values(
        score_col,
        ascending=False,
    )


def best_two_archetypes(row: pd.Series):
    values = {
        "Wide Creator": row.get(
            "wide_creator_score"
        ),
        "Vertical / Box Winger": row.get(
            "vertical_box_score"
        ),
        "Hybrid Winger": row.get(
            "hybrid_score"
        ),
    }

    values = {
        key: value
        for key, value in values.items()
        if pd.notna(value)
    }

    if not values:
        return (
            pd.NA,
            np.nan,
            pd.NA,
            np.nan,
        )

    ordered = sorted(
        values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_name, best_score = ordered[0]

    if len(ordered) > 1:
        second_name, second_score = ordered[1]
    else:
        second_name, second_score = (
            pd.NA,
            np.nan,
        )

    return (
        best_name,
        best_score,
        second_name,
        second_score,
    )


def strongest_dimensions(
    row: pd.Series,
) -> str:
    dimensions = {
        "Creation": row.get(
            "wide_creator_score"
        ),
        "Direct/Box Threat": row.get(
            "vertical_box_score"
        ),
        "Balanced Hybrid": row.get(
            "hybrid_score"
        ),
    }

    valid = [
        (name, float(value))
        for name, value in dimensions.items()
        if pd.notna(value)
    ]

    if not valid:
        return ""

    valid.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return "; ".join(
        f"{name} {value:.1f}"
        for name, value in valid
    )


# =============================================================================
# LOAD / BUILD WINGER POOL
# =============================================================================

def load_pool() -> pd.DataFrame:
    if not FULL_SHORTLIST.exists():
        raise FileNotFoundError(
            FULL_SHORTLIST
        )

    if not HISTORICAL_METRICS.exists():
        raise FileNotFoundError(
            HISTORICAL_METRICS
        )

    shortlist = pd.read_parquet(
        FULL_SHORTLIST
    ).copy()

    historical = pd.read_parquet(
        HISTORICAL_METRICS
    ).copy()

    # Use any natural winger / wide attacking-midfielder appearing in the
    # recruitment shortlist, not only players who survived the old
    # vertical-winger role filter.
    wing = shortlist[
        shortlist.apply(
            is_winger,
            axis=1,
        )
    ].copy()

    # A player may appear once per role in the long shortlist.
    # Keep one metadata row per player/league.
    wing["_player_key"] = player_key(
        wing
    )

    if "minutes" in wing.columns:
        wing["_sort_minutes"] = pd.to_numeric(
            wing["minutes"],
            errors="coerce",
        ).fillna(0)
    else:
        wing["_sort_minutes"] = 0

    wing = (
        wing.sort_values(
            "_sort_minutes",
            ascending=False,
        )
        .drop_duplicates(
            "_player_key",
            keep="first",
        )
        .copy()
    )

    historical["_player_key"] = player_key(
        historical
    )

    hist_minutes = (
        pd.to_numeric(
            historical["minutes"],
            errors="coerce",
        )
        if "minutes" in historical.columns
        else pd.Series(
            np.nan,
            index=historical.index,
        )
    )

    source = pd.DataFrame({
        "_player_key": historical[
            "_player_key"
        ],
        "_src_minutes": hist_minutes,
    })

    detected = {}

    for metric, aliases in ALIASES.items():
        col = find_alias(
            historical,
            aliases,
        )

        detected[metric] = col

        if col is not None:
            source[
                f"_src_{metric}"
            ] = pd.to_numeric(
                historical[col],
                errors="coerce",
            )

    source = source.drop_duplicates(
        "_player_key",
        keep="first",
    )

    wing = wing.merge(
        source,
        on="_player_key",
        how="left",
    )

    # Require meaningful historical sample.
    wing = wing[
        pd.to_numeric(
            wing["_src_minutes"],
            errors="coerce",
        ).fillna(0)
        >= MIN_MINUTES
    ].copy()

    print("=" * 120)
    print(
        "WINGER ARCHETYPE METRIC SOURCES"
    )
    print("=" * 120)

    for metric, col in detected.items():
        print(
            f"{metric:<30} "
            f"{col if col else 'MISSING'}"
        )

    for metric, source_col in (
        detected.items()
    ):
        out_col = f"{metric}_p90"

        if metric == "pass_accuracy":
            out_col = "pass_accuracy"

        if source_col is None:
            wing[out_col] = np.nan
            continue

        raw = pd.to_numeric(
            wing[f"_src_{metric}"],
            errors="coerce",
        )

        source_name = source_col.lower()

        if (
            "per90" in source_name
            or metric == "pass_accuracy"
        ):
            wing[out_col] = raw
        else:
            wing[out_col] = p90(
                raw,
                wing["_src_minutes"],
            )

    # -------------------------------------------------------------------------
    # Archetype scores
    # -------------------------------------------------------------------------

    (
        wing["wide_creator_score"],
        wing["wide_creator_coverage"],
    ) = score_index(
        wing,
        WIDE_CREATOR_WEIGHTS,
    )

    (
        wing["vertical_box_score"],
        wing["vertical_box_coverage"],
    ) = score_index(
        wing,
        VERTICAL_BOX_WEIGHTS,
    )

    (
        wing["hybrid_score"],
        wing["hybrid_coverage"],
    ) = score_index(
        wing,
        HYBRID_WEIGHTS,
    )

    best = wing.apply(
        best_two_archetypes,
        axis=1,
        result_type="expand",
    )

    best.columns = [
        "best_winger_archetype",
        "best_winger_score",
        "second_winger_archetype",
        "second_winger_score",
    ]

    wing = pd.concat(
        [wing, best],
        axis=1,
    )

    wing["archetype_gap"] = (
        wing["best_winger_score"]
        - wing["second_winger_score"]
    )

    wing["benchmark_label"] = (
        wing["player_name"]
        .astype(str)
        .map(benchmark_label)
    )

    wing["is_newcastle_benchmark"] = (
        wing["benchmark_label"]
        .notna()
    )

    wing["profile_strengths"] = (
        wing.apply(
            strongest_dimensions,
            axis=1,
        )
    )

    # Old role fit is retained only as context.
    if "role_fit" in wing.columns:
        wing["legacy_winger_fit"] = (
            pd.to_numeric(
                wing["role_fit"],
                errors="coerce",
            )
        )

    wing = wing.sort_values(
        [
            "best_winger_score",
            "wide_creator_score",
            "vertical_box_score",
        ],
        ascending=False,
    )

    return wing


# =============================================================================
# OUTPUTS
# =============================================================================

def build_chart(
    df: pd.DataFrame,
    top_n: int = 35,
):
    chart = top_with_benchmarks(
        df.dropna(
            subset=[
                "wide_creator_score",
                "vertical_box_score",
                "best_winger_score",
            ]
        ),
        "best_winger_score",
        top_n,
    ).copy()

    if chart.empty:
        return

    bubble = (
        np.clip(
            chart[
                "hybrid_score"
            ].fillna(50) - 25,
            5,
            None,
        )
        * 13
    )

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    scatter = ax.scatter(
        chart["wide_creator_score"],
        chart["vertical_box_score"],
        s=bubble,
        c=chart["best_winger_score"],
        alpha=0.78,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.015,
    )

    cbar.set_label(
        "Best archetype score"
    )

    for _, row in chart.iterrows():
        label = str(
            row["player_name"]
        )

        if row[
            "is_newcastle_benchmark"
        ]:
            label += " [NUFC]"

        ax.annotate(
            (
                f"{label}\n"
                f"{row['best_winger_archetype']} "
                f"{row['best_winger_score']:.1f}"
            ),
            (
                row["wide_creator_score"],
                row["vertical_box_score"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Wide Creator Score"
    )

    ax.set_ylabel(
        "Vertical / Box Winger Score"
    )

    ax.set_title(
        "Jaissle Winger Archetypes\n"
        "Bubble size = Hybrid Score"
    )

    fig.text(
        0.01,
        0.025,
        (
            "50 = league-winger average. "
            "Players are judged by their strongest winger archetype, "
            "not by one universal winger profile."
        ),
        fontsize=8,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    fig.savefig(
        OUT_CHART,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def reference_html() -> str:
    blocks = []

    for name, ref in (
        JAISSLE_REFERENCES.items()
    ):
        blocks.append(
            f"""
<div class="reference">
<h3>{name}</h3>
<p>{ref["description"]}</p>
<table>
<tr><th>xG/90</th><th>xA/90</th><th>Shots/90</th><th>Key Passes/90</th><th>Crosses/90</th><th>Passes/90</th><th>Box Touches/90</th><th>Recoveries/90</th></tr>
<tr>
<td>{ref.get("xg_per90", "")}</td>
<td>{ref.get("xa_per90", "")}</td>
<td>{ref.get("shots_per90", "")}</td>
<td>{ref.get("key_passes_per90", "")}</td>
<td>{ref.get("accurate_crosses_per90", "")}</td>
<td>{ref.get("successful_passes_per90", "")}</td>
<td>{ref.get("box_touches_per90", "")}</td>
<td>{ref.get("recoveries_per90", "")}</td>
</tr>
</table>
</div>
"""
        )

    return "\n".join(blocks)


def save_html(
    df: pd.DataFrame,
):
    cols = [
        "player_name",
        "club_name",
        "league_slug",
        "age",
        "benchmark_label",
        "best_winger_archetype",
        "best_winger_score",
        "second_winger_archetype",
        "second_winger_score",
        "archetype_gap",
        "wide_creator_score",
        "wide_creator_coverage",
        "vertical_box_score",
        "vertical_box_coverage",
        "hybrid_score",
        "hybrid_coverage",
        "profile_strengths",
        "xa_p90",
        "key_passes_p90",
        "big_chances_created_p90",
        "line_breaking_passes_p90",
        "accurate_crosses_p90",
        "successful_passes_p90",
        "xg_p90",
        "shots_p90",
        "shots_on_target_p90",
        "box_touches_p90",
        "successful_dribbles_p90",
        "recoveries_p90",
        "duels_won_p90",
        "market_value_eur_m",
    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    shown = top_with_benchmarks(
        df,
        "best_winger_score",
        100,
    )

    table = shown[
        cols
    ].to_html(
        index=False,
        float_format=lambda x: (
            f"{x:.2f}"
        ),
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jaissle Winger Archetypes</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 28px;
    background: #f4f6f7;
    color: #17202a;
}}
.card, .reference {{
    background: white;
    padding: 16px;
    margin-bottom: 18px;
    border-left: 4px solid #34495e;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    font-size: 12px;
    margin-bottom: 18px;
}}
th {{
    background: #17202a;
    color: white;
    padding: 8px;
}}
td {{
    padding: 7px;
    border-bottom: 1px solid #ddd;
}}
</style>
</head>
<body>

<h1>Jaissle Winger Archetype Recruitment</h1>

<div class="card">
<strong>Model change:</strong> wingers are no longer forced into one
vertical-transition profile.<br><br>

<strong>Wide Creator:</strong> progression and final-ball creation from wide
areas, inspired by the Mahrez-type profile.<br>

<strong>Vertical / Box Winger:</strong> shooting, box occupation, 1v1/direct
threat and defensive activity, inspired by the Galeno-type profile.<br>

<strong>Hybrid:</strong> balanced contribution across creation, direct threat
and counterpress activity.<br><br>

Scores are standardized against other winger candidates in the same league.
50 is approximately league-winger average. The player's highest archetype
score is used as the primary winger evaluation.
</div>

<h2>Jaissle Reference Profiles</h2>
{reference_html()}

<h2>Recruitment / Newcastle Comparison</h2>
{table}

</body>
</html>
"""

    OUT_HTML.write_text(
        doc,
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = load_pool()

    export_cols = [
        "player_name",
        "club_name",
        "league_slug",
        "age",
        "minutes",
        "benchmark_label",
        "is_newcastle_benchmark",
        "best_winger_archetype",
        "best_winger_score",
        "second_winger_archetype",
        "second_winger_score",
        "archetype_gap",
        "wide_creator_score",
        "wide_creator_coverage",
        "vertical_box_score",
        "vertical_box_coverage",
        "hybrid_score",
        "hybrid_coverage",
        "profile_strengths",
        "legacy_winger_fit",
        "xg_p90",
        "xa_p90",
        "shots_p90",
        "shots_on_target_p90",
        "box_touches_p90",
        "key_passes_p90",
        "big_chances_created_p90",
        "line_breaking_passes_p90",
        "accurate_crosses_p90",
        "successful_passes_p90",
        "pass_accuracy",
        "touches_p90",
        "successful_dribbles_p90",
        "progressive_carries_p90",
        "duels_won_p90",
        "fouls_won_p90",
        "recoveries_p90",
        "possession_won_final_third_p90",
        "tackles_won_p90",
        "interceptions_p90",
        "market_value_eur_m",
        "natural_position",
        "tm_sub_position",
    ]

    export_cols = [
        c for c in export_cols
        if c in result.columns
    ]

    result[
        export_cols
    ].to_csv(
        OUT_CSV,
        index=False,
    )

    save_html(
        result
    )

    build_chart(
        result
    )

    print()
    print("=" * 165)
    print(
        "TOP JAISSLE WINGER ARCHETYPES + NEWCASTLE BENCHMARKS"
    )
    print("=" * 165)

    display_cols = [
        "player_name",
        "club_name",
        "age",
        "benchmark_label",
        "best_winger_archetype",
        "best_winger_score",
        "wide_creator_score",
        "vertical_box_score",
        "hybrid_score",
        "xa_p90",
        "key_passes_p90",
        "xg_p90",
        "shots_p90",
        "box_touches_p90",
        "recoveries_p90",
        "market_value_eur_m",
    ]

    display_cols = [
        c for c in display_cols
        if c in result.columns
    ]

    display = top_with_benchmarks(
        result,
        "best_winger_score",
        35,
    )[display_cols].copy()

    display = display.rename(
        columns={
            "player_name": "Player",
            "club_name": "Club",
            "age": "Age",
            "benchmark_label": "NUFC",
            "best_winger_archetype": "Best Archetype",
            "best_winger_score": "Best Score",
            "wide_creator_score": "Creator",
            "vertical_box_score": "Vertical/Box",
            "hybrid_score": "Hybrid",
            "xa_p90": "xA/90",
            "key_passes_p90": "Key Passes/90",
            "xg_p90": "xG/90",
            "shots_p90": "Shots/90",
            "box_touches_p90": "Box Touches/90",
            "recoveries_p90": "Recoveries/90",
            "market_value_eur_m": "Value €M",
        }
    )

    numeric_exceptions = {
        "Player",
        "Club",
        "NUFC",
        "Best Archetype",
    }

    for col in display.columns:
        if col not in numeric_exceptions:
            display[col] = pd.to_numeric(
                display[col],
                errors="coerce",
            ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("Reference interpretation:")
    print(
        "  Mahrez-type = wide creator / progression / final ball."
    )
    print(
        "  Galeno-type = vertical / box occupation / physical-direct threat."
    )
    print(
        "  Hybrid = balanced winger who can contribute to both phases."
    )

    print()
    print("Saved:")
    print(" ", OUT_CSV)
    print(" ", OUT_HTML)
    print(" ", OUT_CHART)


if __name__ == "__main__":
    main()
