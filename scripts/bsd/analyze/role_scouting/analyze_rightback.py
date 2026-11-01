from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]

FULL_SHORTLIST = (
    ROOT / "data" / "processed" / "bsd" / "shortlist"
    / "europe_2025_26_full_role_shortlist.parquet"
)

ENRICHED = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_percentiles_enriched.parquet"
)

HISTORICAL_METRICS = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_season_metrics.parquet"
)

OUT_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"
IMAGE_DIR = OUT_DIR / "images"

OUT_CSV = OUT_DIR / "right_back_jaissle_detailed.csv"
OUT_HTML = OUT_DIR / "right_back_jaissle_detailed.html"
OUT_CHART = IMAGE_DIR / "right_back_jaissle_detailed_chart.png"


# ======================================================================================
# NEWCASTLE HISTORICAL REFERENCE FITS
# ======================================================================================

# Existing 2025/26 tactical reference values from the earlier Newcastle role model.
TINO_OLD_WB_FIT = 49.741
DEDIC_OLD_WB_FIT = 53.400


# ======================================================================================
# METRIC ALIASES
# ======================================================================================

ALIASES = {
    "xa": [
        "xa_per90",
        "xa",
    ],
    "key_passes": [
        "key_passes_per90",
        "key_passes",
    ],
    "accurate_crosses": [
        "accurate_crosses_per90",
        "accurate_crosses",
    ],
    "touches": [
        "touches_per90",
        "touches",
    ],
    "successful_passes": [
        "successful_passes_per90",
        "successful_passes",
    ],
    "duels_won": [
        "duels_won_per90",
        "duels_won",
    ],
    "aerials_won": [
        "aerials_won_per90",
        "aerials_won",
    ],
    "recoveries": [
        "recoveries_per90",
        "recoveries",
    ],
    "interceptions": [
        "interceptions_per90",
        "interceptions",
    ],
    "tackles_won": [
        "tackles_won_per90",
        "tackles_won",
    ],
}


# ======================================================================================
# HELPERS
# ======================================================================================

def norm(value) -> str:
    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    ).strip()


def find_alias(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    normalized = {
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

        if key in normalized:
            return normalized[key]

    return None


def p90(
    values: pd.Series,
    minutes: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    minutes = pd.to_numeric(
        minutes,
        errors="coerce",
    )

    return pd.Series(
        np.where(
            minutes > 0,
            values * 90.0 / minutes,
            np.nan,
        ),
        index=values.index,
        dtype="float64",
    )


def winsor_z(
    series: pd.Series,
) -> pd.Series:
    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = s.dropna()

    if len(valid) < 5:
        return pd.Series(
            np.nan,
            index=s.index,
            dtype="float64",
        )

    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)

    clipped = s.clip(
        lo,
        hi,
    )

    mean = clipped.mean(
        skipna=True,
    )

    std = clipped.std(
        skipna=True,
        ddof=0,
    )

    if pd.isna(std) or std == 0:
        return pd.Series(
            0.0,
            index=s.index,
            dtype="float64",
        )

    return (
        clipped - mean
    ) / std


def position_text(row) -> str:
    values = []

    for col in [
        "tm_sub_position",
        "tm_position",
        "natural_position",
    ]:
        if (
            col in row.index
            and pd.notna(
                row.get(
                    col,
                    np.nan,
                )
            )
        ):
            values.append(
                norm(row[col])
            )

    return " | ".join(values)


def is_right_back(row) -> bool:
    pos = position_text(row)

    left_terms = [
        "left back",
        "left wing back",
        "left wingback",
        "left full back",
        "left fullback",
    ]

    if any(
        term in pos
        for term in left_terms
    ):
        return False

    right_terms = [
        "right back",
        "right wing back",
        "right wingback",
        "right full back",
        "right fullback",
    ]

    return any(
        term in pos
        for term in right_terms
    )


def detect_wb_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    mask = pd.Series(
        False,
        index=df.index,
    )

    if "role_key" in df.columns:
        key = (
            df["role_key"]
            .astype(str)
            .str.lower()
        )

        mask |= key.str.contains(
            r"vertical_wingback|wingback|wing_back|fullback|full_back|fb_wb",
            regex=True,
            na=False,
        )

    if "role" in df.columns:
        role = (
            df["role"]
            .astype(str)
            .str.lower()
        )

        mask |= role.str.contains(
            r"wingback|wing back|fullback|full back|vertical full",
            regex=True,
            na=False,
        )

    return df[
        mask
    ].copy()


def player_key(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["player_name"]
        .astype(str)
        .map(norm)
        + "|"
        + df["league_slug"]
        .astype(str)
    )


def is_tino(value) -> bool:
    x = norm(value)

    return (
        x in {
            "tino livramento",
            "valentino livramento",
            "t livramento",
        }
        or x.endswith(
            " tino livramento"
        )
    )


def is_dedic(value) -> bool:
    x = norm(value)

    return (
        x in {
            "amar dedic",
            "a dedic",
        }
        or x.endswith(
            " amar dedic"
        )
    )


# ======================================================================================
# BUILD HISTORICAL RIGHT-BACK POOL
# ======================================================================================

def build_historical_pool() -> pd.DataFrame:
    shortlist = pd.read_parquet(
        FULL_SHORTLIST
    ).copy()

    enriched = pd.read_parquet(
        ENRICHED
    ).copy()

    historical = pd.read_parquet(
        HISTORICAL_METRICS
    ).copy()

    wb = detect_wb_rows(
        shortlist
    )

    wb = wb[
        wb.apply(
            is_right_back,
            axis=1,
        )
    ].copy()

    print(
        f"Right-back rows from role shortlist: {len(wb)}"
    )

    # ------------------------------------------------------------------
    # NEWCASTLE REFERENCE SAFEGUARD
    #
    # The generated role shortlist can omit a reference player even when
    # his 2025/26 historical data exists.
    #
    # This mirrors the Lewis Hall safeguard from the left-back pipeline.
    #
    # If Tino Livramento or Amar Dedic is missing:
    #   1. pull metadata from the enriched historical dataset
    #   2. append him to the RB pool
    #   3. assign the known historical WB fit
    #   4. merge ACTUAL 2025/26 football metrics below
    # ------------------------------------------------------------------

    for (
        ref_name,
        matcher,
        known_fit,
    ) in [
        (
            "Tino Livramento",
            is_tino,
            TINO_OLD_WB_FIT,
        ),
        (
            "Amar Dedić",
            is_dedic,
            DEDIC_OLD_WB_FIT,
        ),
    ]:
        already_present = (
            wb["player_name"]
            .astype(str)
            .map(matcher)
            .any()
            if "player_name" in wb.columns
            else False
        )

        if already_present:
            continue

        ref = enriched[
            enriched["player_name"]
            .astype(str)
            .map(matcher)
        ].copy()

        if ref.empty:
            print(
                f"[warning] {ref_name} "
                f"missing from role shortlist "
                f"AND enriched historical file"
            )
            continue

        ref = (
            ref.sort_values(
                "minutes",
                ascending=False,
                na_position="last",
            )
            .head(1)
            .copy()
        )

        ref["role_key"] = (
            "vertical_wingback"
        )

        ref["role"] = (
            "Vertical Full-Back / Wingback"
        )

        ref["role_fit"] = (
            known_fit
        )

        if (
            "tm_sub_position"
            in ref.columns
        ):
            ref[
                "natural_position"
            ] = ref[
                "tm_sub_position"
            ]

        if (
            "club_name"
            not in ref.columns
        ):
            ref[
                "club_name"
            ] = pd.NA

        ref[
            "club_name"
        ] = ref[
            "club_name"
        ].fillna(
            "Newcastle United"
        )

        combined_cols = list(
            dict.fromkeys(
                list(
                    wb.columns
                )
                + list(
                    ref.columns
                )
            )
        )

        wb = wb.reindex(
            columns=combined_cols
        )

        ref = ref.reindex(
            columns=combined_cols
        )

        wb = pd.concat(
            [
                wb,
                ref,
            ],
            ignore_index=True,
            sort=False,
        )

        print(
            f"[reference safeguard] "
            f"appended {ref_name} "
            f"with historical WB fit="
            f"{known_fit:.3f}"
        )

    if wb.empty:
        raise RuntimeError(
            "No right-backs found in full "
            "role shortlist using TM "
            "position fields."
        )

    wb[
        "_player_key"
    ] = player_key(
        wb
    )

    historical[
        "_player_key"
    ] = player_key(
        historical
    )

    metric_source = historical[
        [
            "_player_key",
            "minutes",
        ]
    ].copy()

    metric_source = (
        metric_source.rename(
            columns={
                "minutes":
                "_src_minutes"
            }
        )
    )

    detected = {}
    source_map = {}

    for (
        metric,
        aliases,
    ) in ALIASES.items():
        source_col = find_alias(
            historical,
            aliases,
        )

        detected[
            metric
        ] = source_col

        imported_col = (
            f"_src_{metric}"
        )

        if (
            source_col
            is not None
        ):
            metric_source[
                imported_col
            ] = pd.to_numeric(
                historical[
                    source_col
                ],
                errors="coerce",
            )

            source_map[
                metric
            ] = imported_col

        else:
            source_map[
                metric
            ] = None

    metric_source = (
        metric_source
        .drop_duplicates(
            "_player_key",
            keep="first",
        )
    )

    wb = wb.merge(
        metric_source,
        on="_player_key",
        how="left",
    ).copy()

    print()
    print(
        "=" * 110
    )
    print(
        "HISTORICAL METRIC MERGE COVERAGE"
    )
    print(
        "=" * 110
    )

    usable_cols = [
        c
        for c
        in source_map.values()
        if c is not None
    ]

    matched = int(
        wb[
            usable_cols
        ]
        .notna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows with matched historical metrics: "
        f"{matched} / {len(wb)}"
    )

    print()
    print(
        "=" * 110
    )
    print(
        "RIGHT-BACK METRIC SOURCES"
    )
    print(
        "=" * 110
    )

    for (
        metric,
        source,
    ) in detected.items():
        print(
            f"{metric:<24} "
            f"{source if source else 'MISSING'}"
        )

    for (
        metric,
        imported,
    ) in source_map.items():
        out_col = (
            f"{metric}_p90"
        )

        if (
            imported
            is None
        ):
            wb[
                out_col
            ] = np.nan

            continue

        source_name = (
            detected[
                metric
            ]
        )

        raw = pd.to_numeric(
            wb[
                imported
            ],
            errors="coerce",
        )

        if (
            source_name
            and (
                "per90"
                in source_name.lower()
                or "_p90"
                in source_name.lower()
            )
        ):
            wb[
                out_col
            ] = raw

        else:
            wb[
                out_col
            ] = p90(
                raw,
                pd.to_numeric(
                    wb[
                        "_src_minutes"
                    ],
                    errors="coerce",
                ),
            )

    return wb


# ======================================================================================
# EXTRACT 2025/26 NEWCASTLE REFERENCES
# ======================================================================================

def extract_historical_references(
    scored_pool: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pool = scored_pool.copy()

    ref_mask = (
        pool[
            "player_name"
        ]
        .astype(str)
        .map(
            is_tino
        )
        |
        pool[
            "player_name"
        ]
        .astype(str)
        .map(
            is_dedic
        )
    )

    refs = pool[
        ref_mask
    ].copy()

    external = pool[
        ~ref_mask
    ].copy()

    if refs.empty:
        raise RuntimeError(
            "Neither Tino Livramento nor "
            "Amar Dedic was found in the "
            "2025/26 historical right-back pool."
        )

    refs[
        "NUFC Benchmark"
    ] = (
        "2025/26 REFERENCE"
    )

    refs[
        "sample_status"
    ] = (
        "FULL 2025/26 SEASON"
    )

    refs[
        "data_source"
    ] = (
        "Europe 2025/26 historical"
    )

    refs.loc[
        refs[
            "player_name"
        ]
        .astype(str)
        .map(
            is_tino
        ),
        "player_name",
    ] = (
        "Tino Livramento"
    )

    refs.loc[
        refs[
            "player_name"
        ]
        .astype(str)
        .map(
            is_dedic
        ),
        "player_name",
    ] = (
        "Amar Dedić"
    )

    found = set(
        refs[
            "player_name"
        ].astype(str)
    )

    print()
    print(
        "=" * 110
    )
    print(
        "2025/26 NEWCASTLE RIGHT-BACK REFERENCES"
    )
    print(
        "=" * 110
    )

    for name in [
        "Tino Livramento",
        "Amar Dedić",
    ]:
        if name in found:
            row = refs[
                refs[
                    "player_name"
                ]
                .astype(str)
                .eq(
                    name
                )
            ].head(
                1
            )

            mins = pd.to_numeric(
                row[
                    "minutes"
                ],
                errors="coerce",
            ).iloc[
                0
            ]

            print(
                f"[reference] {name}: "
                f"2025/26 historical season, "
                f"minutes={mins}"
            )

        else:
            print(
                f"[warning] "
                f"2025/26 historical reference "
                f"not found for {name}"
            )

    return (
        external,
        refs,
    )


# ======================================================================================
# SCORE
# ======================================================================================

def score_pool(
    historical_rb: pd.DataFrame,
) -> pd.DataFrame:
    creation_inputs = {
        "xa_p90": 0.30,
        "key_passes_p90": 0.25,
        "accurate_crosses_p90": 0.20,
        "successful_passes_p90": 0.15,
        "touches_p90": 0.10,
    }

    physical_inputs = {
        "duels_won_p90": 0.30,
        "aerials_won_p90": 0.15,
        "recoveries_p90": 0.25,
        "interceptions_p90": 0.15,
        "tackles_won_p90": 0.15,
    }

    all_rows = (
        historical_rb.copy()
    )

    all_rows[
        "NUFC Benchmark"
    ] = ""

    all_rows[
        "sample_status"
    ] = (
        "FULL HISTORICAL SEASON"
    )

    all_rows[
        "data_source"
    ] = (
        "Europe 2025/26"
    )

    # ------------------------------------------------------------------
    # ATTACKING CREATION INDEX
    # ------------------------------------------------------------------

    weighted = pd.Series(
        0.0,
        index=all_rows.index,
    )

    observed = pd.Series(
        0.0,
        index=all_rows.index,
    )

    for (
        col,
        weight,
    ) in creation_inputs.items():
        if (
            col
            not in all_rows.columns
        ):
            continue

        z = winsor_z(
            all_rows[
                col
            ]
        )

        valid = (
            z.notna()
        )

        weighted.loc[
            valid
        ] += (
            z.loc[
                valid
            ]
            * weight
        )

        observed.loc[
            valid
        ] += (
            weight
        )

    all_rows[
        "attacking_creation_index"
    ] = (
        50
        + 10
        * np.where(
            observed > 0,
            weighted
            / observed,
            np.nan,
        )
    ).clip(
        20,
        80,
    )

    # ------------------------------------------------------------------
    # PHYSICAL / PRESSING ACTIVITY INDEX
    # ------------------------------------------------------------------

    weighted = pd.Series(
        0.0,
        index=all_rows.index,
    )

    observed = pd.Series(
        0.0,
        index=all_rows.index,
    )

    for (
        col,
        weight,
    ) in physical_inputs.items():
        if (
            col
            not in all_rows.columns
        ):
            continue

        z = winsor_z(
            all_rows[
                col
            ]
        )

        valid = (
            z.notna()
        )

        weighted.loc[
            valid
        ] += (
            z.loc[
                valid
            ]
            * weight
        )

        observed.loc[
            valid
        ] += (
            weight
        )

    all_rows[
        "physical_activity_index"
    ] = (
        50
        + 10
        * np.where(
            observed > 0,
            weighted
            / observed,
            np.nan,
        )
    ).clip(
        20,
        80,
    )

    all_rows[
        "role_fit"
    ] = pd.to_numeric(
        all_rows[
            "role_fit"
        ],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # JAISSLE FB SCORE
    #
    # Same structure as the left-back model:
    #
    # 40% tactical WB fit
    # 30% attacking creation
    # 30% physical / pressing activity
    # ------------------------------------------------------------------

    all_rows[
        "jaissle_fb_score"
    ] = (
        0.40
        * all_rows[
            "role_fit"
        ]
        + 0.30
        * all_rows[
            "attacking_creation_index"
        ]
        + 0.30
        * all_rows[
            "physical_activity_index"
        ]
    )

    all_rows = (
        all_rows.sort_values(
            [
                "jaissle_fb_score",
                "role_fit",
                "attacking_creation_index",
            ],
            ascending=False,
        )
    )

    (
        external,
        refs,
    ) = (
        extract_historical_references(
            all_rows
        )
    )

    combined = pd.concat(
        [
            external,
            refs,
        ],
        ignore_index=True,
        sort=False,
    )

    return (
        combined.sort_values(
            [
                "jaissle_fb_score",
                "role_fit",
                "attacking_creation_index",
            ],
            ascending=False,
        )
    )


# ======================================================================================
# CHART
# ======================================================================================

def build_chart(
    df: pd.DataFrame,
):
    historical = df[
        ~df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .eq(
            "2025/26 REFERENCE"
        )
    ].dropna(
        subset=[
            "jaissle_fb_score"
        ]
    ).head(
        20
    ).copy()

    refs = df[
        df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .eq(
            "2025/26 REFERENCE"
        )
    ].copy()

    chart = pd.concat(
        [
            historical,
            refs,
        ],
        ignore_index=True,
        sort=False,
    )

    if chart.empty:
        return

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(
            15,
            9,
        )
    )

    scatter = ax.scatter(
        chart[
            "role_fit"
        ],
        chart[
            "attacking_creation_index"
        ],
        s=np.clip(
            pd.to_numeric(
                chart[
                    "physical_activity_index"
                ],
                errors="coerce",
            ).fillna(
                50
            )
            - 30,
            5,
            None,
        )
        * 12,
        c=pd.to_numeric(
            chart[
                "xa_p90"
            ],
            errors="coerce",
        ).fillna(
            0
        ),
        alpha=0.76,
        edgecolors="black",
        linewidths=0.8,
    )

    for (
        _,
        row,
    ) in chart.iterrows():
        name = str(
            row[
                "player_name"
            ]
        )

        if (
            str(
                row.get(
                    "NUFC Benchmark",
                    "",
                )
            )
            == "2025/26 REFERENCE"
        ):
            name += (
                " [NUFC 25/26 ref]"
            )

        score = row.get(
            "jaissle_fb_score",
            np.nan,
        )

        label = (
            f"{name} "
            f"({score:.1f})"
            if pd.notna(
                score
            )
            else name
        )

        ax.annotate(
            label,
            (
                row[
                    "role_fit"
                ],
                row[
                    "attacking_creation_index"
                ],
            ),
            xytext=(
                5,
                5,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=8,
            fontweight=(
                "bold"
                if (
                    "[NUFC 25/26 ref]"
                    in name
                )
                else "normal"
            ),
        )

    ax.set_xlabel(
        "Right-Back Tactical Fit"
    )

    ax.set_ylabel(
        "Attacking Creation Index"
    )

    ax.set_title(
        "Newcastle Right-Back Recruitment — "
        "Jaissle FB Score\n"
        "2025/26 benchmarks: "
        "Tino Livramento + Amar Dedić"
    )

    plt.tight_layout()

    fig.savefig(
        OUT_CHART,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ======================================================================================
# HTML
# ======================================================================================

def save_html(
    df: pd.DataFrame,
):
    view = (
        df.head(
            100
        )
        .copy()
    )

    OUT_HTML.write_text(
        view.to_html(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        ),
        encoding="utf-8",
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical_rb = (
        build_historical_pool()
    )

    result = score_pool(
        historical_rb
    )

    export_cols = [
        c
        for c in [
            "player_name",
            "club_name",
            "league_slug",
            "age",
            "minutes",
            "NUFC Benchmark",
            "sample_status",
            "data_source",
            "role_fit",
            "jaissle_fb_score",
            "attacking_creation_index",
            "physical_activity_index",
            "xa_p90",
            "key_passes_p90",
            "accurate_crosses_p90",
            "successful_passes_p90",
            "touches_p90",
            "duels_won_p90",
            "aerials_won_p90",
            "recoveries_p90",
            "interceptions_p90",
            "tackles_won_p90",
            "market_value_eur_m",
            "natural_position",
            "tm_sub_position",
        ]
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
    print(
        "=" * 120
    )
    print(
        "TOP RIGHT-BACKS — "
        "CREATION / PHYSICAL VIEW"
    )
    print(
        "=" * 120
    )

    display_cols = [
        c
        for c in [
            "player_name",
            "club_name",
            "age",
            "NUFC Benchmark",
            "sample_status",
            "role_fit",
            "jaissle_fb_score",
            "attacking_creation_index",
            "physical_activity_index",
            "xa_p90",
            "duels_won_p90",
            "recoveries_p90",
            "interceptions_p90",
            "tackles_won_p90",
            "market_value_eur_m",
        ]
        if c in result.columns
    ]

    display = (
        result[
            display_cols
        ]
        .head(
            35
        )
        .copy()
    )

    display = (
        display.rename(
            columns={
                "player_name":
                "Player",

                "club_name":
                "Club",

                "age":
                "Age",

                "NUFC Benchmark":
                "NUFC",

                "sample_status":
                "Sample",

                "role_fit":
                "RB Fit",

                "jaissle_fb_score":
                "Jaissle RB",

                "attacking_creation_index":
                "Creation",

                "physical_activity_index":
                "Physical",

                "xa_p90":
                "xA/90",

                "duels_won_p90":
                "Duels Won/90",

                "recoveries_p90":
                "Recoveries/90",

                "interceptions_p90":
                "Interceptions/90",

                "tackles_won_p90":
                "Tackles Won/90",

                "market_value_eur_m":
                "Value €M",
            }
        )
    )

    numeric_cols = [
        "Age",
        "RB Fit",
        "Jaissle RB",
        "Creation",
        "Physical",
        "xA/90",
        "Duels Won/90",
        "Recoveries/90",
        "Interceptions/90",
        "Tackles Won/90",
        "Value €M",
    ]

    for col in numeric_cols:
        if col in display.columns:
            display[
                col
            ] = pd.to_numeric(
                display[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "NEWCASTLE RIGHT-BACK "
        "REFERENCES — 2025/26"
    )
    print(
        "=" * 120
    )

    refs = result[
        result[
            "NUFC Benchmark"
        ]
        .astype(str)
        .eq(
            "2025/26 REFERENCE"
        )
    ][
        display_cols
    ].copy()

    refs = refs.rename(
        columns={
            "player_name":
            "Player",

            "club_name":
            "Club",

            "age":
            "Age",

            "NUFC Benchmark":
            "NUFC",

            "sample_status":
            "Sample",

            "role_fit":
            "RB Fit",

            "jaissle_fb_score":
            "Jaissle RB",

            "attacking_creation_index":
            "Creation",

            "physical_activity_index":
            "Physical",

            "xa_p90":
            "xA/90",

            "duels_won_p90":
            "Duels Won/90",

            "recoveries_p90":
            "Recoveries/90",

            "interceptions_p90":
            "Interceptions/90",

            "tackles_won_p90":
            "Tackles Won/90",

            "market_value_eur_m":
            "Value €M",
        }
    )

    for col in numeric_cols:
        if col in refs.columns:
            refs[
                col
            ] = pd.to_numeric(
                refs[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )

    print(
        refs.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        " ",
        OUT_CSV,
    )

    print(
        " ",
        OUT_HTML,
    )

    print(
        " ",
        OUT_CHART,
    )


if __name__ == "__main__":
    main()