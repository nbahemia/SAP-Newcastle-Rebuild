from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]

BSD_DIR = ROOT / "data" / "processed" / "bsd"
SHORTLIST_DIR = BSD_DIR / "shortlist"

BENCHMARK_FILE = (
    SHORTLIST_DIR
    / "newcastle_jaissle_role_benchmarks.csv"
)

OUT = (
    SHORTLIST_DIR
    / "squad_depth_analysis.csv"
)


# ======================================================================================
# SETTINGS
# ======================================================================================

MAX_TARGET_VALUE_EUR_M = 100.0

ROLE_ORDER = [
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

ROLE_ORDER_MAP = {
    role: i
    for i, role in enumerate(ROLE_ORDER)
}


# ======================================================================================
# HELPERS
# ======================================================================================

def clean_name(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def first_existing(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def filter_target_value(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove external recruitment targets valued above €100m.

    Missing market values are retained because missing Transfermarkt data
    should not automatically eliminate a player.
    """

    value_col = first_existing(
        df,
        [
            "market_value_eur_m",
            "Value €M",
            "value_eur_m",
            "market_value_m",
            "value_m",
        ],
    )

    if value_col is None:
        return df.copy()

    values = pd.to_numeric(
        df[
            value_col
        ],
        errors="coerce",
    )

    return df[
        values.isna()
        | (
            values
            <= MAX_TARGET_VALUE_EUR_M
        )
    ].copy()


# ======================================================================================
# LOAD NEWCASTLE BENCHMARKS
# ======================================================================================

def load_benchmarks() -> pd.DataFrame:
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {BENCHMARK_FILE}"
        )

    df = pd.read_csv(
        BENCHMARK_FILE
    )

    df[
        "role_fit"
    ] = pd.to_numeric(
        df[
            "role_fit"
        ],
        errors="coerce",
    )

    df[
        "jaissle_role_score"
    ] = pd.to_numeric(
        df[
            "jaissle_role_score"
        ],
        errors="coerce",
    )

    for col in [
        "wide_creator_score",
        "vertical_box_score",
        "hybrid_score",
    ]:
        if col in df.columns:
            df[
                col
            ] = pd.to_numeric(
                df[
                    col
                ],
                errors="coerce",
            )

    return df


# ======================================================================================
# NEWCASTLE BENCHMARK EXTRACTION
# ======================================================================================

def get_benchmark_rows(
    benchmarks: pd.DataFrame,
    role_key: str,
    score_col: str = "jaissle_role_score",
) -> pd.DataFrame:
    rows = benchmarks[
        benchmarks[
            "role_key"
        ]
        .astype(str)
        .eq(
            role_key
        )
    ].copy()

    if score_col not in rows.columns:
        return pd.DataFrame()

    rows[
        "_score"
    ] = pd.to_numeric(
        rows[
            score_col
        ],
        errors="coerce",
    )

    rows = (
        rows.dropna(
            subset=[
                "_score"
            ]
        )
        .sort_values(
            "_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "player_name"
            ]
        )
    )

    return rows


def get_winger_benchmark_rows(
    benchmarks: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    rows = benchmarks.copy()

    if score_col not in rows.columns:
        return pd.DataFrame()

    rows[
        "_score"
    ] = pd.to_numeric(
        rows[
            score_col
        ],
        errors="coerce",
    )

    rows = (
        rows.dropna(
            subset=[
                "_score"
            ]
        )
        .sort_values(
            "_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "player_name"
            ]
        )
    )

    return rows


# ======================================================================================
# EXTERNAL TARGET HELPERS
# ======================================================================================

def best_external(
    path: Path,
    player_col: str,
    score_col: str,
    current_names: set[str] | None = None,
    marker_col: str | None = None,
) -> tuple[str | None, float]:
    if not path.exists():
        return None, np.nan

    df = pd.read_csv(
        path
    )

    if (
        player_col not in df.columns
        or score_col not in df.columns
    ):
        return None, np.nan

    # Remove targets above €100m.
    df = filter_target_value(
        df
    )

    df[
        "_score"
    ] = pd.to_numeric(
        df[
            score_col
        ],
        errors="coerce",
    )

    if current_names:
        df = df[
            ~df[
                player_col
            ]
            .astype(str)
            .isin(
                current_names
            )
        ]

    if (
        marker_col is not None
        and marker_col in df.columns
    ):
        marker = (
            df[
                marker_col
            ]
            .astype(str)
            .str.upper()
        )

        df = df[
            ~marker.str.contains(
                "REFERENCE|CURRENT|NUFC|NEWCASTLE",
                regex=True,
                na=False,
            )
        ]

    df = (
        df.dropna(
            subset=[
                "_score"
            ]
        )
        .sort_values(
            "_score",
            ascending=False,
        )
    )

    if df.empty:
        return None, np.nan

    row = df.iloc[
        0
    ]

    return (
        clean_name(
            row[
                player_col
            ]
        ),
        safe_float(
            row[
                "_score"
            ]
        ),
    )


# ======================================================================================
# ROLE ROW
# ======================================================================================

def build_row(
    role: str,
    current: pd.DataFrame,
    best_external_name: str | None,
    best_external_score: float,
    risk_mode: str,
) -> dict:
    current = current.head(
        3
    )

    choices = []

    for _, row in current.iterrows():
        choices.append(
            (
                clean_name(
                    row[
                        "player_name"
                    ]
                ),
                safe_float(
                    row[
                        "_score"
                    ]
                ),
            )
        )

    while len(choices) < 3:
        choices.append(
            (
                np.nan,
                np.nan,
            )
        )

    best_current = (
        choices[
            0
        ][1]
    )

    gap = (
        best_external_score
        - best_current
        if (
            pd.notna(
                best_external_score
            )
            and pd.notna(
                best_current
            )
        )
        else np.nan
    )

    # ------------------------------------------------------------------
    # RISK LOGIC
    # ------------------------------------------------------------------

    if risk_mode == "gk":
        risk = (
            "YELLOW — starter trusted, model confidence low"
        )

    elif risk_mode == "striker":
        if pd.isna(
            best_current
        ):
            risk = (
                "GRAY — insufficient current-role coverage"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 8
        ):
            risk = (
                "RED — major starter quality gap"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 5
        ):
            risk = (
                "ORANGE — meaningful starter upgrade available"
            )

        else:
            risk = (
                "YELLOW — monitor striker quality"
            )

    elif risk_mode == "quality":
        if pd.isna(
            best_current
        ):
            risk = (
                "GRAY — insufficient current-role coverage"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 7
        ):
            risk = (
                "ORANGE — specialist quality gap"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 4
        ):
            risk = (
                "YELLOW — upgrade opportunity"
            )

        elif pd.notna(
            choices[
                1
            ][1]
        ):
            risk = (
                "GREEN — adequate quality / depth"
            )

        else:
            risk = (
                "YELLOW — limited specialist depth"
            )

    elif risk_mode == "fullback":
        if pd.isna(
            best_current
        ):
            risk = (
                "GRAY — insufficient current-role coverage"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 6
        ):
            risk = (
                "ORANGE — meaningful starter upgrade available"
            )

        elif (
            pd.notna(
                gap
            )
            and gap >= 3
        ):
            risk = (
                "YELLOW — upgrade opportunity"
            )

        elif pd.notna(
            choices[
                1
            ][1]
        ):
            risk = (
                "GREEN — starter + depth"
            )

        else:
            risk = (
                "YELLOW — starter, shallow depth"
            )

    else:
        if pd.isna(
            best_current
        ):
            risk = (
                "GRAY — insufficient current-role coverage"
            )

        elif pd.notna(
            choices[
                1
            ][1]
        ):
            risk = (
                "GREEN — starter + viable internal alternative"
            )

        else:
            risk = (
                "YELLOW — shallow depth"
            )

    return {
        "Role":
        role,

        "Choice 1":
        choices[
            0
        ][0],

        "Score 1":
        choices[
            0
        ][1],

        "Choice 2":
        choices[
            1
        ][0],

        "Score 2":
        choices[
            1
        ][1],

        "Choice 3":
        choices[
            2
        ][0],

        "Score 3":
        choices[
            2
        ][1],

        "Best External":
        best_external_name,

        "External Score":
        best_external_score,

        "Upgrade Gap":
        gap,

        "Depth Risk":
        risk,
    }


# ======================================================================================
# GOALKEEPER
# ======================================================================================

def goalkeeper_row() -> dict:
    path = (
        SHORTLIST_DIR
        / "goalkeeper_jaissle_detailed.csv"
    )

    df = pd.read_csv(
        path
    )

    current = df[
        df[
            "Club"
        ]
        .astype(str)
        .str.contains(
            "Newcastle",
            case=False,
            na=False,
        )
    ].copy()

    current[
        "_score"
    ] = pd.to_numeric(
        current[
            "Jaissle GK Score"
        ],
        errors="coerce",
    )

    current = current.sort_values(
        "_score",
        ascending=False,
        na_position="last",
    )

    external = df[
        ~df.index.isin(
            current.index
        )
    ].copy()

    external = filter_target_value(
        external
    )

    external[
        "_score"
    ] = pd.to_numeric(
        external[
            "Jaissle GK Score"
        ],
        errors="coerce",
    )

    external = (
        external.dropna(
            subset=[
                "_score"
            ]
        )
        .sort_values(
            "_score",
            ascending=False,
        )
    )

    external_name = (
        external.iloc[
            0
        ][
            "Player"
        ]
        if not external.empty
        else None
    )

    external_score = (
        external.iloc[
            0
        ][
            "_score"
        ]
        if not external.empty
        else np.nan
    )

    current = current.rename(
        columns={
            "Player":
            "player_name"
        }
    )

    return build_row(
        "Sweeper / Build-Up GK",
        current,
        external_name,
        external_score,
        "gk",
    )


# ======================================================================================
# RIGHT BACK
# ======================================================================================

def right_back_row() -> dict:
    path = (
        SHORTLIST_DIR
        / "right_back_jaissle_detailed.csv"
    )

    df = pd.read_csv(
        path
    )

    df[
        "_score"
    ] = pd.to_numeric(
        df[
            "jaissle_fb_score"
        ],
        errors="coerce",
    )

    current = df[
        df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .eq(
            "2025/26 REFERENCE"
        )
    ].copy()

    current = current.sort_values(
        "_score",
        ascending=False,
    )

    external = df[
        ~df.index.isin(
            current.index
        )
    ].copy()

    external = filter_target_value(
        external
    )

    external = external.sort_values(
        "_score",
        ascending=False,
    )

    ext_name = (
        external.iloc[
            0
        ][
            "player_name"
        ]
        if not external.empty
        else None
    )

    ext_score = (
        external.iloc[
            0
        ][
            "_score"
        ]
        if not external.empty
        else np.nan
    )

    return build_row(
        "Right WB / FB",
        current,
        ext_name,
        ext_score,
        "fullback",
    )


# ======================================================================================
# CONTROLLER
# ======================================================================================

def controller_row() -> dict:
    path = (
        SHORTLIST_DIR
        / "holding_controller_jaissle_detailed.csv"
    )

    df = pd.read_csv(
        path
    )

    df[
        "_score"
    ] = pd.to_numeric(
        df[
            "Jaissle Controller"
        ],
        errors="coerce",
    )

    current = df[
        df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "CURRENT"
        )
    ].copy()

    current = current.rename(
        columns={
            "Player":
            "player_name"
        }
    )

    current = current.sort_values(
        "_score",
        ascending=False,
    )

    external = df[
        ~df.index.isin(
            current.index
        )
    ].copy()

    external = filter_target_value(
        external
    )

    external = external.sort_values(
        "_score",
        ascending=False,
    )

    ext_name = (
        external.iloc[
            0
        ][
            "Player"
        ]
        if not external.empty
        else None
    )

    ext_score = (
        external.iloc[
            0
        ][
            "_score"
        ]
        if not external.empty
        else np.nan
    )

    return build_row(
        "Holding Controller",
        current,
        ext_name,
        ext_score,
        "depth",
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    benchmarks = load_benchmarks()

    rows = []

    # ------------------------------------------------------------------
    # GK
    # ------------------------------------------------------------------

    rows.append(
        goalkeeper_row()
    )

    # ------------------------------------------------------------------
    # CB
    # ------------------------------------------------------------------

    cb_current = get_benchmark_rows(
        benchmarks,
        "center_back",
    )

    cb_ext_name, cb_ext_score = best_external(
        SHORTLIST_DIR
        / "center_back_jaissle_detailed.csv",
        "player_name",
        "jaissle_cb_score",
    )

    rows.append(
        build_row(
            "High-Line CB",
            cb_current,
            cb_ext_name,
            cb_ext_score,
            "depth",
        )
    )

    # ------------------------------------------------------------------
    # LEFT WB / FB
    # ------------------------------------------------------------------

    lb_current = get_benchmark_rows(
        benchmarks,
        "vertical_wingback",
    )

    lb_current = lb_current[
        lb_current[
            "player_name"
        ]
        .astype(str)
        .eq(
            "Lewis Hall"
        )
    ].copy()

    lb_ext_name, lb_ext_score = best_external(
        SHORTLIST_DIR
        / "left_wingback_recruitment_progressive.csv",
        "player_name",
        "role_fit",
        current_names={
            "L. Hall",
            "Lewis Hall",
        },
    )

    rows.append(
        build_row(
            "Left WB / FB",
            lb_current,
            lb_ext_name,
            lb_ext_score,
            "fullback",
        )
    )

    # ------------------------------------------------------------------
    # RIGHT WB / FB
    # ------------------------------------------------------------------

    rows.append(
        right_back_row()
    )

    # ------------------------------------------------------------------
    # HOLDING CONTROLLER
    # ------------------------------------------------------------------

    rows.append(
        controller_row()
    )

    # ------------------------------------------------------------------
    # WORKHORSE / ADVANCED 8
    # ------------------------------------------------------------------

    cm_current = get_benchmark_rows(
        benchmarks,
        "advanced_pivot",
    )

    cm_ext_name, cm_ext_score = best_external(
        SHORTLIST_DIR
        / "workhorse_cm_jaissle_detailed.csv",
        "player_name",
        "jaissle_workhorse_score",
    )

    rows.append(
        build_row(
            "Advanced / Workhorse 8",
            cm_current,
            cm_ext_name,
            cm_ext_score,
            "depth",
        )
    )

    # ------------------------------------------------------------------
    # CREATIVE 10
    # ------------------------------------------------------------------

    cam_current = get_benchmark_rows(
        benchmarks,
        "creative_cam",
    )

    if cam_current.empty:
        cam_current = get_benchmark_rows(
            benchmarks,
            "creative_10",
        )

    cam_ext_name, cam_ext_score = best_external(
        SHORTLIST_DIR
        / "creative_10_jaissle_detailed.csv",
        "Player",
        "CAM Fit",
    )

    rows.append(
        build_row(
            "Creative 10",
            cam_current,
            cam_ext_name,
            cam_ext_score,
            "quality",
        )
    )

    # ------------------------------------------------------------------
    # WINGERS
    # ------------------------------------------------------------------

    winger_path = (
        SHORTLIST_DIR
        / "winger_jaissle_archetypes.csv"
    )

    winger_df = pd.read_csv(
        winger_path
    )

    for (
        role_name,
        benchmark_col,
        external_col,
    ) in [
        (
            "Wide Creator",
            "wide_creator_score",
            "wide_creator_score",
        ),
        (
            "Vertical / Box Winger",
            "vertical_box_score",
            "vertical_box_score",
        ),
        (
            "Hybrid Winger",
            "hybrid_score",
            "hybrid_score",
        ),
    ]:
        current = get_winger_benchmark_rows(
            benchmarks,
            benchmark_col,
        )

        ext = winger_df[
            ~winger_df[
                "is_newcastle_benchmark"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        ].copy()

        ext = filter_target_value(
            ext
        )

        ext[
            "_score"
        ] = pd.to_numeric(
            ext[
                external_col
            ],
            errors="coerce",
        )

        ext = (
            ext.dropna(
                subset=[
                    "_score"
                ]
            )
            .sort_values(
                "_score",
                ascending=False,
            )
        )

        ext_name = (
            ext.iloc[
                0
            ][
                "player_name"
            ]
            if not ext.empty
            else None
        )

        ext_score = (
            ext.iloc[
                0
            ][
                "_score"
            ]
            if not ext.empty
            else np.nan
        )

        rows.append(
            build_row(
                role_name,
                current,
                ext_name,
                ext_score,
                "quality",
            )
        )

    # ------------------------------------------------------------------
    # STRIKER
    # ------------------------------------------------------------------

    striker_current = get_benchmark_rows(
        benchmarks,
        "focal_striker",
    )

    st_ext_name, st_ext_score = best_external(
        SHORTLIST_DIR
        / "focal_striker_jaissle_detailed.csv",
        "player_name",
        "jaissle_striker_score",
    )

    rows.append(
        build_row(
            "Focal Striker",
            striker_current,
            st_ext_name,
            st_ext_score,
            "striker",
        )
    )

    result = pd.DataFrame(
        rows
    )

    result[
        "_role_order"
    ] = (
        result[
            "Role"
        ]
        .map(
            ROLE_ORDER_MAP
        )
        .fillna(
            999
        )
    )

    result = result.sort_values(
        "_role_order"
    ).reset_index(
        drop=True
    )

    numeric_cols = [
        "Score 1",
        "Score 2",
        "Score 3",
        "External Score",
        "Upgrade Gap",
    ]

    for col in numeric_cols:
        result[
            col
        ] = pd.to_numeric(
            result[
                col
            ],
            errors="coerce",
        ).round(
            2
        )

    save_result = result.drop(
        columns=[
            "_role_order"
        ],
        errors="ignore",
    )

    save_result.to_csv(
        OUT,
        index=False,
    )

    print()
    print(
        "=" * 120
    )
    print(
        "NEWCASTLE SQUAD DEPTH / QUALITY DIAGNOSIS"
    )
    print(
        "=" * 120
    )

    print(
        save_result[
            [
                "Role",
                "Choice 1",
                "Score 1",
                "Choice 2",
                "Score 2",
                "Best External",
                "External Score",
                "Upgrade Gap",
                "Depth Risk",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Maximum target value: €{MAX_TARGET_VALUE_EUR_M:.0f}m"
    )

    print(
        f"Saved: {OUT}"
    )


if __name__ == "__main__":
    main()