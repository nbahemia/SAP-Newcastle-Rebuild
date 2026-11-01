from __future__ import annotations
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
# ======================================================================================
# PATHS
# ======================================================================================
ROOT = Path(__file__).resolve().parents[4]
BSD_DIR = ROOT / "data" / "processed" / "bsd"
SHORTLIST_DIR = BSD_DIR / "shortlist"
DEPTH_FILE = (
    SHORTLIST_DIR
    / "squad_depth_analysis.csv"
)
FULL_ROLE_FILE = (
    BSD_DIR
    / "europe_2025_26_full_role_shortlist.parquet"
)
WORKHORSE_FILE = (
    SHORTLIST_DIR
    / "workhorse_cm_jaissle_detailed.csv"
)
TRANSFER_OVERRIDE_FILE = (
    SHORTLIST_DIR
    / "current_transfer_overrides.csv"
)
OUT = (
    SHORTLIST_DIR
    / "recruitment_value_analysis.csv"
)
TARGET_CHECK_OUT = SHORTLIST_DIR / "priority_target_full_check.csv"
TARGET_METRICS_OUT = SHORTLIST_DIR / "priority_target_all_metrics_long.csv"
YOUNG_CB_OUT = SHORTLIST_DIR / "young_cb_recruitment.csv"
BONA_FIDE_WINGER_OUT = SHORTLIST_DIR / "bona_fide_winger_recruitment.csv"
MIXED_FULLBACK_OUT = SHORTLIST_DIR / "mixed_fullback_recruitment.csv"
SEASON_METRICS_FILE = BSD_DIR / "europe_2025_26_player_season_metrics.parquet"
SEASON_METRICS_ENRICHED_FILE = BSD_DIR / "player_season_metrics_enriched.parquet"
POSITION_HISTORY_FILE = BSD_DIR / "player_position_history.parquet"
NEWCASTLE_ROLE_BENCHMARKS_FILE = SHORTLIST_DIR / "newcastle_jaissle_role_benchmarks.csv"
NEWCASTLE_OPTIMAL_ROLES_FILE = SHORTLIST_DIR / "newcastle_jaissle_optimal_roles.csv"
# ======================================================================================
# SETTINGS
# ======================================================================================
MAX_TARGET_VALUE_EUR_M = 100.0
MAX_RECRUITMENT_AGE = 30
MAX_YOUNG_CB_AGE = 23
MAX_YOUNG_CB_VALUE_EUR_M = 35.0
MIN_YOUNG_CB_MINUTES = 900.0
MAX_WINGER_AGE = 28
MAX_WINGER_VALUE_EUR_M = 100.0
MIN_WINGER_MINUTES = 700.0
MAX_MIXED_FB_AGE = 28
MAX_MIXED_FB_VALUE_EUR_M = 35.0
MIN_MIXED_FB_MINUTES = 700.0
TARGET_DEEP_DIVE_ALIASES = {
    "Nikola Krstović": ["N. Krstović", "Nikola Krstovic", "Nikola Krstović"],
    "Rayan Cherki": ["R. Cherki", "Rayan Cherki"],
    "Lazar Samardžić": ["L. Samardžić", "Lazar Samardzic", "Lazar Samardžić"],
    "Bart van Rooij": ["B. van Rooij", "Bart van Rooij"],
    "Rayan Aït-Nouri": ["R. Aït-Nouri", "Rayan Ait-Nouri", "Rayan Aït-Nouri"],
    "Rodrigo Zalazar": ["R. Zalazar", "Rodrigo Zalazar"],
    "Saïd El Mala": ["S. El Mala", "Said El Mala", "Saïd El Mala"],
    "Kenan Yıldız": ["K. Yıldız", "Kenan Yildiz", "Kenan Yıldız"],
    "Igor Matanović": ["I. Matanović", "Igor Matanovic", "Igor Matanović"],
    "William Osula": ["W. Osula", "William Osula"],
    "Bazoumana Touré": ["B. Touré", "Bazoumana Toure", "Bazoumana Touré"],
    "Anthony Elanga": ["A. Elanga", "Anthony Elanga"],
}
# ======================================================================================
# RECRUITMENT SUITABILITY WEIGHTS
#
# This board is specifically designed around RIGHT-NOW Newcastle performance.
#
# League strength is therefore weighted heavily:
#
# 45% actual tactical / role performance
# 25% improvement over Newcastle's current option
# 25% proven league environment
#  5% age
#
# Value is handled separately.
# ======================================================================================
TACTICAL_QUALITY_WEIGHT = 0.45
UPGRADE_WEIGHT = 0.25
LEAGUE_STRENGTH_WEIGHT = 0.25
AGE_WEIGHT = 0.05
# ======================================================================================
# VALUE SCORE WEIGHTS
# ======================================================================================
VALUE_SUITABILITY_WEIGHT = 0.45
VALUE_AFFORDABILITY_WEIGHT = 0.25
VALUE_AGE_WEIGHT = 0.15
VALUE_UPGRADE_WEIGHT = 0.15
# ======================================================================================
# ROLE ORDER
# ======================================================================================
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
    for i, role in enumerate(
        ROLE_ORDER
    )
}
# ======================================================================================
# FALLBACK LEAGUE STRENGTH
#
# Primary values come from:
# europe_2025_26_full_role_shortlist.parquet
#
# These are ONLY used when the historical shortlist does not contain the league.
# ======================================================================================
FALLBACK_LEAGUE_STRENGTH = {
    "premier_league": 100.0,
    "serie_a": 88.9,
    "la_liga": 82.9,
    "bundesliga": 80.0,
    "ligue_1": 72.9,
    "primeira_liga": 70.0,
    "eredivisie": 63.4,
    "belgian_pro_league": 63.0,
    "turkish_super_lig": 61.0,
    "swiss_super_league": 60.0,
    "austrian_bundesliga": 58.0,
    "scottish_premiership": 56.0,
    "danish_superliga": 54.0,
    "championship": 67.0,
    "english_championship": 67.0,
}
LEAGUE_DISPLAY = {
    "premier_league": "Premier League",
    "serie_a": "Serie A",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "ligue_1": "Ligue 1",
    "primeira_liga": "Primeira Liga",
    "eredivisie": "Eredivisie",
    "belgian_pro_league": "Belgian Pro League",
    "turkish_super_lig": "Turkish Süper Lig",
    "swiss_super_league": "Swiss Super League",
    "austrian_bundesliga": "Austrian Bundesliga",
    "scottish_premiership": "Scottish Premiership",
    "danish_superliga": "Danish Superliga",
    "championship": "Championship",
    "english_championship": "Championship",
}
# ======================================================================================
# NAME HELPERS
# ======================================================================================
def normalize_name(
    value,
) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize(
        "NFKD",
        str(value),
    )
    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
    )
    value = value.lower()
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )
    return value.strip()
def name_aliases(
    value,
) -> set[str]:
    normalized = normalize_name(
        value
    )
    if not normalized:
        return set()
    parts = normalized.split()
    aliases = {
        normalized
    }
    if len(parts) >= 2:
        surname = " ".join(
            parts[
                1:
            ]
        )
        aliases.add(
            f"{parts[0][0]} {surname}"
        )
    return aliases
def find_col(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        normalize_name(col):
        col
        for col in df.columns
    }
    for candidate in candidates:
        key = normalize_name(
            candidate
        )
        if key in lookup:
            return lookup[
                key
            ]
    return None
# ======================================================================================
# LEAGUE HELPERS
# ======================================================================================
def normalize_league(
    value,
) -> str:
    if pd.isna(value):
        return ""
    value = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    while "__" in value:
        value = value.replace(
            "__",
            "_",
        )
    aliases = {
        "premierleague":
        "premier_league",
        "english_premier_league":
        "premier_league",
        "england_premier_league":
        "premier_league",
        "laliga":
        "la_liga",
        "seriea":
        "serie_a",
        "ligue1":
        "ligue_1",
        "liga_portugal":
        "primeira_liga",
        "portuguese_league":
        "primeira_liga",
        "portuguese_primeira_liga":
        "primeira_liga",
        "jupiler_pro_league":
        "belgian_pro_league",
        "belgian_first_division_a":
        "belgian_pro_league",
        "super_lig":
        "turkish_super_lig",
        "scottish_premier_league":
        "scottish_premiership",
        "danish_superligaen":
        "danish_superliga",
    }
    return aliases.get(
        value,
        value,
    )
def display_league(
    value,
) -> str:
    key = normalize_league(
        value
    )
    if not key:
        return ""
    return LEAGUE_DISPLAY.get(
        key,
        key.replace(
            "_",
            " ",
        ).title(),
    )
def fallback_league_strength(
    league,
) -> float:
    key = normalize_league(
        league
    )
    if not key:
        return 60.0
    return FALLBACK_LEAGUE_STRENGTH.get(
        key,
        60.0,
    )
# ======================================================================================
# GENERIC SCORING HELPERS
# ======================================================================================
def percentile_score(
    series: pd.Series,
) -> pd.Series:
    s = pd.to_numeric(
        series,
        errors="coerce",
    )
    return (
        s.rank(
            pct=True,
            method="average",
        )
        * 100
    )
def age_component(
    series: pd.Series,
) -> pd.Series:
    age = pd.to_numeric(
        series,
        errors="coerce",
    )
    result = pd.Series(
        50.0,
        index=age.index,
        dtype="float64",
    )
    # Young development player
    result.loc[
        age <= 20
    ] = 90
    # Ideal acquisition range
    result.loc[
        age.between(
            21,
            24,
            inclusive="both",
        )
    ] = 100
    # Prime
    result.loc[
        age.between(
            25,
            26,
            inclusive="both",
        )
    ] = 90
    # Experienced prime
    result.loc[
        age.between(
            27,
            28,
            inclusive="both",
        )
    ] = 75
    # Immediate performance but shorter horizon
    result.loc[
        age.between(
            29,
            30,
            inclusive="both",
        )
    ] = 55
    result.loc[
        age >= 31
    ] = 20
    return result
# ======================================================================================
# BENCHMARKS
# ======================================================================================
def load_benchmarks() -> dict[str, float]:
    if not DEPTH_FILE.exists():
        raise FileNotFoundError(
            DEPTH_FILE
        )
    df = pd.read_csv(
        DEPTH_FILE
    )
    benchmarks = {}
    for _, row in df.iterrows():
        score = pd.to_numeric(
            pd.Series(
                [
                    row[
                        "Score 1"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]
        if pd.notna(score):
            benchmarks[
                str(
                    row[
                        "Role"
                    ]
                ).strip()
            ] = float(
                score
            )
    print()
    print(
        "Loaded squad benchmarks:"
    )
    for role in ROLE_ORDER:
        if role in benchmarks:
            print(
                f"  "
                f"{role:<30} "
                f"{benchmarks[role]:.2f}"
            )
    return benchmarks
# ======================================================================================
# LEAGUE STRENGTH MAP
#
# IMPORTANT:
# League strength is determined by league_slug.
#
# It is NOT transferred between players during player-name enrichment.
# ======================================================================================
def load_league_strength_map() -> dict[str, float]:
    df = pd.read_parquet(
        FULL_ROLE_FILE
    )
    temp = df[
        [
            "league_slug",
            "league_strength",
        ]
    ].copy()
    temp[
        "league_strength"
    ] = pd.to_numeric(
        temp[
            "league_strength"
        ],
        errors="coerce",
    )
    temp = temp.dropna(
        subset=[
            "league_slug",
            "league_strength",
        ]
    )
    raw_map = (
        temp.groupby(
            "league_slug"
        )[
            "league_strength"
        ]
        .median()
        .to_dict()
    )
    # Also normalize keys.
    result = {}
    for league, strength in (
        raw_map.items()
    ):
        result[
            normalize_league(
                league
            )
        ] = float(
            strength
        )
    return result
# ======================================================================================
# FULL ROLE CONTEXT
# ======================================================================================
def load_role_context() -> pd.DataFrame:
    df = pd.read_parquet(
        FULL_ROLE_FILE
    )
    required = [
        "player_name",
        "league_slug",
        "minutes",
        "role_fit",
        "age",
        "market_value_eur_m",
        "club_name",
    ]
    context = df[
        required
    ].copy()
    context[
        "_name_key"
    ] = (
        context[
            "player_name"
        ]
        .astype(str)
        .map(
            normalize_name
        )
    )
    context[
        "minutes"
    ] = pd.to_numeric(
        context[
            "minutes"
        ],
        errors="coerce",
    )
    context[
        "role_fit"
    ] = pd.to_numeric(
        context[
            "role_fit"
        ],
        errors="coerce",
    )
    context[
        "age"
    ] = pd.to_numeric(
        context[
            "age"
        ],
        errors="coerce",
    )
    context[
        "market_value_eur_m"
    ] = pd.to_numeric(
        context[
            "market_value_eur_m"
        ],
        errors="coerce",
    )
    # Player can appear in several role models.
    # Prefer the strongest sample.
    context = context.sort_values(
        [
            "minutes",
            "role_fit",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )
    context = context.drop_duplicates(
        subset=[
            "_name_key"
        ],
        keep="first",
    )
    return context.rename(
        columns={
            "league_slug":
            "_ctx_league",
            "minutes":
            "_ctx_minutes",
            "age":
            "_ctx_age",
            "market_value_eur_m":
            "_ctx_value",
            "club_name":
            "_ctx_club",
        }
    )
# ======================================================================================
# WORKHORSE CM CONTEXT
#
# Controller metadata first looks here.
# ======================================================================================
def load_workhorse_cm_context() -> dict[str, dict]:
    if not WORKHORSE_FILE.exists():
        return {}
    df = pd.read_csv(
        WORKHORSE_FILE
    )
    required = [
        "player_name",
        "club_name",
        "league_slug",
        "age",
        "market_value_eur_m",
        "minutes",
    ]
    if any(
        col not in df.columns
        for col in required
    ):
        return {}
    result = {}
    for _, row in df.iterrows():
        record = {
            "club":
            (
                str(
                    row[
                        "club_name"
                    ]
                ).strip()
                if pd.notna(
                    row[
                        "club_name"
                    ]
                )
                else ""
            ),
            "league":
            (
                str(
                    row[
                        "league_slug"
                    ]
                ).strip()
                if pd.notna(
                    row[
                        "league_slug"
                    ]
                )
                else ""
            ),
            "age":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "age"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
            "value":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "market_value_eur_m"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
            "minutes":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "minutes"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
        }
        for alias in name_aliases(
            row[
                "player_name"
            ]
        ):
            result[
                alias
            ] = record
    return result
# ======================================================================================
# TRANSFER OVERRIDES
# ======================================================================================
def load_transfer_overrides() -> dict[str, dict]:
    if not TRANSFER_OVERRIDE_FILE.exists():
        return {}
    df = pd.read_csv(
        TRANSFER_OVERRIDE_FILE
    )
    player_col = find_col(
        df,
        [
            "player_name",
            "Player",
        ],
    )
    club_col = find_col(
        df,
        [
            "current_club",
            "Current Club",
        ],
    )
    value_col = find_col(
        df,
        [
            "current_market_value_eur_m",
            "Current Market Value €M",
        ],
    )
    if player_col is None:
        return {}
    overrides = {}
    for _, row in df.iterrows():
        player = row[
            player_col
        ]
        if pd.isna(player):
            continue
        record = {
            "current_club":
            (
                str(
                    row[
                        club_col
                    ]
                ).strip()
                if (
                    club_col is not None
                    and pd.notna(
                        row[
                            club_col
                        ]
                    )
                )
                else ""
            ),
            "current_value":
            (
                pd.to_numeric(
                    pd.Series(
                        [
                            row[
                                value_col
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[
                    0
                ]
                if value_col is not None
                else np.nan
            ),
        }
        for alias in name_aliases(
            player
        ):
            overrides[
                alias
            ] = record
    return overrides
# ======================================================================================
# CONTROLLER ENRICHMENT
# ======================================================================================
def build_role_alias_lookup(
    role_context: pd.DataFrame,
) -> dict[str, dict]:
    result = {}
    for _, row in (
        role_context.iterrows()
    ):
        record = {
            "club":
            (
                str(
                    row[
                        "_ctx_club"
                    ]
                ).strip()
                if pd.notna(
                    row[
                        "_ctx_club"
                    ]
                )
                else ""
            ),
            "league":
            (
                str(
                    row[
                        "_ctx_league"
                    ]
                ).strip()
                if pd.notna(
                    row[
                        "_ctx_league"
                    ]
                )
                else ""
            ),
            "age":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "_ctx_age"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
            "value":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "_ctx_value"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
            "minutes":
            pd.to_numeric(
                pd.Series(
                    [
                        row[
                            "_ctx_minutes"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ],
        }
        for alias in name_aliases(
            row[
                "player_name"
            ]
        ):
            result[
                alias
            ] = record
    return result
def enrich_controller_metadata(
    out: pd.DataFrame,
    workhorse_context: dict[str, dict],
    role_alias_lookup: dict[str, dict],
) -> pd.DataFrame:
    result = out.copy()
    for idx in result.index:
        player = result.loc[
            idx,
            "Player",
        ]
        record = None
        # --------------------------------------------------------------
        # FIRST: Workhorse CM metadata
        # --------------------------------------------------------------
        for alias in name_aliases(
            player
        ):
            if alias in workhorse_context:
                record = (
                    workhorse_context[
                        alias
                    ]
                )
                break
        # --------------------------------------------------------------
        # SECOND: full role shortlist
        # --------------------------------------------------------------
        if record is None:
            for alias in name_aliases(
                player
            ):
                if alias in role_alias_lookup:
                    record = (
                        role_alias_lookup[
                            alias
                        ]
                    )
                    break
        if record is None:
            continue
        if (
            not str(
                result.loc[
                    idx,
                    "TM / Current Club",
                ]
            ).strip()
            and record[
                "club"
            ]
        ):
            result.loc[
                idx,
                "TM / Current Club",
            ] = record[
                "club"
            ]
        if (
            not str(
                result.loc[
                    idx,
                    "_raw_league",
                ]
            ).strip()
            and record[
                "league"
            ]
        ):
            result.loc[
                idx,
                "_raw_league",
            ] = record[
                "league"
            ]
        if pd.isna(
            result.loc[
                idx,
                "Age",
            ]
        ):
            result.loc[
                idx,
                "Age",
            ] = record[
                "age"
            ]
        if pd.isna(
            result.loc[
                idx,
                "Model Value €M",
            ]
        ):
            result.loc[
                idx,
                "Model Value €M",
            ] = record[
                "value"
            ]
        if pd.isna(
            result.loc[
                idx,
                "Minutes",
            ]
        ):
            result.loc[
                idx,
                "Minutes",
            ] = record[
                "minutes"
            ]
    return result
# ======================================================================================
# STANDARDIZE ROLE DATA
# ======================================================================================
def standardize(
    df: pd.DataFrame,
    role: str,
    player_col: str,
    score_col: str,
    benchmark: float,
    role_context: pd.DataFrame,
    league_strength_map: dict[str, float],
    workhorse_context: dict[str, dict],
    role_alias_lookup: dict[str, dict],
    club_col: str | None = None,
    league_col: str | None = None,
    age_col: str | None = None,
    value_col: str | None = None,
    minutes_col: str | None = None,
    exclude_mask: pd.Series | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if exclude_mask is not None:
        exclude_mask = (
            exclude_mask
            .reindex(
                work.index,
                fill_value=False,
            )
            .fillna(
                False
            )
            .astype(
                bool
            )
        )
        work = work.loc[
            ~exclude_mask
        ].copy()
    work = work.reset_index(
        drop=True
    )
    if work.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "Role":
            np.repeat(
                role,
                len(
                    work
                ),
            ),
            "Player":
            work[
                player_col
            ]
            .astype(str)
            .to_numpy(),
            "Target Score":
            pd.to_numeric(
                work[
                    score_col
                ],
                errors="coerce",
            ).to_numpy(),
            "Current Benchmark":
            np.repeat(
                float(
                    benchmark
                ),
                len(
                    work
                ),
            ),
        }
    )
    out[
        "_name_key"
    ] = (
        out[
            "Player"
        ]
        .map(
            normalize_name
        )
    )
    out[
        "TM / Current Club"
    ] = (
        work[
            club_col
        ]
        .fillna("")
        .astype(str)
        .to_numpy()
        if (
            club_col is not None
            and club_col in work.columns
        )
        else ""
    )
    out[
        "_raw_league"
    ] = (
        work[
            league_col
        ]
        .fillna("")
        .astype(str)
        .to_numpy()
        if (
            league_col is not None
            and league_col in work.columns
        )
        else ""
    )
    out[
        "Age"
    ] = (
        pd.to_numeric(
            work[
                age_col
            ],
            errors="coerce",
        ).to_numpy()
        if (
            age_col is not None
            and age_col in work.columns
        )
        else np.nan
    )
    out[
        "Model Value €M"
    ] = (
        pd.to_numeric(
            work[
                value_col
            ],
            errors="coerce",
        ).to_numpy()
        if (
            value_col is not None
            and value_col in work.columns
        )
        else np.nan
    )
    out[
        "Minutes"
    ] = (
        pd.to_numeric(
            work[
                minutes_col
            ],
            errors="coerce",
        ).to_numpy()
        if (
            minutes_col is not None
            and minutes_col in work.columns
        )
        else np.nan
    )
    # ==================================================================================
    # GENERAL FULL-ROLE ENRICHMENT
    # ==================================================================================
    out = out.merge(
        role_context,
        on="_name_key",
        how="left",
    )
    missing_club = (
        out[
            "TM / Current Club"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        == 0
    )
    out.loc[
        missing_club,
        "TM / Current Club",
    ] = (
        out.loc[
            missing_club,
            "_ctx_club",
        ]
        .fillna("")
        .astype(str)
    )
    missing_league = (
        out[
            "_raw_league"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        == 0
    )
    out.loc[
        missing_league,
        "_raw_league",
    ] = (
        out.loc[
            missing_league,
            "_ctx_league",
        ]
        .fillna("")
        .astype(str)
    )
    out[
        "Age"
    ] = (
        pd.to_numeric(
            out[
                "Age"
            ],
            errors="coerce",
        )
        .fillna(
            pd.to_numeric(
                out[
                    "_ctx_age"
                ],
                errors="coerce",
            )
        )
    )
    out[
        "Model Value €M"
    ] = (
        pd.to_numeric(
            out[
                "Model Value €M"
            ],
            errors="coerce",
        )
        .fillna(
            pd.to_numeric(
                out[
                    "_ctx_value"
                ],
                errors="coerce",
            )
        )
    )
    out[
        "Minutes"
    ] = (
        pd.to_numeric(
            out[
                "Minutes"
            ],
            errors="coerce",
        )
        .fillna(
            pd.to_numeric(
                out[
                    "_ctx_minutes"
                ],
                errors="coerce",
            )
        )
    )
    # ==================================================================================
    # CONTROLLER SPECIAL ENRICHMENT
    # ==================================================================================
    if role == "Holding Controller":
        out = enrich_controller_metadata(
            out,
            workhorse_context,
            role_alias_lookup,
        )
    # ==================================================================================
    # LEAGUE STRENGTH
    #
    # ALWAYS derive from league_slug.
    # Never derive from matched player identity.
    # ==================================================================================
    normalized_league = (
        out[
            "_raw_league"
        ]
        .map(
            normalize_league
        )
    )
    out[
        "League Strength"
    ] = (
        normalized_league
        .map(
            league_strength_map
        )
    )
    missing_strength = (
        out[
            "League Strength"
        ].isna()
    )
    out.loc[
        missing_strength,
        "League Strength",
    ] = (
        out.loc[
            missing_strength,
            "_raw_league",
        ]
        .map(
            fallback_league_strength
        )
    )
    out[
        "Statistical League"
    ] = (
        out[
            "_raw_league"
        ]
        .map(
            display_league
        )
    )
    out[
        "Current Value €M"
    ] = (
        out[
            "Model Value €M"
        ]
    )
    out[
        "Upgrade"
    ] = (
        out[
            "Target Score"
        ]
        - out[
            "Current Benchmark"
        ]
    )
    return (
        out.dropna(
            subset=[
                "Target Score"
            ]
        )
        .reset_index(
            drop=True
        )
    )
# ======================================================================================
# BUILD ROLE POOLS
# ======================================================================================
def build_roles(
    benchmarks: dict[str, float],
    role_context: pd.DataFrame,
    league_strength_map: dict[str, float],
    workhorse_context: dict[str, dict],
    role_alias_lookup: dict[str, dict],
) -> list[pd.DataFrame]:
    frames = []
    def add(
        role,
        filename,
        player_col,
        score_col,
        club_col=None,
        league_col=None,
        age_col=None,
        value_col=None,
        minutes_col=None,
        exclude_mask=None,
    ):
        if role not in benchmarks:
            return
        df = pd.read_csv(
            SHORTLIST_DIR
            / filename
        )
        local_exclude = (
            exclude_mask(
                df
            )
            if callable(
                exclude_mask
            )
            else exclude_mask
        )
        frames.append(
            standardize(
                df=df,
                role=role,
                player_col=player_col,
                score_col=score_col,
                benchmark=benchmarks[
                    role
                ],
                role_context=role_context,
                league_strength_map=league_strength_map,
                workhorse_context=workhorse_context,
                role_alias_lookup=role_alias_lookup,
                club_col=club_col,
                league_col=league_col,
                age_col=age_col,
                value_col=value_col,
                minutes_col=minutes_col,
                exclude_mask=local_exclude,
            )
        )
    # ==================================================================================
    # GK
    # ==================================================================================
    add(
        "Sweeper / Build-Up GK",
        "goalkeeper_jaissle_detailed.csv",
        "Player",
        "Jaissle GK Score",
        club_col="Club",
        exclude_mask=lambda df:
        df[
            "Club"
        ]
        .astype(str)
        .str.contains(
            "Newcastle",
            case=False,
            na=False,
        ),
    )
    # ==================================================================================
    # CB
    # ==================================================================================
    add(
        "High-Line CB",
        "center_back_jaissle_detailed.csv",
        "player_name",
        "jaissle_cb_score",
        club_col="club_name",
        league_col="league_slug",
        age_col="age",
        value_col="market_value_eur_m",
        minutes_col="minutes",
    )
    # ==================================================================================
    # LEFT WB / FB
    #
    # IMPORTANT:
    # Hall's Newcastle benchmark is a Jaissle composite.
    #
    # External LBs must therefore use the SAME scoring structure:
    #
    # 40% role fit
    # 30% attacking creation
    # 30% physical activity
    # ==================================================================================
    role = (
        "Left WB / FB"
    )
    if role in benchmarks:
        lb_df = pd.read_csv(
            SHORTLIST_DIR
            / "left_wingback_recruitment_progressive.csv"
        )
        lb_df[
            "_jaissle_fb_score"
        ] = (
            0.40
            * pd.to_numeric(
                lb_df[
                    "role_fit"
                ],
                errors="coerce",
            )
            + 0.30
            * pd.to_numeric(
                lb_df[
                    "attacking_creation_index"
                ],
                errors="coerce",
            )
            + 0.30
            * pd.to_numeric(
                lb_df[
                    "physical_activity_index"
                ],
                errors="coerce",
            )
        )
        exclude = (
            lb_df[
                "player_name"
            ]
            .astype(str)
            .isin(
                [
                    "L. Hall",
                    "Lewis Hall",
                ]
            )
        )
        frames.append(
            standardize(
                df=lb_df,
                role=role,
                player_col="player_name",
                score_col="_jaissle_fb_score",
                benchmark=benchmarks[
                    role
                ],
                role_context=role_context,
                league_strength_map=league_strength_map,
                workhorse_context=workhorse_context,
                role_alias_lookup=role_alias_lookup,
                club_col="club_name",
                league_col="league_slug",
                age_col="age",
                value_col="market_value_eur_m",
                minutes_col="minutes",
                exclude_mask=exclude,
            )
        )
    # ==================================================================================
    # RIGHT WB / FB
    # ==================================================================================
    add(
        "Right WB / FB",
        "right_back_jaissle_detailed.csv",
        "player_name",
        "jaissle_fb_score",
        club_col="club_name",
        league_col="league_slug",
        age_col="age",
        value_col="market_value_eur_m",
        minutes_col="minutes",
        exclude_mask=lambda df:
        df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .eq(
            "2025/26 REFERENCE"
        ),
    )
    # ==================================================================================
    # HOLDING CONTROLLER
    # ==================================================================================
    add(
        "Holding Controller",
        "holding_controller_jaissle_detailed.csv",
        "Player",
        "Jaissle Controller",
        club_col="Club",
        age_col="Age",
        value_col="Value €M",
        minutes_col="Minutes",
        exclude_mask=lambda df:
        df[
            "NUFC Benchmark"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "CURRENT"
        ),
    )
    # ==================================================================================
    # WORKHORSE 8
    # ==================================================================================
    add(
        "Advanced / Workhorse 8",
        "workhorse_cm_jaissle_detailed.csv",
        "player_name",
        "jaissle_workhorse_score",
        club_col="club_name",
        league_col="league_slug",
        age_col="age",
        value_col="market_value_eur_m",
        minutes_col="minutes",
    )
    # ==================================================================================
    # CREATIVE 10
    # ==================================================================================
    add(
        "Creative 10",
        "creative_10_jaissle_detailed.csv",
        "Player",
        "CAM Fit",
        club_col="Club",
        league_col="League",
        age_col="Age",
        value_col="Value €M",
    )
    # ==================================================================================
    # WINGERS
    # ==================================================================================
    winger_df = pd.read_csv(
        SHORTLIST_DIR
        / "winger_jaissle_archetypes.csv"
    )
    benchmark_rows = (
        winger_df[
            "is_newcastle_benchmark"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )
    for (
        role,
        score_col,
    ) in [
        (
            "Wide Creator",
            "wide_creator_score",
        ),
        (
            "Vertical / Box Winger",
            "vertical_box_score",
        ),
        (
            "Hybrid Winger",
            "hybrid_score",
        ),
    ]:
        if role not in benchmarks:
            continue
        frames.append(
            standardize(
                df=winger_df,
                role=role,
                player_col="player_name",
                score_col=score_col,
                benchmark=benchmarks[
                    role
                ],
                role_context=role_context,
                league_strength_map=league_strength_map,
                workhorse_context=workhorse_context,
                role_alias_lookup=role_alias_lookup,
                club_col="club_name",
                league_col="league_slug",
                age_col="age",
                value_col="market_value_eur_m",
                minutes_col="minutes",
                exclude_mask=benchmark_rows,
            )
        )
    # ==================================================================================
    # STRIKER
    # ==================================================================================
    add(
        "Focal Striker",
        "focal_striker_jaissle_detailed.csv",
        "player_name",
        "jaissle_striker_score",
        club_col="club_name",
        league_col="league_slug",
        age_col="age",
        value_col="market_value_eur_m",
        minutes_col="minutes",
    )
    return [
        frame
        for frame in frames
        if not frame.empty
    ]
# ======================================================================================
# CURRENT CLUB / VALUE OVERRIDES
# ======================================================================================
def apply_transfer_overrides(
    board: pd.DataFrame,
    overrides: dict[str, dict],
) -> tuple[pd.DataFrame, int]:
    result = board.copy()
    matched = 0
    for idx in result.index:
        player = result.loc[
            idx,
            "Player",
        ]
        record = None
        for alias in name_aliases(
            player
        ):
            if alias in overrides:
                record = overrides[
                    alias
                ]
                break
        if record is None:
            continue
        matched += 1
        if record[
            "current_club"
        ]:
            result.loc[
                idx,
                "TM / Current Club",
            ] = record[
                "current_club"
            ]
        if pd.notna(
            record[
                "current_value"
            ]
        ):
            result.loc[
                idx,
                "Current Value €M",
            ] = record[
                "current_value"
            ]
    return (
        result,
        matched,
    )
# ======================================================================================
# FILTERS
# ======================================================================================
def apply_recruitment_filters(
    board: pd.DataFrame,
) -> pd.DataFrame:
    result = board.copy()
    # ------------------------------------------------------------------
    # CURRENT VALUE CAP
    # ------------------------------------------------------------------
    value = pd.to_numeric(
        result[
            "Current Value €M"
        ],
        errors="coerce",
    )
    over_value = (
        value.notna()
        & (
            value
            > MAX_TARGET_VALUE_EUR_M
        )
    )
    print(
        f"Players removed above "
        f"€{MAX_TARGET_VALUE_EUR_M:.0f}m: "
        f"{int(over_value.sum())}"
    )
    result = result[
        ~over_value
    ].copy()
    # ------------------------------------------------------------------
    # AGE CAP
    # ------------------------------------------------------------------
    age = pd.to_numeric(
        result[
            "Age"
        ],
        errors="coerce",
    )
    too_old = (
        age.notna()
        & (
            age
            > MAX_RECRUITMENT_AGE
        )
    )
    print(
        f"Players removed above age "
        f"{MAX_RECRUITMENT_AGE}: "
        f"{int(too_old.sum())}"
    )
    result = result[
        ~too_old
    ].copy()
    return result.reset_index(
        drop=True
    )
# ======================================================================================
# CLASSIFICATION
# ======================================================================================
def classify_target(
    row: pd.Series,
) -> str:
    suitability = row[
        "Recruitment Suitability Score"
    ]
    upgrade = row[
        "Upgrade"
    ]
    age = row[
        "Age"
    ]
    value = row[
        "Current Value €M"
    ]
    league_strength = row[
        "League Strength"
    ]
    # Immediate starting-quality acquisition
    if (
        suitability >= 85
        and upgrade >= 5
        and league_strength >= 70
    ):
        return "STARTER TARGET"
    # Young player not yet clearing starter threshold
    if (
        pd.notna(age)
        and age <= 22
        and upgrade >= 2
    ):
        return "DEVELOPMENT TARGET"
    # Cheap meaningful upgrade
    if (
        upgrade >= 4
        and pd.notna(value)
        and value <= 15
    ):
        return "VALUE TARGET"
    if upgrade >= 2:
        return "DEPTH / MONITOR"
    return "LOW PRIORITY"
# ======================================================================================
# SCORE BOARD
# ======================================================================================
def score_board(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    # ==================================================================================
    # POSITIVE UPGRADES ONLY
    # ==================================================================================
    result = result[
        result[
            "Upgrade"
        ].notna()
        & (
            result[
                "Upgrade"
            ]
            > 0
        )
    ].copy()
    result = result.reset_index(
        drop=True
    )
    if result.empty:
        return result
    # ==================================================================================
    # QUALITY COMPONENT
    # ==================================================================================
    result[
        "Quality Component"
    ] = (
        result.groupby(
            "Role"
        )[
            "Target Score"
        ]
        .transform(
            percentile_score
        )
    )
    # ==================================================================================
    # RELATIVE UPGRADE COMPONENT
    # ==================================================================================
    result[
        "Upgrade Component"
    ] = (
        result.groupby(
            "Role"
        )[
            "Upgrade"
        ]
        .transform(
            percentile_score
        )
    )
    # ==================================================================================
    # LEAGUE STRENGTH
    # ==================================================================================
    result[
        "League Strength Component"
    ] = pd.to_numeric(
        result[
            "League Strength"
        ],
        errors="coerce",
    ).fillna(
        60
    )
    # ==================================================================================
    # AGE
    # ==================================================================================
    result[
        "Age Component"
    ] = age_component(
        result[
            "Age"
        ]
    )
    # ==================================================================================
    # AFFORDABILITY
    # ==================================================================================
    result[
        "Affordability Component"
    ] = 50.0
    for role, idx in (
        result.groupby(
            "Role"
        ).groups.items()
    ):
        values = pd.to_numeric(
            result.loc[
                idx,
                "Current Value €M",
            ],
            errors="coerce",
        )
        valid = values.notna()
        if valid.sum() < 3:
            continue
        ranks = (
            values.loc[
                valid
            ]
            .rank(
                pct=True,
                method="average",
            )
            * 100
        )
        result.loc[
            ranks.index,
            "Affordability Component",
        ] = (
            100
            - ranks
        )
    # ==================================================================================
    # BASE RECRUITMENT SUITABILITY
    #
    # RIGHT-NOW Newcastle model
    #
    # 45% tactical performance
    # 25% improvement over Newcastle
    # 25% proven league environment
    #  5% age
    # ==================================================================================
    result[
        "Base Suitability Score"
    ] = (
        TACTICAL_QUALITY_WEIGHT
        * result[
            "Quality Component"
        ]
        + UPGRADE_WEIGHT
        * result[
            "Upgrade Component"
        ]
        + LEAGUE_STRENGTH_WEIGHT
        * result[
            "League Strength Component"
        ]
        + AGE_WEIGHT
        * result[
            "Age Component"
        ]
    )
    # ==================================================================================
    # ABSOLUTE UPGRADE ADJUSTMENT
    #
    # Relative percentile alone can make a +0.5 player look elite when the
    # position already has an excellent Newcastle starter.
    #
    # +8 or more receives no penalty.
    # ==================================================================================
    result[
        "Absolute Upgrade Factor"
    ] = np.clip(
        result[
            "Upgrade"
        ]
        / 8.0,
        0.0,
        1.0,
    )
    result[
        "Upgrade Adjustment"
    ] = (
        0.65
        + 0.35
        * result[
            "Absolute Upgrade Factor"
        ]
    )
    result[
        "Recruitment Suitability Score"
    ] = (
        result[
            "Base Suitability Score"
        ]
        * result[
            "Upgrade Adjustment"
        ]
    )
    # ==================================================================================
    # VALUE SCORE
    #
    # Proven talent is already represented in suitability.
    #
    # Value then asks whether the acquisition makes financial/squad-building sense.
    # ==================================================================================
    result[
        "Value Score"
    ] = (
        VALUE_SUITABILITY_WEIGHT
        * result[
            "Recruitment Suitability Score"
        ]
        + VALUE_AFFORDABILITY_WEIGHT
        * result[
            "Affordability Component"
        ]
        + VALUE_AGE_WEIGHT
        * result[
            "Age Component"
        ]
        + VALUE_UPGRADE_WEIGHT
        * result[
            "Upgrade Component"
        ]
    )
    # ==================================================================================
    # LEAGUE-ADJUSTED PROFILE
    #
    # Interpretive output only.
    #
    # Higher league weight here too because this is intended to express
    # confidence in immediate translation.
    # ==================================================================================
    result[
        "League-Adjusted Profile"
    ] = (
        0.70
        * result[
            "Target Score"
        ]
        + 0.30
        * result[
            "League Strength"
        ]
    )
    # ==================================================================================
    # ORDER
    # ==================================================================================
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
    result[
        "Suitability Rank"
    ] = (
        result.groupby(
            "Role"
        )[
            "Recruitment Suitability Score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
    )
    result[
        "Value Rank"
    ] = (
        result.groupby(
            "Role"
        )[
            "Value Score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(
            int
        )
    )
    result[
        "Class"
    ] = result.apply(
        classify_target,
        axis=1,
    )
    result = result.sort_values(
        [
            "_role_order",
            "Suitability Rank",
            "Value Rank",
        ]
    )
    return result

# ======================================================================================
# EXTENDED RECRUITMENT SCOUTING SCREENS
#
# These screens are diagnostic / recruitment-support outputs. They DO NOT alter the
# core suitability/value board above.
# ======================================================================================

def _safe_read_table(path: Path) -> pd.DataFrame:
    """Read CSV/parquet if it exists; otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[extended scouting] could not read {path.name}: {exc}")
        return pd.DataFrame()


def _numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _text(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def _pct_or_neutral(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(50.0, index=s.index, dtype="float64")
    if valid.sum() >= 2:
        out.loc[valid] = s.loc[valid].rank(pct=True, method="average") * 100.0
    elif valid.sum() == 1:
        out.loc[valid] = 50.0
    return out


def _player_matches(value, aliases: set[str]) -> bool:
    return bool(name_aliases(value) & aliases)


def _canonical_target_for_player(value) -> str | None:
    player_aliases = name_aliases(value)
    if not player_aliases:
        return None
    for canonical, variants in TARGET_DEEP_DIVE_ALIASES.items():
        aliases = set()
        for variant in [canonical, *variants]:
            aliases.update(name_aliases(variant))
        if player_aliases & aliases:
            return canonical
    return None


def _candidate_metric_sources() -> list[Path]:
    # Newcastle-specific files come first so current Newcastle players use the
    # already-built club analysis rather than being represented only by the
    # external/historical recruitment pools.
    paths = [
        NEWCASTLE_ROLE_BENCHMARKS_FILE,
        NEWCASTLE_OPTIMAL_ROLES_FILE,
        FULL_ROLE_FILE,
        BSD_DIR / "europe_2025_26_player_season_metrics.parquet",
        BSD_DIR / "player_season_metrics.parquet",
        BSD_DIR / "player_season_metrics_enriched.parquet",
        POSITION_HISTORY_FILE,
        SHORTLIST_DIR / "center_back_jaissle_detailed.csv",
        SHORTLIST_DIR / "left_wingback_recruitment_progressive.csv",
        SHORTLIST_DIR / "right_back_jaissle_detailed.csv",
        SHORTLIST_DIR / "workhorse_cm_jaissle_detailed.csv",
        SHORTLIST_DIR / "creative_10_jaissle_detailed.csv",
        SHORTLIST_DIR / "winger_jaissle_archetypes.csv",
        SHORTLIST_DIR / "focal_striker_jaissle_detailed.csv",
    ]
    seen = set()
    result = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _newcastle_target_board_row(canonical: str) -> dict | None:
    """Return the best row for a current Newcastle player from club-specific CSVs.

    This deliberately uses the already-generated Newcastle analysis first.  It
    prevents Osula, Touré and Elanga from being labelled ``NOT IN CORE BOARD``
    merely because they are excluded from an external recruitment pool.
    """
    newcastle_targets = {"William Osula", "Bazoumana Touré", "Anthony Elanga"}
    if canonical not in newcastle_targets:
        return None

    aliases = set()
    for variant in [canonical, *TARGET_DEEP_DIVE_ALIASES.get(canonical, [])]:
        aliases.update(name_aliases(variant))

    candidates: list[dict] = []
    for path in [NEWCASTLE_ROLE_BENCHMARKS_FILE, NEWCASTLE_OPTIMAL_ROLES_FILE]:
        df = _safe_read_table(path)
        if df.empty:
            continue
        player_col = find_col(df, ["Player", "player_name", "player", "name"])
        if player_col is None:
            continue
        matched = df[df[player_col].map(lambda x: _player_matches(x, aliases))].copy()
        if matched.empty:
            continue

        role_col = find_col(matched, ["Role", "Optimal Role", "Best Role", "role"])
        score_col = find_col(matched, ["Score", "Role Score", "Fit", "role_fit", "score"])
        fit_col = find_col(matched, ["Fit", "Role Fit", "role_fit"])
        minutes_col = find_col(matched, ["Minutes", "minutes"])
        value_col = find_col(matched, ["Value €M", "market_value_eur_m", "Value"])
        archetype_col = find_col(matched, ["Archetype", "Best Archetype", "archetype"])

        if score_col is not None:
            matched["_selection_score"] = pd.to_numeric(matched[score_col], errors="coerce")
            matched = matched.sort_values("_selection_score", ascending=False, na_position="last")
        row = matched.iloc[0]
        candidates.append({
            "Matched Player": row.get(player_col, canonical),
            "Best Role": row.get(role_col, "NEWCASTLE SQUAD") if role_col else "NEWCASTLE SQUAD",
            "Target Score": pd.to_numeric(pd.Series([row.get(score_col, np.nan)]), errors="coerce").iloc[0] if score_col else np.nan,
            "Role Fit": pd.to_numeric(pd.Series([row.get(fit_col, np.nan)]), errors="coerce").iloc[0] if fit_col else np.nan,
            "Minutes": pd.to_numeric(pd.Series([row.get(minutes_col, np.nan)]), errors="coerce").iloc[0] if minutes_col else np.nan,
            "Current Value €M": pd.to_numeric(pd.Series([row.get(value_col, np.nan)]), errors="coerce").iloc[0] if value_col else np.nan,
            "Archetype": row.get(archetype_col, "") if archetype_col else "",
            "Newcastle Source": path.name,
        })

    if not candidates:
        return None
    candidates.sort(
        key=lambda r: -9999 if pd.isna(r.get("Target Score", np.nan)) else float(r["Target Score"]),
        reverse=True,
    )
    return candidates[0]


def build_priority_target_full_check(board: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one-row target summaries plus a long dump of every available source metric."""
    metric_rows: list[dict] = []

    for source_path in _candidate_metric_sources():
        df = _safe_read_table(source_path)
        if df.empty:
            continue
        player_col = find_col(df, [
            "player_name", "Player", "player", "name", "playerName", "display_name"
        ])
        if player_col is None:
            continue

        canonical = df[player_col].map(_canonical_target_for_player)
        matched = df.loc[canonical.notna()].copy()
        if matched.empty:
            continue
        matched["_canonical_target"] = canonical.loc[matched.index]

        for _, row in matched.iterrows():
            target = row["_canonical_target"]
            matched_name = row[player_col]
            for col in df.columns:
                if col == player_col:
                    continue
                value = row.get(col, np.nan)
                if pd.isna(value):
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                metric_rows.append({
                    "Target": target,
                    "Matched Player": matched_name,
                    "Source": source_path.name,
                    "Metric": col,
                    "Value": value,
                })

    metrics_long = pd.DataFrame(metric_rows)
    if metrics_long.empty:
        metrics_long = pd.DataFrame(columns=[
            "Target", "Matched Player", "Source", "Metric", "Value"
        ])

    summary_rows = []
    metric_preferences = {
        "xG/90": ["xg_per90", "xg_p90", "xG/90"],
        "xA/90": ["xa_per90", "xa_p90", "xA/90"],
        "Shots/90": ["shots_per90", "shots_p90", "Shots/90"],
        "Shots on Target/90": ["shots_on_target_per90", "shots_on_target_p90"],
        "Key Passes/90": ["key_passes_per90", "key_passes_p90", "Key Passes/90"],
        "Successful Dribbles/90": ["successful_dribbles_per90", "successful_dribbles_p90"],
        "Progressive Carries/90": ["progressive_carries_per90", "progressive_carries_p90"],
        "Recoveries/90": ["recoveries_per90", "recoveries_p90", "Recoveries/90"],
        "Interceptions/90": ["interceptions_per90", "interceptions_p90", "Interceptions/90"],
        "Tackles Won/90": ["tackles_won_per90", "tackles_won_p90", "Tackles Won/90"],
        "Duels Won/90": ["duels_won_per90", "duels_won_p90", "Duels Won/90"],
        "Aerials Won/90": ["aerials_won_per90", "aerials_won_p90", "Aerials Won/90"],
        "Pass Accuracy %": ["pass_accuracy_pct", "pass_accuracy", "Pass %"],
        "Touches/90": ["touches_per90", "touches_p90", "Touches/90"],
        "Possession Won Final 3rd/90": ["possession_won_final_third_per90"],
        "Line-Breaking Passes/90": ["line_breaking_passes_per90"],
    }

    for canonical in TARGET_DEEP_DIVE_ALIASES:
        aliases = set()
        for variant in [canonical, *TARGET_DEEP_DIVE_ALIASES[canonical]]:
            aliases.update(name_aliases(variant))

        b = board[board["Player"].map(lambda x: _player_matches(x, aliases))].copy()
        newcastle_row = _newcastle_target_board_row(canonical)
        if newcastle_row is not None:
            base = {
                "Target": canonical,
                "Matched Player": newcastle_row.get("Matched Player", canonical),
                "Roles Found": str(newcastle_row.get("Best Role", "")),
                "Best Role": newcastle_row.get("Best Role", "NEWCASTLE SQUAD"),
                "Current Club": "Newcastle United",
                "Statistical League": "Premier League",
                "League Strength": 100.0,
                "Age": np.nan,
                "Minutes": newcastle_row.get("Minutes", np.nan),
                "Target Score": newcastle_row.get("Target Score", np.nan),
                "Upgrade": 0.0,
                "Current Value €M": newcastle_row.get("Current Value €M", np.nan),
                "Recruitment Suitability Score": np.nan,
                "Value Score": np.nan,
                "Class": "CURRENT NEWCASTLE PLAYER",
                "Newcastle Source": newcastle_row.get("Newcastle Source", ""),
                "Archetype": newcastle_row.get("Archetype", ""),
            }
        elif not b.empty:
            b = b.sort_values(
                ["Recruitment Suitability Score", "Target Score"],
                ascending=False,
                na_position="last",
            )
            best = b.iloc[0]
            roles = ", ".join(sorted(b["Role"].dropna().astype(str).unique()))
            base = {
                "Target": canonical,
                "Matched Player": best.get("Player", ""),
                "Roles Found": roles,
                "Best Role": best.get("Role", ""),
                "Current Club": best.get("TM / Current Club", ""),
                "Statistical League": best.get("Statistical League", ""),
                "League Strength": best.get("League Strength", np.nan),
                "Age": best.get("Age", np.nan),
                "Minutes": best.get("Minutes", np.nan),
                "Target Score": best.get("Target Score", np.nan),
                "Upgrade": best.get("Upgrade", np.nan),
                "Current Value €M": best.get("Current Value €M", np.nan),
                "Recruitment Suitability Score": best.get("Recruitment Suitability Score", np.nan),
                "Value Score": best.get("Value Score", np.nan),
                "Class": best.get("Class", ""),
            }
        else:
            base = {
                "Target": canonical,
                "Matched Player": "",
                "Roles Found": "",
                "Best Role": "NOT IN CORE BOARD",
                "Current Club": "",
                "Statistical League": "",
                "League Strength": np.nan,
                "Age": np.nan,
                "Minutes": np.nan,
                "Target Score": np.nan,
                "Upgrade": np.nan,
                "Current Value €M": np.nan,
                "Recruitment Suitability Score": np.nan,
                "Value Score": np.nan,
                "Class": "DEEP CHECK ONLY",
            }

        target_metrics = metrics_long[metrics_long["Target"] == canonical]
        base["Available Raw Metrics"] = int(len(target_metrics))
        base["Metric Sources"] = ", ".join(sorted(target_metrics["Source"].unique())) if not target_metrics.empty else ""

        if not target_metrics.empty:
            norm_metric = target_metrics["Metric"].map(normalize_name)
            for display, candidates in metric_preferences.items():
                chosen = np.nan
                candidate_norms = {normalize_name(x) for x in candidates}
                hit = target_metrics[norm_metric.isin(candidate_norms)]
                if not hit.empty:
                    numeric = pd.to_numeric(hit["Value"], errors="coerce").dropna()
                    if not numeric.empty:
                        chosen = float(numeric.iloc[0])
                base[display] = chosen
        else:
            for display in metric_preferences:
                base[display] = np.nan

        summary_rows.append(base)

    summary = pd.DataFrame(summary_rows)
    return summary, metrics_long


def build_young_cb_recruitment(league_strength_map: dict[str, float]) -> pd.DataFrame:
    path = SHORTLIST_DIR / "center_back_jaissle_detailed.csv"
    df = _safe_read_table(path)
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["Player"] = _text(df, ["player_name", "Player"])
    out["Club"] = _text(df, ["club_name", "Club"])
    out["League"] = _text(df, ["league_slug", "League"])
    out["Age"] = _numeric(df, ["age", "Age"])
    out["Minutes"] = _numeric(df, ["minutes", "Minutes"])
    out["Value €M"] = _numeric(df, ["market_value_eur_m", "Value €M"])
    out["Jaissle CB"] = _numeric(df, ["jaissle_cb_score", "Jaissle CB"])
    out["CB Fit"] = _numeric(df, ["cb_fit", "role_fit", "CB Fit"])
    out["Build-Up"] = _numeric(df, ["build_up_index", "Build-Up"])
    out["Front-Foot"] = _numeric(df, ["front_foot_defending_index", "Front-Foot"])
    out["Physical"] = _numeric(df, ["physical_security_index", "physical_index", "Physical"])
    out["Recoveries/90"] = _numeric(df, ["recoveries_per90", "Recoveries/90"])
    out["Interceptions/90"] = _numeric(df, ["interceptions_per90", "Interceptions/90"])
    out["Tackles Won/90"] = _numeric(df, ["tackles_won_per90", "Tackles Won/90"])
    out["Duels Won/90"] = _numeric(df, ["duels_won_per90", "Duels Won/90"])
    out["Aerials Won/90"] = _numeric(df, ["aerials_won_per90", "Aerials Won/90"])
    out["Pass %"] = _numeric(df, ["pass_accuracy_pct", "Pass %"])

    out["League Strength"] = out["League"].map(
        lambda x: league_strength_map.get(normalize_league(x), fallback_league_strength(x))
    )

    keep = (
        out["Player"].astype(str).str.len().gt(0)
        & (out["Age"].isna() | (out["Age"] <= MAX_YOUNG_CB_AGE))
        & (out["Minutes"].isna() | (out["Minutes"] >= MIN_YOUNG_CB_MINUTES))
        & (out["Value €M"].isna() | (out["Value €M"] <= MAX_YOUNG_CB_VALUE_EUR_M))
    )
    out = out.loc[keep].copy()
    if out.empty:
        return out

    out["CB Quality Pctl"] = _pct_or_neutral(out["Jaissle CB"])
    out["Build-Up Pctl"] = _pct_or_neutral(out["Build-Up"])
    out["Front-Foot Pctl"] = _pct_or_neutral(out["Front-Foot"])
    out["Physical Pctl"] = _pct_or_neutral(out["Physical"])
    out["Young CB Recruitment Score"] = (
        0.40 * out["CB Quality Pctl"]
        + 0.20 * out["Build-Up Pctl"]
        + 0.20 * out["Front-Foot Pctl"]
        + 0.10 * out["Physical Pctl"]
        + 0.10 * out["League Strength"].fillna(60.0)
    )
    out = out.sort_values(
        ["Young CB Recruitment Score", "Jaissle CB", "Age"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


def _build_position_evidence() -> dict[str, str]:
    evidence: dict[str, set[str]] = {}
    for path in [FULL_ROLE_FILE, POSITION_HISTORY_FILE]:
        df = _safe_read_table(path)
        if df.empty:
            continue
        player_col = find_col(df, ["player_name", "Player", "player", "name"])
        if player_col is None:
            continue
        position_cols = [
            col for col in df.columns
            if any(token in normalize_name(col) for token in [
                "position", "side", "role", "lineup"
            ])
        ]
        if not position_cols:
            continue
        for _, row in df[[player_col, *position_cols]].iterrows():
            key = normalize_name(row[player_col])
            if not key:
                continue
            for col in position_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    evidence.setdefault(key, set()).add(str(value).strip())
    return {k: " | ".join(sorted(v)) for k, v in evidence.items()}


def _wide_position_flag(text: str) -> bool:
    n = normalize_name(text)
    patterns = [
        r"\blw\b", r"\brw\b", r"\bwing", r"\bwide\b",
        r"\bam l\b", r"\bam r\b", r"\baml\b", r"\bamr\b",
        r"\bml\b", r"\bmr\b", r"left forward", r"right forward",
        r"left wing", r"right wing", r"forward left", r"forward right",
    ]
    return any(re.search(pattern, n) for pattern in patterns)


def _collect_player_metric_lookup(metric_candidates: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Collect numeric player metrics from all known historical metric sources.

    The first non-null occurrence of each requested metric is retained for every
    player alias.  This lets diagnostic screens use metrics that are not present
    in the role-specific shortlist CSV itself.
    """
    lookup: dict[str, dict[str, float]] = {}

    for source_path in _candidate_metric_sources():
        df = _safe_read_table(source_path)
        if df.empty:
            continue

        player_col = find_col(df, [
            "player_name", "Player", "player", "name", "playerName", "display_name"
        ])
        if player_col is None:
            continue

        resolved: dict[str, str] = {}
        for output_name, candidates in metric_candidates.items():
            col = find_col(df, candidates)
            if col is not None:
                resolved[output_name] = col

        if not resolved:
            continue

        cols = [player_col, *dict.fromkeys(resolved.values())]
        for _, row in df[cols].iterrows():
            aliases = name_aliases(row[player_col])
            if not aliases:
                continue

            values: dict[str, float] = {}
            for output_name, col in resolved.items():
                val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if pd.notna(val):
                    values[output_name] = float(val)

            if not values:
                continue

            for alias in aliases:
                record = lookup.setdefault(alias, {})
                for key, value in values.items():
                    record.setdefault(key, value)

    return lookup


def _enrich_numeric_metrics_by_name(
    df: pd.DataFrame,
    player_col: str,
    metric_candidates: dict[str, list[str]],
) -> pd.DataFrame:
    """Fill missing numeric metric columns by matching player aliases across sources."""
    result = df.copy()
    lookup = _collect_player_metric_lookup(metric_candidates)

    for output_name in metric_candidates:
        if output_name not in result.columns:
            result[output_name] = np.nan
        result[output_name] = pd.to_numeric(result[output_name], errors="coerce")

    for idx in result.index:
        record = None
        for alias in name_aliases(result.at[idx, player_col]):
            if alias in lookup:
                record = lookup[alias]
                break
        if record is None:
            continue

        for output_name, value in record.items():
            if pd.isna(result.at[idx, output_name]):
                result.loc[idx, output_name] = value

    return result


def _position_side_flags(text: str) -> tuple[bool, bool, bool]:
    """Return (left, right, fullback_or_wingback) evidence from position text.

    Football datasets commonly encode two-sided defenders as D(LR), D(RL),
    WB(LR), FB(LR), DRL, etc.  normalize_name() turns several of those into
    strings such as ``d lr``; the original parser missed them and therefore
    rejected genuine utility fullbacks.
    """
    raw = "" if text is None else str(text)
    n = normalize_name(raw)

    # Explicit both-side encodings.  These are strongest evidence because they
    # say the player can operate on left AND right, rather than merely containing
    # an unrelated letter somewhere in the position description.
    both_patterns = [
        r"\b(?:d|df|defender|fb|full back|fullback|wb|wing back|wingback)\s*(?:lr|rl)\b",
        r"\b(?:dlr|drl|fblr|fbrl|wblr|wbrl)\b",
        r"\blb\s*(?:[/,+&]|and)?\s*rb\b",
        r"\brb\s*(?:[/,+&]|and)?\s*lb\b",
        r"\blwb\s*(?:[/,+&]|and)?\s*rwb\b",
        r"\brwb\s*(?:[/,+&]|and)?\s*lwb\b",
        r"left\s*(?:and|/|&)?\s*right\s*(?:back|fullback|wingback)",
        r"right\s*(?:and|/|&)?\s*left\s*(?:back|fullback|wingback)",
        r"both\s+sides?",
        r"either\s+side",
    ]
    explicit_both = any(re.search(p, n) for p in both_patterns)

    left_patterns = [
        r"\blb\b", r"\blwb\b", r"left back", r"left full", r"left wing back",
        r"left wingback", r"\bd l\b", r"\bwb l\b", r"defender left",
        r"full back left", r"fullback left", r"wing back left", r"wingback left",
    ]
    right_patterns = [
        r"\brb\b", r"\brwb\b", r"right back", r"right full", r"right wing back",
        r"right wingback", r"\bd r\b", r"\bwb r\b", r"defender right",
        r"full back right", r"fullback right", r"wing back right", r"wingback right",
    ]
    fb_patterns = [
        r"fullback", r"full back", r"wingback", r"wing back", r"\bfb\b", r"\bwb\b",
        r"left back", r"right back", r"\blb\b", r"\brb\b", r"\blwb\b", r"\brwb\b",
        r"vertical full back", r"vertical wingback", r"\bd\s*(?:lr|rl)\b",
    ]

    left = explicit_both or any(re.search(p, n) for p in left_patterns)
    right = explicit_both or any(re.search(p, n) for p in right_patterns)
    fullback = explicit_both or any(re.search(p, n) for p in fb_patterns)
    return left, right, fullback


def build_bona_fide_winger_recruitment(
    board: pd.DataFrame,
    league_strength_map: dict[str, float],
) -> pd.DataFrame:
    path = SHORTLIST_DIR / "winger_jaissle_archetypes.csv"
    df = _safe_read_table(path)
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["Player"] = _text(df, ["player_name", "Player"])
    out["Club"] = _text(df, ["club_name", "Club"])
    out["League"] = _text(df, ["league_slug", "League"])
    out["Age"] = _numeric(df, ["age", "Age"])
    out["Minutes"] = _numeric(df, ["minutes", "Minutes"])
    out["Value €M"] = _numeric(df, ["market_value_eur_m", "Value €M"])
    out["Wide Creator"] = _numeric(df, ["wide_creator_score", "Creator"])
    out["Vertical / Box"] = _numeric(df, ["vertical_box_score", "Vertical/Box"])
    out["Hybrid"] = _numeric(df, ["hybrid_score", "Hybrid"])

    # Start with role-file values where available, then fill the gaps from the
    # broader historical metric sources.  This is the important fix: the winger
    # archetype CSV does not always carry the raw attacking columns itself.
    raw_metric_candidates = {
        "xG/90": ["xg_per90", "xg_p90", "xG/90"],
        "Shots/90": ["shots_per90", "shots_p90", "Shots/90"],
        "Shots on Target/90": ["shots_on_target_per90", "shots_on_target_p90", "Shots on Target/90"],
        "Box Touches/90": ["box_touches_per90", "box_touches_p90", "Box Touches/90"],
        "xA/90": ["xa_per90", "xa_p90", "xA/90"],
        "Key Passes/90": ["key_passes_per90", "key_passes_p90", "Key Passes/90"],
        "Big Chances Created/90": ["big_chances_created_per90", "big_chances_created_p90"],
        "Successful Dribbles/90": ["successful_dribbles_per90", "successful_dribbles_p90"],
        "Progressive Carries/90": ["progressive_carries_per90", "progressive_carries_p90"],
        "Recoveries/90": ["recoveries_per90", "recoveries_p90", "Recoveries/90"],
        "Possession Won Final 3rd/90": ["possession_won_final_third_per90", "possession_won_final_third_p90"],
    }
    for output_name, candidates in raw_metric_candidates.items():
        out[output_name] = _numeric(df, candidates)
    out = _enrich_numeric_metrics_by_name(out, "Player", raw_metric_candidates)

    evidence = _build_position_evidence()
    out["Position Evidence"] = out["Player"].map(
        lambda x: evidence.get(normalize_name(x), "")
    )
    have_position_evidence = out["Position Evidence"].astype(str).str.len().gt(0).any()
    out["Bona Fide Wide"] = out["Position Evidence"].map(_wide_position_flag)

    if not have_position_evidence:
        # The winger source itself is already position-filtered; do not throw the
        # whole pool away if no position-history file is available.
        out["Bona Fide Wide"] = True
        out["Position Evidence"] = "winger shortlist source"

    winger_board = board[board["Role"].isin([
        "Wide Creator", "Vertical / Box Winger", "Hybrid Winger"
    ])].copy()
    metadata = {}
    for _, row in winger_board.sort_values(
        "Recruitment Suitability Score", ascending=False
    ).iterrows():
        for alias in name_aliases(row["Player"]):
            metadata.setdefault(alias, row)

    # Create columns first so pandas never has to expand the frame via .at.
    out["League Strength"] = np.nan
    out["Suitability"] = np.nan

    for idx in out.index:
        rec = None
        for alias in name_aliases(out.at[idx, "Player"]):
            if alias in metadata:
                rec = metadata[alias]
                break

        if rec is not None:
            current_club = rec.get("TM / Current Club", "")
            if pd.notna(current_club) and str(current_club).strip():
                out.loc[idx, "Club"] = str(current_club).strip()

            current_value = rec.get("Current Value €M", np.nan)
            if pd.notna(current_value):
                out.loc[idx, "Value €M"] = current_value

            out.loc[idx, "League Strength"] = rec.get("League Strength", np.nan)
            out.loc[idx, "Suitability"] = rec.get(
                "Recruitment Suitability Score", np.nan
            )
        else:
            out.loc[idx, "League Strength"] = league_strength_map.get(
                normalize_league(out.at[idx, "League"]),
                fallback_league_strength(out.at[idx, "League"]),
            )

    keep = (
        out["Bona Fide Wide"].fillna(False)
        & (out["Age"].isna() | (out["Age"] <= MAX_WINGER_AGE))
        & (out["Minutes"].isna() | (out["Minutes"] >= MIN_WINGER_MINUTES))
        & (out["Value €M"].isna() | (out["Value €M"] <= MAX_WINGER_VALUE_EUR_M))
    )
    out = out.loc[keep].copy()
    if out.empty:
        return out

    # A component is neutral only when its underlying metric is genuinely absent.
    # Once the historical merge succeeds, these percentiles differentiate scorer,
    # creator and dribbler/progression profiles.
    goal_parts = [
        _pct_or_neutral(out["xG/90"]),
        _pct_or_neutral(out["Shots/90"]),
        _pct_or_neutral(out["Shots on Target/90"]),
        _pct_or_neutral(out["Box Touches/90"]),
    ]
    creation_parts = [
        _pct_or_neutral(out["xA/90"]),
        _pct_or_neutral(out["Key Passes/90"]),
        _pct_or_neutral(out["Big Chances Created/90"]),
    ]
    progression_parts = [
        _pct_or_neutral(out["Successful Dribbles/90"]),
        _pct_or_neutral(out["Progressive Carries/90"]),
    ]

    out["Goal Threat"] = sum(goal_parts) / len(goal_parts)
    out["Creation"] = sum(creation_parts) / len(creation_parts)
    out["1v1 / Progression"] = sum(progression_parts) / len(progression_parts)
    out["Raw Attack Metrics Available"] = out[[
        "xG/90", "Shots/90", "Shots on Target/90", "Box Touches/90",
        "xA/90", "Key Passes/90", "Big Chances Created/90",
        "Successful Dribbles/90", "Progressive Carries/90",
    ]].notna().sum(axis=1)

    out["Winger Recruitment Score"] = (
        0.25 * _pct_or_neutral(out["Vertical / Box"])
        + 0.15 * _pct_or_neutral(out["Hybrid"])
        + 0.20 * out["Goal Threat"]
        + 0.15 * out["Creation"]
        + 0.10 * out["1v1 / Progression"]
        + 0.10 * pd.to_numeric(out["League Strength"], errors="coerce").fillna(60.0)
        + 0.05 * age_component(out["Age"])
    )

    def label(row):
        goal = row["Goal Threat"]
        creation = row["Creation"]
        progression = row["1v1 / Progression"]

        if goal >= creation + 10 and goal >= progression:
            return "GOAL-ORIENTED WINGER"
        if creation >= goal + 10 and creation >= progression:
            return "CREATIVE WINGER"
        if progression >= max(goal, creation) + 10:
            return "1V1 / PROGRESSION WINGER"
        return "BALANCED ATTACKER"

    out["Attacking Profile"] = out.apply(label, axis=1)
    out = out.sort_values(
        ["Winger Recruitment Score", "League Strength", "Goal Threat"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


def build_mixed_fullback_recruitment(
    league_strength_map: dict[str, float],
) -> pd.DataFrame:
    """Find genuine two-sided fullback/wingback depth options.

    Unlike the first version, a player does NOT have to be present in both the LB
    and RB shortlist CSVs.  Positional versatility comes from position-history /
    role evidence; football quality comes from whichever detailed fullback source
    contains the player.
    """
    left = _safe_read_table(
        SHORTLIST_DIR / "left_wingback_recruitment_progressive.csv"
    )
    right = _safe_read_table(
        SHORTLIST_DIR / "right_back_jaissle_detailed.csv"
    )
    if left.empty and right.empty:
        return pd.DataFrame()

    def prep(df: pd.DataFrame, source_side: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        o = pd.DataFrame(index=df.index)
        o["Player"] = _text(df, ["player_name", "Player"])
        o["Club"] = _text(df, ["club_name", "Club"])
        o["League"] = _text(df, ["league_slug", "League"])
        o["Age"] = _numeric(df, ["age", "Age"])
        o["Minutes"] = _numeric(df, ["minutes", "Minutes"])
        o["Value €M"] = _numeric(df, ["market_value_eur_m", "Value €M"])

        score = _numeric(
            df,
            ["jaissle_fb_score", "_jaissle_fb_score", "Jaissle FB Score", "Jaissle RB"],
        )
        role_fit = _numeric(df, ["role_fit", "RB Fit", "WB Fit"])
        creation = _numeric(df, ["attacking_creation_index", "Creation"])
        physical = _numeric(df, ["physical_activity_index", "Physical"])

        # Rebuild the common Jaissle FB composite where a file does not already
        # provide it.  The composite is intentionally side-neutral for a depth
        # versatility screen; side eligibility is handled by position evidence.
        rebuilt = 0.40 * role_fit + 0.30 * creation + 0.30 * physical
        score = score.fillna(rebuilt)

        o["Fullback Score"] = score
        o["Role Fit"] = role_fit
        o["Creation"] = creation
        o["Physical"] = physical
        o["xA/90"] = _numeric(df, ["xa_per90", "xA/90"])
        o["Recoveries/90"] = _numeric(df, ["recoveries_per90", "Recoveries/90"])
        o["Interceptions/90"] = _numeric(df, ["interceptions_per90", "Interceptions/90"])
        o["Tackles Won/90"] = _numeric(df, ["tackles_won_per90", "Tackles Won/90"])
        o["Duels Won/90"] = _numeric(df, ["duels_won_per90", "Duels Won/90"])
        o["Metric Source Side"] = source_side
        o["_key"] = o["Player"].map(normalize_name)
        return o

    source_frames = [
        x for x in [prep(left, "LB POOL"), prep(right, "RB POOL")]
        if not x.empty
    ]
    combined = pd.concat(source_frames, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame()

    # Prefer the strongest available fullback sample for each player, while still
    # retaining whether the player appeared in the LB pool, RB pool, or both.
    pool_flags = combined.groupby("_key")["Metric Source Side"].agg(
        lambda s: " + ".join(sorted(set(s.astype(str))))
    )
    combined = combined.sort_values(
        ["Fullback Score", "Minutes"],
        ascending=[False, False],
        na_position="last",
    ).drop_duplicates("_key", keep="first")
    combined["Source Pools"] = combined["_key"].map(pool_flags)

    evidence = _build_position_evidence()
    combined["Position Evidence"] = combined["Player"].map(
        lambda x: evidence.get(normalize_name(x), "")
    )

    # If position history is sparse, also inspect all full-role rows for the player
    # and append those descriptions.  This catches natural-position codes even if
    # player_position_history.parquet is incomplete.
    role_df = _safe_read_table(FULL_ROLE_FILE)
    if not role_df.empty:
        player_col = find_col(role_df, ["player_name", "Player"])
        if player_col is not None:
            pos_cols = [c for c in [
                find_col(role_df, ["natural_position"]),
                find_col(role_df, ["natural_position_code"]),
                find_col(role_df, ["position_group"]),
                find_col(role_df, ["role"]),
                find_col(role_df, ["role_key"]),
            ] if c is not None]

            role_evidence: dict[str, set[str]] = {}
            if pos_cols:
                for _, row in role_df[[player_col, *pos_cols]].iterrows():
                    key = normalize_name(row[player_col])
                    if not key:
                        continue
                    for col in pos_cols:
                        value = row[col]
                        if pd.notna(value) and str(value).strip():
                            role_evidence.setdefault(key, set()).add(str(value).strip())

                for idx in combined.index:
                    key = combined.at[idx, "_key"]
                    extra = " | ".join(sorted(role_evidence.get(key, set())))
                    existing = str(combined.at[idx, "Position Evidence"]).strip()
                    if extra:
                        combined.loc[idx, "Position Evidence"] = (
                            f"{existing} | {extra}" if existing else extra
                        )

    side_flags = combined["Position Evidence"].map(_position_side_flags)
    combined["Left-Side Evidence"] = side_flags.map(lambda x: x[0])
    combined["Right-Side Evidence"] = side_flags.map(lambda x: x[1])
    combined["Fullback Evidence"] = side_flags.map(lambda x: x[2])

    # Supplemental pool evidence: a player appearing in one role pool proves that
    # side, but it does not by itself prove the opposite side.
    combined.loc[
        combined["Source Pools"].str.contains("LB POOL", na=False),
        "Left-Side Evidence",
    ] = True
    combined.loc[
        combined["Source Pools"].str.contains("RB POOL", na=False),
        "Right-Side Evidence",
    ] = True

    out = combined[
        combined["Left-Side Evidence"].fillna(False)
        & combined["Right-Side Evidence"].fillna(False)
    ].copy()

    # If no explicit two-sided position history exists, fall back to players who
    # genuinely appear in both role pools.  This preserves useful output without
    # falsely treating every one-sided fullback as versatile.
    if out.empty:
        out = combined[
            combined["Source Pools"].str.contains("LB POOL", na=False)
            & combined["Source Pools"].str.contains("RB POOL", na=False)
        ].copy()

    if out.empty:
        return pd.DataFrame()

    out["League Strength"] = out["League"].map(
        lambda x: league_strength_map.get(
            normalize_league(x), fallback_league_strength(x)
        )
    )

    keep = (
        (out["Age"].isna() | (out["Age"] <= MAX_MIXED_FB_AGE))
        & (out["Minutes"].isna() | (out["Minutes"] >= MIN_MIXED_FB_MINUTES))
        & (out["Value €M"].isna() | (out["Value €M"] <= MAX_MIXED_FB_VALUE_EUR_M))
        & out["Fullback Score"].notna()
    )
    out = out.loc[keep].copy()
    if out.empty:
        return out

    out["Mixed Fullback Score"] = (
        0.45 * _pct_or_neutral(out["Fullback Score"])
        + 0.20 * _pct_or_neutral(out["Creation"])
        + 0.15 * _pct_or_neutral(out["Physical"])
        + 0.10 * out["League Strength"].fillna(60.0)
        + 0.10 * age_component(out["Age"])
    )

    out = out.sort_values(
        ["Mixed Fullback Score", "Fullback Score", "Value €M"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    keep_cols = [
        "Rank", "Player", "Club", "League", "Age", "Minutes", "Value €M",
        "Fullback Score", "Role Fit", "Creation", "Physical", "xA/90",
        "Recoveries/90", "Interceptions/90", "Tackles Won/90", "Duels Won/90",
        "League Strength", "Source Pools", "Left-Side Evidence",
        "Right-Side Evidence", "Position Evidence", "Mixed Fullback Score",
    ]
    return out[[c for c in keep_cols if c in out.columns]]

def run_extended_recruitment_screens(
    board: pd.DataFrame,
    league_strength_map: dict[str, float],
) -> None:
    print()
    print("=" * 160)
    print("EXTENDED RECRUITMENT SCOUTING")
    print("=" * 160)

    target_summary, target_metrics = build_priority_target_full_check(board)
    target_summary.to_csv(TARGET_CHECK_OUT, index=False)
    target_metrics.to_csv(TARGET_METRICS_OUT, index=False)
    print("\nPRIORITY TARGET FULL CHECK")
    if not target_summary.empty:
        preview_cols = [c for c in [
            "Target", "Best Role", "Current Club", "Age", "Current Value €M",
            "Target Score", "Upgrade", "Recruitment Suitability Score",
            "xG/90", "xA/90", "Shots/90", "Key Passes/90",
            "Recoveries/90", "Duels Won/90", "Available Raw Metrics"
        ] if c in target_summary.columns]
        print(target_summary[preview_cols].to_string(index=False))
    print(f"Saved: {TARGET_CHECK_OUT}")
    print(f"Saved: {TARGET_METRICS_OUT}")

    young_cb = build_young_cb_recruitment(league_strength_map)
    young_cb.to_csv(YOUNG_CB_OUT, index=False)
    print("\nYOUNG HIGH-LINE CB RECRUITMENT — TOP 15")
    if young_cb.empty:
        print("No candidates survived the current young-CB filters.")
    else:
        cols = [c for c in [
            "Rank", "Player", "Club", "League", "Age", "Minutes", "Value €M",
            "Jaissle CB", "Build-Up", "Front-Foot", "Physical", "League Strength",
            "Young CB Recruitment Score"
        ] if c in young_cb.columns]
        print(young_cb.head(15)[cols].to_string(index=False))
    print(f"Saved: {YOUNG_CB_OUT}")

    wingers = build_bona_fide_winger_recruitment(board, league_strength_map)
    wingers.to_csv(BONA_FIDE_WINGER_OUT, index=False)
    print("\nBONA FIDE ATTACKING WINGERS — TOP 20")
    if wingers.empty:
        print("No candidates survived the current bona-fide winger filters.")
    else:
        cols = [c for c in [
            "Rank", "Player", "Club", "Age", "Value €M", "Attacking Profile",
            "Winger Recruitment Score", "Goal Threat", "Creation", "1v1 / Progression",
            "Vertical / Box", "Hybrid", "xG/90", "xA/90", "Shots/90",
            "League Strength", "Position Evidence"
        ] if c in wingers.columns]
        print(wingers.head(20)[cols].to_string(index=False))
    print(f"Saved: {BONA_FIDE_WINGER_OUT}")

    mixed = build_mixed_fullback_recruitment(league_strength_map)
    mixed.to_csv(MIXED_FULLBACK_OUT, index=False)
    print("\nTWO-SIDED FULLBACK / WINGBACK DEPTH — TOP 15")
    if mixed.empty:
        print("No verified two-sided LB/RB candidates survived the current evidence and value filters.")
    else:
        cols = [c for c in [
            "Rank", "Player", "Club", "Age", "Value €M", "Fullback Score",
            "Creation", "Physical", "xA/90", "Source Pools",
            "League Strength", "Mixed Fullback Score", "Position Evidence"
        ] if c in mixed.columns]
        print(mixed.head(15)[cols].to_string(index=False))
    print(f"Saved: {MIXED_FULLBACK_OUT}")

# ======================================================================================
# MAIN
# ======================================================================================
def main():
    benchmarks = (
        load_benchmarks()
    )
    role_context = (
        load_role_context()
    )
    league_strength_map = (
        load_league_strength_map()
    )
    workhorse_context = (
        load_workhorse_cm_context()
    )
    role_alias_lookup = (
        build_role_alias_lookup(
            role_context
        )
    )
    overrides = (
        load_transfer_overrides()
    )
    frames = build_roles(
        benchmarks,
        role_context,
        league_strength_map,
        workhorse_context,
        role_alias_lookup,
    )
    if not frames:
        raise RuntimeError(
            "No recruitment role pools loaded."
        )
    board = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    board, matched = (
        apply_transfer_overrides(
            board,
            overrides,
        )
    )
    board = (
        apply_recruitment_filters(
            board
        )
    )
    print()
    print(
        f"Candidates after recruitment filters: "
        f"{len(board)}"
    )
    board = score_board(
        board
    )
    # ==================================================================================
    # ROUNDING
    # ==================================================================================
    numeric_cols = [
        "Age",
        "Minutes",
        "Target Score",
        "Current Benchmark",
        "Upgrade",
        "Model Value €M",
        "Current Value €M",
        "League Strength",
        "League-Adjusted Profile",
        "Quality Component",
        "Upgrade Component",
        "League Strength Component",
        "Age Component",
        "Affordability Component",
        "Base Suitability Score",
        "Absolute Upgrade Factor",
        "Upgrade Adjustment",
        "Recruitment Suitability Score",
        "Value Score",
    ]
    for col in numeric_cols:
        if col in board.columns:
            board[
                col
            ] = pd.to_numeric(
                board[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )
    # ==================================================================================
    # SAVE
    # ==================================================================================
    save_board = board.drop(
        columns=[
            "_role_order",
            "_name_key",
            "_raw_league",
            "_ctx_league",
            "_ctx_minutes",
            "_ctx_age",
            "_ctx_value",
            "_ctx_club",
            "role_fit",
            "player_name",
        ],
        errors="ignore",
    )
    save_board.to_csv(
        OUT,
        index=False,
    )
    # Extended diagnostic scouting screens; these do not change the core board.
    run_extended_recruitment_screens(
        board=board,
        league_strength_map=league_strength_map,
    )
    # ==================================================================================
    # SUITABILITY BOARD
    # ==================================================================================
    print()
    print(
        "=" * 200
    )
    print(
        "RECRUITMENT SUITABILITY BOARD"
    )
    print(
        "=" * 200
    )
    suitability_top = (
        board[
            board[
                "Suitability Rank"
            ]
            <= 5
        ]
        .sort_values(
            [
                "_role_order",
                "Suitability Rank",
            ]
        )
    )
    suitability_cols = [
        "Role",
        "Suitability Rank",
        "Player",
        "TM / Current Club",
        "Statistical League",
        "League Strength",
        "Age",
        "Target Score",
        "Upgrade",
        "Current Value €M",
        "Recruitment Suitability Score",
        "Value Score",
        "Class",
    ]
    print(
        suitability_top[
            suitability_cols
        ].to_string(
            index=False
        )
    )
    # ==================================================================================
    # VALUE BOARD
    #
    # IMPORTANT:
    # Print Value Rank, not Suitability Rank.
    # ==================================================================================
    print()
    print(
        "=" * 200
    )
    print(
        "TOP 5 VALUE OPTIONS"
    )
    print(
        "=" * 200
    )
    value_top = (
        board[
            board[
                "Value Rank"
            ]
            <= 5
        ]
        .sort_values(
            [
                "_role_order",
                "Value Rank",
            ]
        )
    )
    value_cols = [
        "Role",
        "Value Rank",
        "Player",
        "TM / Current Club",
        "Statistical League",
        "League Strength",
        "Age",
        "Target Score",
        "Upgrade",
        "Current Value €M",
        "Recruitment Suitability Score",
        "Value Score",
        "Class",
    ]
    print(
        value_top[
            value_cols
        ].to_string(
            index=False
        )
    )
    # ==================================================================================
    # SUMMARY
    # ==================================================================================
    print()
    print(
        "Recruitment Suitability:"
    )
    print(
        f"  Tactical quality:   "
        f"{TACTICAL_QUALITY_WEIGHT:.0%}"
    )
    print(
        f"  Newcastle upgrade:  "
        f"{UPGRADE_WEIGHT:.0%}"
    )
    print(
        f"  League strength:    "
        f"{LEAGUE_STRENGTH_WEIGHT:.0%}"
    )
    print(
        f"  Age:                 "
        f"{AGE_WEIGHT:.0%}"
    )
    print(
        "  + absolute upgrade adjustment"
    )
    print()
    print(
        "Value Score:"
    )
    print(
        f"  Suitability:        "
        f"{VALUE_SUITABILITY_WEIGHT:.0%}"
    )
    print(
        f"  Affordability:      "
        f"{VALUE_AFFORDABILITY_WEIGHT:.0%}"
    )
    print(
        f"  Age / development:  "
        f"{VALUE_AGE_WEIGHT:.0%}"
    )
    print(
        f"  Upgrade:            "
        f"{VALUE_UPGRADE_WEIGHT:.0%}"
    )
    print()
    print(
        f"Maximum current market value: "
        f"€{MAX_TARGET_VALUE_EUR_M:.0f}m"
    )
    print(
        f"Maximum normal target age: "
        f"{MAX_RECRUITMENT_AGE}"
    )
    print(
        f"Transfer override aliases loaded: "
        f"{len(overrides)}"
    )
    print(
        f"Recruitment rows matched to overrides: "
        f"{matched}"
    )
    print(
        f"Workhorse CM metadata aliases loaded: "
        f"{len(workhorse_context)}"
    )
    print(
        f"League strength values loaded: "
        f"{len(league_strength_map)}"
    )
    print(
        f"Saved: {OUT}"
    )
if __name__ == "__main__":
    main()
