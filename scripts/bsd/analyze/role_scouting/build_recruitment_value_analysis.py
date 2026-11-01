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


# ======================================================================================
# SETTINGS
# ======================================================================================

MAX_TARGET_VALUE_EUR_M = 100.0
MAX_RECRUITMENT_AGE = 30


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