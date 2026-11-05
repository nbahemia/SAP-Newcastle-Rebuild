#!/usr/bin/env python3
"""

find_final_cb_rw_targets.py



Final focused Newcastle recruitment search.



Outputs:

- Top 25 High-Line Succession CB targets

- Top 10 Elite / Complete RW targets

- Top 30 statistical holding-controller comparisons to Nico González

- Top 15 realistic Nico González backup targets

- Nico KNN visualization in <repo root>/visualizations/



NICO KNN DESIGN

---------------

The KNN uses the four existing Jaissle holding-controller dimensions:



    Transition Protection

    Deep Progression

    Possession Security

    Deep Involvement



The underlying raw statistics remain in the output for interpretation, but are

not included again in KNN distance. This avoids double-counting the information

already summarized by the four role dimensions.



Nico González is the reference observation. His 444-minute sample is allowed

because he is not a transfer candidate. External candidates require at least

900 minutes.



Transfer suitability is applied AFTER similarity:

- age 26-30

- maximum €15m source/project value

- preference for €5-10m backup options



Run from repo root:



    python scripts/bsd/analyze/find_final_cb_rw_targets.py

"""



from __future__ import annotations



from pathlib import Path

from typing import Iterable, Optional

import re

import unicodedata



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler





# ======================================================================================
# PATHS
# ======================================================================================
ROOT = Path(__file__).resolve().parents[3]



SHORTLIST_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"

OUT_DIR = SHORTLIST_DIR / "final_recruitment_focus"



# Root-level visualization folder requested for project graphics.
VIS_DIR = ROOT / "visualizations"



CB_CANDIDATES = [

    SHORTLIST_DIR / "young_cb_recruitment.csv",

    SHORTLIST_DIR / "center_back_jaissle_detailed.csv",

]



WINGER_CANDIDATES = [

    SHORTLIST_DIR / "bona_fide_winger_recruitment.csv",

    SHORTLIST_DIR / "winger_jaissle_archetypes.csv",

    SHORTLIST_DIR / "winger_jaissle_extended.csv",

    SHORTLIST_DIR / "winger_jaissle_detailed.csv",

]



# Confirmed correct 720-row holding-controller source.
NICO_KNN_SOURCE = SHORTLIST_DIR / "holding_controller_jaissle_detailed.csv"

NICO_PERCENTILES_SOURCE = ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_percentiles.parquet"

NICO_METADATA_SOURCE = SHORTLIST_DIR / "europe_2025_26_unique_shortlist.csv"





# ======================================================================================
# SEARCH RULES
# ======================================================================================
CB_MIN_AGE = 16

CB_MAX_AGE = 23

CB_TOP_N = 25



RW_MIN_AGE = 22

RW_MAX_AGE = 28

RW_TOP_N = 10



NICO_NAMES = [

    "Nico González",

    "Nico Gonzalez",

    "N. González",

    "N. Gonzalez",

]



# These four dimensions alone define statistical similarity.
NICO_KNN_FEATURES = [

    "Transition Protection",

    "Deep Progression",

    "Possession Security",

    "Deep Involvement",

]



NICO_KNN_MIN_MINUTES = 900

NICO_KNN_TOP_N = 30



NICO_BACKUP_MIN_AGE = 26

NICO_BACKUP_MAX_AGE = 30

NICO_BACKUP_PREFERRED_FEE = 10.0

NICO_BACKUP_MAX_FEE = 15.0

NICO_BACKUP_TOP_N = 20



NICO_VIS_TOP_N = 10



# Final backup profile: remain Nico-like, but tilt toward stronger defensive and aerial security.
NICO_BACKUP_SIMILARITY_WEIGHT = 0.65

NICO_BACKUP_DEFENSIVE_WEIGHT = 0.25

NICO_BACKUP_AERIAL_WEIGHT = 0.10

NICO_BACKUP_MIN_DEFENSIVE_SCORE = 65.0

NICO_BACKUP_MIN_AERIAL_SCORE = 60.0



# Manually verified 2026 movers are excluded from the final backup shortlist.
# Keep them in the unrestricted KNN comps so the model remains diagnostically useful.
NICO_2026_MOVERS = {

    "Lukasz Lakomy",

    "Kervin Arriaga",

    "Luis Milla",

    "Ylber Ramadani",

    "Roberto Gagliardini",

    "Etienne Camara",

    "Adrian Beck",

    "Bartosz Slisz",

    "Ramiz Zerrouki",

}





# ======================================================================================
# PROJECT-SPECIFIC PLAYER KNOWLEDGE
# ======================================================================================
EXCLUDED_PLAYERS = {

    "Chibuike Nwaiwu",

    "Ahmetcan Kaplan",

    "Julio Enciso",

    "M. Freckleton",

    "B. Okoh",

    "L. Graham",

    "Y. Bojang",

    "A. Amenda",

    "A. Silva",

    "I. Ba",

    "D. Ghilardi",

    "B. Özyurt",

}



PROJECT_FEE_OVERRIDES = {

    "Ousmane Diao": 11.0,

    "Kike Salas": 17.0,

    "Francisco Conceição": 35.0,

}



ALIASES = {

    "Ousmane Diao": ["O. Diao"],

    "Kike Salas": ["K. Salas", "Enrique Salas"],

    "Francisco Conceição": ["F. Conceição", "Francisco Conceicao"],

    "Chibuike Nwaiwu": ["C. Nwaiwu"],

    "Ahmetcan Kaplan": ["A. Kaplan"],

    "Julio Enciso": ["J. Enciso"],

    "Nico González": ["Nico Gonzalez", "N. González", "N. Gonzalez"],

}





# ======================================================================================
# GENERAL HELPERS
# ======================================================================================
def normalize(value: object) -> str:

    if value is None or pd.isna(value):

        return ""



    text = unicodedata.normalize("NFKD", str(value))

    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return text.strip()





def choose_column(

    df: pd.DataFrame,

    candidates: Iterable[str],

) -> Optional[str]:

    lookup = {normalize(col): col for col in df.columns}



    for candidate in candidates:

        hit = lookup.get(normalize(candidate))

        if hit is not None:

            return hit



    return None





def choose_existing(paths: list[Path]) -> Path:

    for path in paths:

        if path.exists():

            return path



    raise FileNotFoundError(

        "None of these files exist:\n"

        + "\n".join(f"  {path}" for path in paths)

    )





def numeric(

    df: pd.DataFrame,

    candidates: list[str],

    default: float = np.nan,

) -> pd.Series:

    col = choose_column(df, candidates)



    if col is None:

        return pd.Series(default, index=df.index, dtype=float)



    return pd.to_numeric(df[col], errors="coerce")





def text_series(

    df: pd.DataFrame,

    candidates: list[str],

) -> pd.Series:

    col = choose_column(df, candidates)



    if col is None:

        return pd.Series("", index=df.index, dtype=object)



    return df[col].fillna("").astype(str)





def percentile(series: pd.Series) -> pd.Series:

    values = pd.to_numeric(series, errors="coerce")



    if values.notna().sum() <= 1:

        return pd.Series(50.0, index=series.index)



    return values.rank(pct=True, method="average") * 100.0





def inverse_percentile(series: pd.Series) -> pd.Series:

    values = pd.to_numeric(series, errors="coerce")

    result = 100.0 - percentile(values)

    return result.where(values.notna(), 50.0)





def canonical_player_name(name: object) -> str:

    n = normalize(name)



    for canonical, aliases in ALIASES.items():

        choices = {normalize(canonical), *[normalize(alias) for alias in aliases]}



        if n in choices:

            return canonical



    return str(name).strip()





def excluded_mask(players: pd.Series) -> pd.Series:

    excluded = {normalize(player) for player in EXCLUDED_PLAYERS}



    for canonical in EXCLUDED_PLAYERS:

        for alias in ALIASES.get(canonical, []):

            excluded.add(normalize(alias))



    return players.astype(str).map(normalize).isin(excluded)





def fee_override(player: str, source_value: float) -> float:

    canonical = canonical_player_name(player)



    if canonical in PROJECT_FEE_OVERRIDES:

        return float(PROJECT_FEE_OVERRIDES[canonical])



    return source_value





def useful_columns(

    df: pd.DataFrame,

    columns: list[str],

) -> list[str]:

    kept = []



    for col in columns:

        if col not in df.columns:

            continue



        series = df[col]



        if pd.api.types.is_numeric_dtype(series):

            values = pd.to_numeric(series, errors="coerce")



            if values.notna().any():

                kept.append(col)



        else:

            values = series.fillna("").astype(str).str.strip()



            if values.ne("").any():

                kept.append(col)



    return kept





def print_table(

    title: str,

    df: pd.DataFrame,

    columns: list[str],

) -> None:

    print()

    print("=" * 170)

    print(title)

    print("=" * 170)



    shown = useful_columns(df, columns)



    if not shown:

        print("No useful columns available.")

        return



    print(

        df[shown].to_string(

            index=False,

            float_format=lambda x: f"{x:.2f}",

        )

    )





# ======================================================================================
# CENTER BACK SEARCH
# ======================================================================================
def build_cb_targets() -> tuple[pd.DataFrame, Path]:

    path = choose_existing(CB_CANDIDATES)

    raw = pd.read_csv(path)



    player = text_series(raw, ["Player", "player_name", "Name"])

    age = numeric(raw, ["Age", "age"])

    club = text_series(raw, ["Club", "Current Club", "Squad", "Team"])

    league = text_series(raw, ["League", "league"])



    preferred_foot = text_series(

        raw,

        [

            "Preferred Foot",

            "preferred_foot",

            "PreferredFoot",

            "dominant_foot",

            "Dominant Foot",

            "foot",

            "Foot",

        ],

    )



    fee = numeric(

        raw,

        ["Value €M", "Current Value €M", "market_value_eur_m", "Value"],

    )



    overall = numeric(

        raw,

        [

            "Young CB Recruitment Score",

            "Young CB Score",

            "Jaissle CB",

            "jaissle_cb_score",

            "Role Score",

        ],

    )



    buildup = numeric(

        raw,

        ["Build-Up", "Build Up", "Build-Up Score", "Progression"],

    )



    front_foot = numeric(

        raw,

        [

            "Front-Foot",

            "Front Foot",

            "Front-Foot Score",

            "Defensive Activity",

        ],

    )



    physical = numeric(

        raw,

        ["Physical", "Physical Score", "Athleticism"],

    )



    line_breaking = numeric(

        raw,

        ["Line-Breaking Passes/90", "line_breaking_passes_per90"],

    )



    pass_pct = numeric(

        raw,

        ["Pass %", "Pass Accuracy %", "pass_accuracy_pct"],

    )



    duel_pct = numeric(

        raw,

        ["Duel Win %", "Duels Won %", "duel_win_pct"],

    )



    aerial_pct = numeric(

        raw,

        ["Aerial Win %", "Aerial Duels Won %", "aerial_win_pct"],

    )



    interceptions = numeric(

        raw,

        ["Interceptions/90", "interceptions_per90"],

    )



    recoveries = numeric(

        raw,

        ["Recoveries/90", "recoveries_per90"],

    )



    tackles = numeric(

        raw,

        [

            "Tackles Won/90",

            "Tackles/90",

            "tackles_won_per90",

            "tackles_per90",

        ],

    )



    duels_won = numeric(

        raw,

        ["Duels Won/90", "duels_won_per90"],

    )



    aerials_won = numeric(

        raw,

        ["Aerials Won/90", "Aerial Duels Won/90", "aerials_won_per90"],

    )



    successful_passes = numeric(

        raw,

        [

            "Successful Passes/90",

            "successful_passes_per90",

            "successful_passes_p90",

        ],

    )



    df = pd.DataFrame({

        "Player": player,

        "Club": club,

        "League": league,

        "Age": age,

        "Preferred Foot": preferred_foot,

        "Value €M": fee,

        "Existing CB Score": overall,

        "Build-Up": buildup,

        "Front-Foot": front_foot,

        "Physical": physical,

        "Line-Breaking/90": line_breaking,

        "Pass %": pass_pct,

        "Duel Win %": duel_pct,

        "Aerial Win %": aerial_pct,

        "Interceptions/90": interceptions,

        "Recoveries/90": recoveries,

        "Tackles Won/90": tackles,

        "Duels Won/90": duels_won,

        "Aerials Won/90": aerials_won,

        "Successful Passes/90": successful_passes,

    })



    df["Player"] = df["Player"].map(canonical_player_name)



    df = df[

        df["Age"].between(

            CB_MIN_AGE,

            CB_MAX_AGE,

            inclusive="both",

        )

    ].copy()



    df = df[

        ~excluded_mask(df["Player"])

    ].copy()



    df["Project Fee €M"] = [

        fee_override(player_name, source_fee)

        for player_name, source_fee in zip(

            df["Player"],

            df["Value €M"],

        )

    ]



    build_rank = percentile(df["Build-Up"])



    raw_progression = (

        0.55 * percentile(df["Line-Breaking/90"])

        + 0.45 * percentile(df["Pass %"])

    )



    build_rank = build_rank.where(

        df["Build-Up"].notna(),

        raw_progression,

    )



    defensive_rank = percentile(df["Front-Foot"])



    raw_defense = (

        0.55 * percentile(df["Interceptions/90"])

        + 0.45 * percentile(df["Recoveries/90"])

    )



    defensive_rank = defensive_rank.where(

        df["Front-Foot"].notna(),

        raw_defense,

    )



    ground_rank = percentile(df["Duel Win %"])

    physical_rank = percentile(df["Physical"])

    aerial_rank = percentile(df["Aerial Win %"])



    ground_rank = ground_rank.where(

        df["Duel Win %"].notna(),

        physical_rank,

    )



    aerial_rank = aerial_rank.where(

        df["Aerial Win %"].notna(),

        physical_rank,

    )



    age_rank = 100.0 - percentile(df["Age"])

    fee_rank = inverse_percentile(df["Project Fee €M"])



    df["CB Football Score"] = (

        0.30 * build_rank

        + 0.25 * defensive_rank

        + 0.20 * ground_rank

        + 0.15 * aerial_rank

        + 0.10 * percentile(df["Existing CB Score"])

    )



    df["CB Final Score"] = (

        0.90 * df["CB Football Score"]

        + 0.05 * age_rank

        + 0.05 * fee_rank

    )



    df = (

        df.sort_values(

            ["CB Final Score", "CB Football Score"],

            ascending=False,

        )

        .drop_duplicates("Player")

        .head(CB_TOP_N)

        .reset_index(drop=True)

    )



    df.insert(

        0,

        "Rank",

        np.arange(1, len(df) + 1),

    )



    return df, path





# ======================================================================================
# WINGER SEARCH
# ======================================================================================
def build_rw_targets() -> tuple[pd.DataFrame, Path]:

    path = choose_existing(WINGER_CANDIDATES)

    raw = pd.read_csv(path)



    player = text_series(raw, ["Player", "player_name", "Name"])

    age = numeric(raw, ["Age", "age"])

    club = text_series(raw, ["Club", "Current Club", "Squad", "Team"])

    league = text_series(raw, ["League", "league"])



    fee = numeric(

        raw,

        ["Value €M", "Current Value €M", "market_value_eur_m", "Value"],

    )



    overall = numeric(

        raw,

        [

            "Winger Recruitment Score",

            "Best Score",

            "overall_winger_score",

            "Hybrid",

        ],

    )



    creation = numeric(

        raw,

        ["Creation", "Creator", "wide_creator_score"],

    )



    one_v_one = numeric(

        raw,

        ["1v1 / Progression", "1v1/Progression", "Progression"],

    )



    goal_threat = numeric(

        raw,

        ["Goal Threat", "Vertical/Box", "vertical_box_score"],

    )



    xg = numeric(raw, ["xG/90", "xg_per90"])

    xa = numeric(raw, ["xA/90", "xa_per90"])

    shots = numeric(raw, ["Shots/90", "shots_per90"])



    dribbles = numeric(

        raw,

        ["Successful Dribbles/90", "successful_dribbles_per90"],

    )



    box_touches = numeric(

        raw,

        ["Touches In Box/90", "Box Touches/90", "box_touches_per90"],

    )



    key_passes = numeric(

        raw,

        ["Chances Created/90", "Key Passes/90", "key_passes_per90"],

    )



    recoveries = numeric(

        raw,

        ["Recoveries/90", "recoveries_per90"],

    )



    final_third_wins = numeric(

        raw,

        [

            "Possession Won Final 3rd/90",

            "possession_won_final_third_per90",

        ],

    )



    df = pd.DataFrame({

        "Player": player,

        "Club": club,

        "League": league,

        "Age": age,

        "Value €M": fee,

        "Winger Score": overall,

        "Creation": creation,

        "1v1 / Progression": one_v_one,

        "Goal Threat": goal_threat,

        "xG/90": xg,

        "xA/90": xa,

        "xG+xA/90": xg + xa,

        "Shots/90": shots,

        "Successful Dribbles/90": dribbles,

        "Box Touches/90": box_touches,

        "Chances Created/90": key_passes,

        "Recoveries/90": recoveries,

        "Final 3rd Wins/90": final_third_wins,

    })



    df["Player"] = df["Player"].map(canonical_player_name)



    df = df[

        df["Age"].between(

            RW_MIN_AGE,

            RW_MAX_AGE,

            inclusive="both",

        )

    ].copy()



    df = df[

        ~excluded_mask(df["Player"])

    ].copy()



    df["Project Fee €M"] = [

        fee_override(player_name, source_fee)

        for player_name, source_fee in zip(

            df["Player"],

            df["Value €M"],

        )

    ]



    creation_rank = percentile(df["Creation"])

    one_v_one_rank = percentile(df["1v1 / Progression"])

    threat_rank = percentile(df["Goal Threat"])

    overall_rank = percentile(df["Winger Score"])



    raw_creation = (

        0.60 * percentile(df["xA/90"])

        + 0.40 * percentile(df["Chances Created/90"])

    )



    raw_one_v_one = percentile(

        df["Successful Dribbles/90"]

    )



    raw_threat = (

        0.45 * percentile(df["xG/90"])

        + 0.25 * percentile(df["Shots/90"])

        + 0.30 * percentile(df["Box Touches/90"])

    )



    creation_rank = creation_rank.where(

        df["Creation"].notna(),

        raw_creation,

    )



    one_v_one_rank = one_v_one_rank.where(

        df["1v1 / Progression"].notna(),

        raw_one_v_one,

    )



    threat_rank = threat_rank.where(

        df["Goal Threat"].notna(),

        raw_threat,

    )



    df["Elite RW Football Score"] = (

        0.34 * creation_rank

        + 0.31 * one_v_one_rank

        + 0.25 * threat_rank

        + 0.10 * overall_rank

    )



    component_floor = pd.concat(

        [

            creation_rank,

            one_v_one_rank,

            threat_rank,

        ],

        axis=1,

    ).min(axis=1)



    df["Complete Profile Floor"] = component_floor



    df["Elite RW Score"] = (

        0.85 * df["Elite RW Football Score"]

        + 0.15 * component_floor

    )



    strongest = pd.concat(

        {

            "Creator": creation_rank,

            "1v1": one_v_one_rank,

            "Goal Threat": threat_rank,

        },

        axis=1,

    )



    df["Primary Strength"] = strongest.idxmax(axis=1)



    raw_quality_mask = pd.Series(

        True,

        index=df.index,

    )



    raw_quality_mask &= (

        df["xG+xA/90"].isna()

        | (df["xG+xA/90"] >= 0.35)

    )



    raw_quality_mask &= (

        df["Successful Dribbles/90"].isna()

        | (df["Successful Dribbles/90"] >= 1.25)

    )



    qualified = df[raw_quality_mask].copy()



    if len(qualified) >= RW_TOP_N:

        df = qualified



    df = (

        df.sort_values(

            [

                "Elite RW Score",

                "Elite RW Football Score",

                "Complete Profile Floor",

            ],

            ascending=False,

        )

        .drop_duplicates("Player")

        .head(RW_TOP_N)

        .reset_index(drop=True)

    )



    df.insert(

        0,

        "Rank",

        np.arange(1, len(df) + 1),

    )



    return df, path





# ======================================================================================
# NICO GONZÁLEZ KNN
# ======================================================================================
def build_nico_knn_targets() -> tuple[

    pd.DataFrame,

    pd.DataFrame,

    pd.DataFrame,

    Path,

]:

    """

    Build Nico Gonzalez similarity comps, then re-rank realistic backup targets.



    KNN similarity is intentionally based ONLY on the four controller dimensions.

    Defensive and aerial qualities are applied afterward so the model searches for

    "Nico-like, but more defensively/aerially secure" rather than allowing ball-winning

    metrics to redefine what a controller is.

    """

    path = NICO_KNN_SOURCE



    if not path.exists():

        raise FileNotFoundError(f"Nico KNN source does not exist:\n  {path}")

    if not NICO_PERCENTILES_SOURCE.exists():

        raise FileNotFoundError(

            "Europe percentile source does not exist:\n"

            f"  {NICO_PERCENTILES_SOURCE}"

        )



    raw = pd.read_csv(path)



    required = [

        "Player", "Club", "Age", "Jaissle Controller",

        "Transition Protection", "Deep Progression",

        "Possession Security", "Deep Involvement",

        "Coverage %", "Recoveries/90", "Interceptions/90", "Tackles/90",

        "Line-Breaking Passes/90", "Successful Passes/90",

        "Dispossessed/90", "Pass %", "Touches/90", "Minutes", "Value €M",

    ]

    missing = [c for c in required if c not in raw.columns]

    if missing:

        raise ValueError(

            "holding_controller_jaissle_detailed.csv is missing expected columns:\n"

            f"  {missing}"

        )



    df = raw.copy()

    df["Player"] = df["Player"].fillna("").astype(str).map(canonical_player_name)



    numeric_columns = [c for c in required if c not in {"Player", "Club", "Source", "NUFC Benchmark"}]

    for col in numeric_columns:

        df[col] = pd.to_numeric(df[col], errors="coerce")



    df["Project Fee €M"] = [

        fee_override(player_name, source_fee)

        for player_name, source_fee in zip(df["Player"], df["Value €M"])

    ]



    # --------------------------------------------------------------------------
    # MERGE EUROPE-WIDE DEFENSIVE + AERIAL PERCENTILES
    # --------------------------------------------------------------------------
    pct = pd.read_parquet(NICO_PERCENTILES_SOURCE).copy()



    def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:

        lookup = {str(c).strip().lower(): c for c in frame.columns}

        for name in names:

            if name.lower() in lookup:

                return lookup[name.lower()]

        return None



    pct_player = first_existing(

        pct, ["player_name", "Player", "player", "name", "playerName"]

    )

    if pct_player is None:

        raise ValueError(

            "Could not identify player-name column in "

            "europe_2025_26_player_percentiles.parquet."

        )



    metric_aliases = {

        "Def Recoveries Pctl": ["recoveries_per90_percentile"],

        "Def Interceptions Pctl": ["interceptions_per90_percentile"],

        "Def Tackles Won Pctl": ["tackles_won_per90_percentile"],

        "Aerials Won/90 Pctl": ["aerials_won_per90_percentile"],

        "Aerial Win % Pctl": ["aerial_win_pct_percentile"],

        "Duels Won/90 Pctl": ["duels_won_per90_percentile"],

        "Duel Win % Pctl": ["duel_win_pct_percentile"],

        "Aerials Won/90": ["aerials_won_per90"],

        "Aerial Win %": ["aerial_win_pct"],

        "Duels Won/90": ["duels_won_per90"],

        "Duel Win %": ["duel_win_pct"],

    }



    pct["_merge_player"] = (

        pct[pct_player].fillna("").astype(str).map(canonical_player_name).map(normalize)

    )



    keep = ["_merge_player"]

    rename_map = {}

    for output_name, aliases in metric_aliases.items():

        source_col = first_existing(pct, aliases)

        if source_col is not None:

            keep.append(source_col)

            rename_map[source_col] = output_name



    essential = {

        "Def Recoveries Pctl",

        "Def Interceptions Pctl",

        "Def Tackles Won Pctl",

        "Aerials Won/90 Pctl",

        "Aerial Win % Pctl",

    }

    found_outputs = set(rename_map.values())

    missing_metrics = sorted(essential - found_outputs)

    if missing_metrics:

        raise ValueError(

            "Europe percentile file is missing required Nico backup metrics:\n"

            f"  {missing_metrics}"

        )



    pct_small = pct[keep].rename(columns=rename_map).copy()

    pct_numeric = [c for c in pct_small.columns if c != "_merge_player"]

    for col in pct_numeric:

        pct_small[col] = pd.to_numeric(pct_small[col], errors="coerce")



    # If duplicate player-season records exist, prefer the row with the greatest
    # non-null metric coverage. This avoids multiplying controller rows on merge.
    pct_small["_coverage"] = pct_small[pct_numeric].notna().sum(axis=1)

    pct_small = (

        pct_small.sort_values("_coverage", ascending=False)

        .drop_duplicates("_merge_player", keep="first")

        .drop(columns="_coverage")

    )



    df["_merge_player"] = df["Player"].map(normalize)

    df = df.merge(pct_small, on="_merge_player", how="left")



    # --------------------------------------------------------------------------
    # MERGE RECRUITMENT METADATA (height/value/position; fills gaps only)
    # --------------------------------------------------------------------------
    if NICO_METADATA_SOURCE.exists():

        meta = pd.read_csv(NICO_METADATA_SOURCE).copy()

        if "Player" in meta.columns:

            meta["_merge_player"] = (

                meta["Player"].fillna("").astype(str)

                .map(canonical_player_name).map(normalize)

            )

            meta_cols = [

                c for c in [

                    "_merge_player", "Age", "Height", "Natural Position",

                    "Position Code", "Market Value €M", "League"

                ] if c in meta.columns

            ]

            meta = (

                meta[meta_cols]

                .drop_duplicates("_merge_player", keep="first")

                .rename(columns={

                    "Age": "Metadata Age",

                    "Height": "Height",

                    "Natural Position": "Natural Position",

                    "Position Code": "Position Code",

                    "Market Value €M": "Metadata Value €M",

                    "League": "League",

                })

            )

            df = df.merge(meta, on="_merge_player", how="left")



            if "Metadata Value €M" in df.columns:

                meta_fee = pd.to_numeric(df["Metadata Value €M"], errors="coerce")

                df["Value €M"] = df["Value €M"].fillna(meta_fee)

                project_fee = pd.to_numeric(df["Project Fee €M"], errors="coerce")

                df["Project Fee €M"] = project_fee.fillna(meta_fee)



            # External-player ages in the controller file are incomplete.
            # Recruitment metadata is the authoritative source for backup filtering.
            if "Metadata Age" in df.columns:

                metadata_age = pd.to_numeric(df["Metadata Age"], errors="coerce")

                controller_age = pd.to_numeric(df["Age"], errors="coerce")

                df["Age"] = metadata_age.fillna(controller_age)



    # --------------------------------------------------------------------------
    # DEFENSIVE + AERIAL PROFILE SCORES
    # --------------------------------------------------------------------------
    # Percentile files can be either 0-1 or 0-100 depending on pipeline stage.
    # Normalize each metric to 0-100 defensively.
    pctl_cols = [

        "Def Recoveries Pctl", "Def Interceptions Pctl", "Def Tackles Won Pctl",

        "Aerials Won/90 Pctl", "Aerial Win % Pctl",

    ]

    for col in pctl_cols:

        vals = pd.to_numeric(df[col], errors="coerce")

        finite = vals.dropna()

        if not finite.empty and finite.quantile(0.95) <= 1.5:

            vals = vals * 100.0

        df[col] = vals.clip(0, 100)



    df["Defensive Score"] = (

        0.40 * df["Def Recoveries Pctl"]

        + 0.35 * df["Def Interceptions Pctl"]

        + 0.25 * df["Def Tackles Won Pctl"]

    )

    df["Aerial Score"] = (

        0.60 * df["Aerial Win % Pctl"]

        + 0.40 * df["Aerials Won/90 Pctl"]

    )



    # --------------------------------------------------------------------------
    # IDENTIFY THE ONE TRUE CURRENT NICO ROW + BUILD KNN POOL
    # --------------------------------------------------------------------------
    nico_names = {normalize(name) for name in NICO_NAMES}
    player_is_nico = df["Player"].astype(str).map(normalize).isin(nico_names)
    benchmark = df["NUFC Benchmark"].fillna("").astype(str).str.strip().str.upper()
    source = df["Source"].fillna("").astype(str)
    club = df["Club"].fillna("").astype(str)
    current_nico_mask = (
        player_is_nico & benchmark.eq("CURRENT")
        & (source.str.contains("Newcastle", case=False, na=False)
           | club.str.contains("Newcastle", case=False, na=False))
    )
    if current_nico_mask.sum() != 1:
        candidates = df.loc[player_is_nico, ["Player","Club","Source","NUFC Benchmark","Minutes",*NICO_KNN_FEATURES]]
        raise ValueError("Expected exactly one CURRENT Newcastle Nico Gonzalez reference row, "
                         f"found {int(current_nico_mask.sum())}.\nNico rows available:\n{candidates.to_string(index=False)}")
    nico_row = df.loc[current_nico_mask].copy()
    expected_nico = {"Transition Protection":59.986111,"Deep Progression":60.166667,
                     "Possession Security":52.0,"Deep Involvement":59.041667}
    for feature, expected in expected_nico.items():
        actual=float(nico_row.iloc[0][feature])
        if not np.isclose(actual,expected,atol=0.05):
            raise ValueError(f"Wrong Nico reference: {feature}={actual:.4f}; expected ~{expected:.4f}")
    external_mask=(~player_is_nico & df["Minutes"].ge(NICO_KNN_MIN_MINUTES)
                   & df[NICO_KNN_FEATURES].notna().all(axis=1))
    external_df=(df.loc[external_mask].sort_values("Minutes",ascending=False,na_position="last")
                 .drop_duplicates("Player",keep="first").copy())
    model_df=pd.concat([nico_row,external_df],ignore_index=True,sort=False)
    model_df["_is_nico"]=False
    model_df.loc[0,"_is_nico"]=True
    nico_index=0
    is_nico_model=model_df["_is_nico"].copy()
    print("\n"+"="*120)
    print("VERIFIED CURRENT NICO REFERENCE")
    print("="*120)
    print(f"Player: {model_df.loc[0,'Player']}")
    print(f"Club: {model_df.loc[0,'Club']}")
    print(f"Source: {model_df.loc[0,'Source']}")
    print(f"NUFC Benchmark: {model_df.loc[0,'NUFC Benchmark']}")
    for feature in NICO_KNN_FEATURES:
        print(f"{feature:<28}{model_df.loc[0,feature]:>8.2f}")
    print("="*120)




    X_raw = model_df[NICO_KNN_FEATURES].astype(float)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_raw)



    knn = NearestNeighbors(

        n_neighbors=min(len(model_df), NICO_KNN_TOP_N + 1),

        metric="euclidean",

    )

    knn.fit(X_scaled)

    nico_vector = X_scaled[nico_index].reshape(1, -1)

    knn.kneighbors(nico_vector)



    distances = np.linalg.norm(X_scaled - nico_vector, axis=1)

    model_df["Nico KNN Distance"] = distances

    model_df["Nico Similarity Index"] = 100.0 / (1.0 + distances)



    for feature_index, feature in enumerate(NICO_KNN_FEATURES):

        dimension_distance = np.abs(

            X_scaled[:, feature_index] - X_scaled[nico_index, feature_index]

        )

        model_df[f"{feature} Similarity"] = 100.0 / (1.0 + dimension_distance)

        model_df[f"{feature} Δ vs Nico"] = (

            model_df[feature] - model_df.loc[nico_index, feature]

        )



    nico_def = model_df.loc[nico_index, "Defensive Score"]

    nico_aerial = model_df.loc[nico_index, "Aerial Score"]



    model_df["Defensive Δ vs Nico"] = model_df["Defensive Score"] - nico_def

    model_df["Aerial Δ vs Nico"] = model_df["Aerial Score"] - nico_aerial



    # Upgrade components are centered at 50: matching Nico = 50, outperforming = >50.
    # A 1 percentile-point improvement adds 1 point, capped for stability.
    model_df["Defensive Upgrade Index"] = (

        50.0 + model_df["Defensive Δ vs Nico"]

    ).clip(0, 100)

    model_df["Aerial Upgrade Index"] = (

        50.0 + model_df["Aerial Δ vs Nico"]

    ).clip(0, 100)



    print()

    print("=" * 120)

    print("NICO GONZÁLEZ KNN + DEFENSIVE/AERIAL REFERENCE PROFILE")

    print("=" * 120)

    print(f"Controller source: {path}")

    print(f"Defensive/aerial source: {NICO_PERCENTILES_SOURCE}")

    print(f"Recruitment metadata: {NICO_METADATA_SOURCE}")

    print(f"Candidates in KNN pool: {len(model_df) - 1}")

    print(f"External candidate minimum minutes: {NICO_KNN_MIN_MINUTES}")

    print()



    for feature in NICO_KNN_FEATURES:

        print(f"{feature:<28}{model_df.loc[nico_index, feature]:>8.2f}")



    print()

    for feature in [

        "Defensive Score", "Aerial Score",

        "Def Recoveries Pctl", "Def Interceptions Pctl",

        "Def Tackles Won Pctl", "Aerial Win % Pctl", "Aerials Won/90 Pctl",

    ]:

        value = model_df.loc[nico_index, feature]

        label = feature

        print(f"{label:<28}{value:>8.2f}" if pd.notna(value)

              else f"{label:<28}{'N/A':>8}")



    # --------------------------------------------------------------------------
    # UNRESTRICTED COMPS
    # --------------------------------------------------------------------------
    comparisons = model_df[~is_nico_model].copy()

    comparisons = comparisons[~excluded_mask(comparisons["Player"])].copy()

    comparisons = comparisons.sort_values(

        ["Nico KNN Distance", "Nico Similarity Index"],

        ascending=[True, False],

    ).reset_index(drop=True)



    full_comps = comparisons.head(NICO_KNN_TOP_N).copy().reset_index(drop=True)

    full_comps.insert(0, "Rank", np.arange(1, len(full_comps) + 1))



    # --------------------------------------------------------------------------
    # REALISTIC BACKUP FILTER + RE-RANK
    # --------------------------------------------------------------------------
    # Recruitment brief:
    #   - age 26-30
    #   - natural DM/CM
    #   - project fee <= €15m (known fee required)
    #   - >= 900 minutes already enforced by the KNN pool
    #   - Defensive Score >= 65
    #   - Aerial Score >= 60
    #   - known 2026 movers excluded
    #
    # Similarity remains dominant: this is a Nico deputy search, not a generic
    # ball-winner ranking. Defensive/aerial strength supplies the desired tilt.
    age_mask = comparisons["Age"].between(

        NICO_BACKUP_MIN_AGE, NICO_BACKUP_MAX_AGE, inclusive="both"

    )

    fee_num = pd.to_numeric(comparisons["Project Fee €M"], errors="coerce")

    fee_mask = fee_num.notna() & fee_num.le(NICO_BACKUP_MAX_FEE)



    natural_position = comparisons.get(

        "Natural Position", pd.Series("", index=comparisons.index)

    ).fillna("").astype(str).str.lower()

    position_code = comparisons.get(

        "Position Code", pd.Series("", index=comparisons.index)

    ).fillna("").astype(str).str.upper()

    position_mask = (

        natural_position.str.contains("defensive midfield", regex=False)

        | natural_position.str.contains("central midfield", regex=False)

        | position_code.str.contains(r"(?:^|[, /])DM(?:[, /]|$)", regex=True)

        | position_code.str.contains(r"(?:^|[, /])CM(?:[, /]|$)", regex=True)

    )



    defensive_mask = comparisons["Defensive Score"].ge(

        NICO_BACKUP_MIN_DEFENSIVE_SCORE

    )

    aerial_mask = comparisons["Aerial Score"].ge(

        NICO_BACKUP_MIN_AERIAL_SCORE

    )

    recent_movers = {normalize(name) for name in NICO_2026_MOVERS}

    mover_mask = comparisons["Player"].astype(str).map(normalize).isin(recent_movers)



    print()

    print("REALISTIC BACKUP FILTER AUDIT")

    print(f"  Total non-Nico comparisons: {len(comparisons)}")

    print(f"  Age data available: {int(comparisons['Age'].notna().sum())}")

    print(f"  Age {NICO_BACKUP_MIN_AGE}-{NICO_BACKUP_MAX_AGE}: {int(age_mask.sum())}")

    print(f"  Natural DM/CM: {int(position_mask.sum())}")

    print(f"  Fee <= €{NICO_BACKUP_MAX_FEE:.0f}m: {int(fee_mask.sum())}")

    print(f"  Defensive score >= {NICO_BACKUP_MIN_DEFENSIVE_SCORE:.0f}: {int(defensive_mask.sum())}")

    print(f"  Aerial score >= {NICO_BACKUP_MIN_AERIAL_SCORE:.0f}: {int(aerial_mask.sum())}")

    statistical_mask = age_mask & position_mask & fee_mask & defensive_mask & aerial_mask

    print(f"  All statistical filters: {int(statistical_mask.sum())}")

    print(f"  Known 2026 movers in filtered pool: {int((statistical_mask & mover_mask).sum())}")



    backup = comparisons[statistical_mask & ~mover_mask].copy()

    backup_filter_status = (

        "Strict: age 26-30, natural DM/CM, fee <= €15m, "

        "DEF >= 65, AER >= 60, known 2026 movers excluded"

    )

    print(f"  Final eligible pool: {len(backup)}")

    print(f"  Filter status: {backup_filter_status}")



    backup["Backup Filter Status"] = backup_filter_status

    backup["Nico Backup Fit"] = (

        NICO_BACKUP_SIMILARITY_WEIGHT * backup["Nico Similarity Index"]

        + NICO_BACKUP_DEFENSIVE_WEIGHT * backup["Defensive Score"]

        + NICO_BACKUP_AERIAL_WEIGHT * backup["Aerial Score"]

    )



    backup = (

        backup.sort_values(

            [

                "Nico Backup Fit",

                "Nico Similarity Index",

                "Defensive Score",

                "Aerial Score",

                "Project Fee €M",

            ],

            ascending=[False, False, False, False, True],

            na_position="last",

        )

        .head(NICO_BACKUP_TOP_N)

        .reset_index(drop=True)

    )

    backup.insert(0, "Rank", np.arange(1, len(backup) + 1))



    # Plot the actual recommended backup shortlist, not merely unrestricted neighbors.
    reference = pd.concat(

        [

            model_df.loc[[nico_index]],

            backup.head(NICO_VIS_TOP_N),

        ],

        ignore_index=True,

        sort=False,

    )



    return full_comps, backup, reference, path



# ======================================================================================
# NICO VISUALIZATION
# ======================================================================================
def create_nico_knn_visualization(reference: pd.DataFrame) -> Path:

    """

    Radial map of the final realistic Nico-backup shortlist.



    Geometry:

      - center: Nico Gonzalez

      - radius: KNN distance, min-max rescaled among displayed targets for readability

      - circle size: defensive score

      - outline width: aerial score

      - #1 callout: controller dimensions + defensive/aerial deltas vs Nico



    The display scaling changes only the graphic; it never changes model distances,

    similarities, ranks, or final scores.

    """

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    output = VIS_DIR / "nico_gonzalez_knn_similarity.png"



    plot_df = reference.copy().drop_duplicates("Player", keep="first")

    nico_names = {normalize(name) for name in NICO_NAMES}

    nico_mask = plot_df["Player"].astype(str).map(normalize).isin(nico_names)

    if not nico_mask.any():

        raise ValueError("Nico Gonzalez is missing from visualization data.")



    nico = plot_df.loc[nico_mask].iloc[0]

    comps = (

        plot_df.loc[~nico_mask]

        .sort_values(["Nico Backup Fit", "Nico KNN Distance"],

                     ascending=[False, True], na_position="last")

        .head(NICO_VIS_TOP_N)

        .reset_index(drop=True)

    )

    if comps.empty:

        print("WARNING: No backup players available for Nico visualization; skipping plot.")

        return VIS_DIR / "nico_gonzalez_knn_similarity.png"



    distances = pd.to_numeric(comps["Nico KNN Distance"], errors="coerce").to_numpy()

    dmin = float(np.nanmin(distances))

    dmax = float(np.nanmax(distances))



    # Deliberately expand the displayed range so close KNN distances do not collapse
    # onto one ring. This is VISUAL rescaling only.
    min_radius, max_radius = 1.55, 5.00

    if np.isclose(dmin, dmax):

        scaled = np.full(len(comps), 0.5)

    else:

        scaled = (distances - dmin) / (dmax - dmin)

    # Mild power transform adds separation among the closest few candidates.
    scaled = np.power(np.clip(scaled, 0, 1), 0.72)

    radii = min_radius + scaled * (max_radius - min_radius)



    angles = np.linspace(

        np.pi / 2,

        np.pi / 2 - 2 * np.pi,

        len(comps),

        endpoint=False,

    )

    xs, ys = radii * np.cos(angles), radii * np.sin(angles)



    fig, ax = plt.subplots(figsize=(14, 12))



    for r in np.linspace(min_radius, max_radius, 4):

        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=0.8, alpha=0.18))



    # Circle size = defensive quality. Border width = aerial quality.
    def_scores = pd.to_numeric(comps["Defensive Score"], errors="coerce").fillna(50)

    aerial_scores = pd.to_numeric(comps["Aerial Score"], errors="coerce").fillna(50)

    sizes = 650 + 10.0 * def_scores.to_numpy()

    edge_widths = 1.0 + 3.0 * (aerial_scores.to_numpy() / 100.0)



    # Plot individually so aerial score can control each outline width.
    for i, row in comps.iterrows():

        is_first = i == 0

        ax.scatter(

            [xs[i]], [ys[i]],

            s=sizes[i] * (1.22 if is_first else 1.0),

            alpha=0.92,

            edgecolors="black" if is_first else "white",

            linewidths=edge_widths[i] + (1.0 if is_first else 0.0),

            zorder=4 if is_first else 3,

        )



        def_delta = row.get("Defensive Δ vs Nico", np.nan)

        aer_delta = row.get("Aerial Δ vs Nico", np.nan)

        sim = row.get("Nico Similarity Index", np.nan)

        fit = row.get("Nico Backup Fit", np.nan)



        ax.text(

            xs[i], ys[i] + 0.10,

            f"#{i+1}  {row['Player']}",

            ha="center", va="center",

            fontsize=9.2 if is_first else 8.5,

            fontweight="bold",

            zorder=5,

        )

        ax.text(

            xs[i], ys[i] - 0.25,

            f"Fit {fit:.1f} | Sim {sim:.1f}\n"

            f"DEF {def_delta:+.1f} | AER {aer_delta:+.1f}",

            ha="center", va="center",

            fontsize=7.6,

            zorder=5,

        )



    # Nico center.
    ax.scatter(

        [0], [0], s=2800, marker="o",

        edgecolors="black", linewidths=2.0, zorder=6,

    )

    ax.text(

        0, 0.10, "NICO\nGONZALEZ",

        ha="center", va="center",

        fontsize=12, fontweight="bold", zorder=7,

    )

    ax.text(

        0, -0.56, "REFERENCE",

        ha="center", va="center", fontsize=8, zorder=7,

    )



    # Detailed callout for the #1 recommendation.
    top = comps.iloc[0]

    def fmt(v, digits=1):

        return "N/A" if pd.isna(v) else f"{float(v):.{digits}f}"



    callout = (

        f"#1 BACKUP TARGET — {top['Player']}\n"

        f"Target Fit {fmt(top.get('Nico Backup Fit'))}   |   "

        f"Nico Similarity {fmt(top.get('Nico Similarity Index'))}\n\n"

        f"{'ATTRIBUTE':<18}{'TARGET':>8}{'NICO':>8}{'DELTA':>8}\n"

        f"{'Transition':<18}{fmt(top['Transition Protection']):>8}"

        f"{fmt(nico['Transition Protection']):>8}"

        f"{fmt(top['Transition Protection']-nico['Transition Protection']):>8}\n"

        f"{'Progression':<18}{fmt(top['Deep Progression']):>8}"

        f"{fmt(nico['Deep Progression']):>8}"

        f"{fmt(top['Deep Progression']-nico['Deep Progression']):>8}\n"

        f"{'Security':<18}{fmt(top['Possession Security']):>8}"

        f"{fmt(nico['Possession Security']):>8}"

        f"{fmt(top['Possession Security']-nico['Possession Security']):>8}\n"

        f"{'Involvement':<18}{fmt(top['Deep Involvement']):>8}"

        f"{fmt(nico['Deep Involvement']):>8}"

        f"{fmt(top['Deep Involvement']-nico['Deep Involvement']):>8}\n"

        f"{'Defensive score':<18}{fmt(top.get('Defensive Score')):>8}"

        f"{fmt(nico.get('Defensive Score')):>8}"

        f"{fmt(top.get('Defensive Δ vs Nico')):>8}\n"

        f"{'Aerial score':<18}{fmt(top.get('Aerial Score')):>8}"

        f"{fmt(nico.get('Aerial Score')):>8}"

        f"{fmt(top.get('Aerial Δ vs Nico')):>8}\n\n"

        f"Age {fmt(top.get('Age'),0)}   |   "

        f"Fee €{fmt(top.get('Project Fee €M'))}m   |   "

        f"Height {top.get('Height','N/A')}"

    )

    ax.text(

        1.015, 0.80, callout,

        transform=ax.transAxes,

        ha="left", va="top",

        fontsize=8.7,

        family="monospace",

        bbox=dict(boxstyle="round,pad=0.7", alpha=0.10),

    )



    ax.set_title(

        "Nico Gonzalez Backup DM Search — Corrected CURRENT Reference",

        fontsize=16, pad=22, fontweight="bold",

    )

    ax.text(

        0.5, 0.965,

        "Radius is rescaled among displayed targets for readability; model distances and rankings are unchanged",

        transform=ax.transAxes, ha="center", va="top", fontsize=9.5,

    )



    profile_text = (

        f"Nico baseline — Transition {nico['Transition Protection']:.1f} | "

        f"Progression {nico['Deep Progression']:.1f} | "

        f"Security {nico['Possession Security']:.1f} | "

        f"Involvement {nico['Deep Involvement']:.1f} | "

        f"DEF {fmt(nico.get('Defensive Score'))} | "

        f"AER {fmt(nico.get('Aerial Score'))}"

    )

    ax.text(

        0.5, 0.02, profile_text,

        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.7,

    )



    ax.text(

        0.01, 0.02,

        "Radius = controller KNN distance (display-rescaled)\nCircle size = defensive score\nOutline thickness = aerial score",

        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,

    )



    limit = max_radius + 1.2

    ax.set_xlim(-limit, limit)

    ax.set_ylim(-limit, limit)

    ax.set_aspect("equal", adjustable="box")

    ax.axis("off")



    # Extra right margin for #1 callout.
    fig.subplots_adjust(right=0.78)

    fig.savefig(output, dpi=240, bbox_inches="tight")

    plt.close(fig)

    return output



# ======================================================================================
# MAIN
# ======================================================================================
def main() -> None:

    OUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    VIS_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    cb, cb_source = (

        build_cb_targets()

    )



    rw, rw_source = (

        build_rw_targets()

    )



    (

        nico_comps,

        nico_backup,

        nico_reference,

        nico_source,

    ) = build_nico_knn_targets()



    visualization_out = (

        create_nico_knn_visualization(

            nico_reference

        )

    )



    cb_out = (

        OUT_DIR

        / "top_25_high_line_succession_cb.csv"

    )



    rw_out = (

        OUT_DIR

        / "top_10_elite_complete_rw.csv"

    )



    nico_comps_out = (

        OUT_DIR

        / "top_30_nico_gonzalez_knn_comps.csv"

    )



    nico_backup_out = (

        OUT_DIR

        / "top_15_nico_gonzalez_backup_targets.csv"

    )



    cb.to_csv(

        cb_out,

        index=False,

    )



    rw.to_csv(

        rw_out,

        index=False,

    )



    nico_comps.to_csv(

        nico_comps_out,

        index=False,

    )



    nico_backup.to_csv(

        nico_backup_out,

        index=False,

    )



    print()

    print("=" * 170)

    print("FINAL NEWCASTLE RECRUITMENT FOCUS")

    print("=" * 170)



    print(

        f"CB source: {cb_source.name}"

    )



    print(

        f"RW source: {rw_source.name}"

    )



    print(

        f"Nico KNN source: {nico_source.name}"

    )



    print_table(

        "TOP 25 HIGH-LINE SUCCESSION CB TARGETS — AGE 16-23",

        cb,

        [

            "Rank",

            "Player",

            "Club",

            "League",

            "Age",

            "Preferred Foot",

            "Project Fee €M",

            "CB Final Score",

            "CB Football Score",

            "Existing CB Score",

            "Build-Up",

            "Front-Foot",

            "Physical",

            "Line-Breaking/90",

            "Successful Passes/90",

            "Pass %",

            "Duel Win %",

            "Aerial Win %",

            "Interceptions/90",

            "Recoveries/90",

            "Tackles Won/90",

        ],

    )



    print_table(

        "TOP 10 UPPER-ECHELON / COMPLETE RW TARGETS — AGE 22-28",

        rw,

        [

            "Rank",

            "Player",

            "Club",

            "League",

            "Age",

            "Project Fee €M",

            "Elite RW Score",

            "Elite RW Football Score",

            "Complete Profile Floor",

            "Primary Strength",

            "Winger Score",

            "Creation",

            "1v1 / Progression",

            "Goal Threat",

            "xG/90",

            "xA/90",

            "xG+xA/90",

            "Successful Dribbles/90",

            "Chances Created/90",

            "Recoveries/90",

            "Final 3rd Wins/90",

        ],

    )



    print_table(

        "TOP 30 NICO GONZÁLEZ HOLDING-CONTROLLER COMPS — KNN",

        nico_comps,

        [

            "Rank",

            "Player",

            "Club",

            "Age",

            "Minutes",

            "Project Fee €M",

            "Nico Similarity Index",

            "Nico KNN Distance",

            "Jaissle Controller",

            "Transition Protection",

            "Deep Progression",

            "Possession Security",

            "Deep Involvement",

            "Transition Protection Similarity",

            "Deep Progression Similarity",

            "Possession Security Similarity",

            "Deep Involvement Similarity",

            "Transition Protection Δ vs Nico",

            "Deep Progression Δ vs Nico",

            "Possession Security Δ vs Nico",

            "Deep Involvement Δ vs Nico",

            "Coverage %",

            "Recoveries/90",

            "Interceptions/90",

            "Tackles/90",

            "Line-Breaking Passes/90",

            "Successful Passes/90",

            "Dispossessed/90",

            "Pass %",

            "Touches/90",

        ],

    )



    print_table(

        "TOP 15 REALISTIC NICO BACKUPS — AGE 26-30 / MAX €15M",

        nico_backup,

        [

            "Rank",

            "Player",

            "Club",

            "Age",

            "Minutes",

            "Project Fee €M",

            "Backup Value Tier",

            "Nico Backup Fit",

            "Nico Similarity Index",

            "Nico KNN Distance",

            "Jaissle Controller",

            "Transition Protection",

            "Deep Progression",

            "Possession Security",

            "Deep Involvement",

            "Transition Protection Similarity",

            "Deep Progression Similarity",

            "Possession Security Similarity",

            "Deep Involvement Similarity",

            "Coverage %",

            "Recoveries/90",

            "Interceptions/90",

            "Tackles/90",

            "Line-Breaking Passes/90",

            "Successful Passes/90",

            "Dispossessed/90",

            "Pass %",

            "Touches/90",

        ],

    )



    print()

    print("=" * 170)

    print("FILES SAVED")

    print("=" * 170)



    print(

        f"CB -> {cb_out}"

    )



    print(

        f"RW -> {rw_out}"

    )



    print(

        f"Nico KNN comps -> {nico_comps_out}"

    )



    print(

        f"Nico backup targets -> {nico_backup_out}"

    )



    print(

        f"Nico visualization -> {visualization_out}"

    )





if __name__ == "__main__":

    main()
