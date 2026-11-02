#!/usr/bin/env python3

"""

find_final_cb_rw_targets.py



Final focused Newcastle recruitment search.



Outputs:

- Top 20 High-Line Succession CB targets (ages 18-23)

- Top 10 Elite / Complete RW targets (ages 22-28)



The RW search intentionally does NOT force a single winger archetype.

Newcastle already has verticality through Elanga, so the goal is to identify

upper-echelon wingers who combine:

    - creation

    - 1v1 / progression

    - goal threat

    - overall winger quality



The script is designed to use the existing SAP-Newcastle-Rebuild processed CSVs.



Run from anywhere:

    python scripts/bsd/analyze/find_final_cb_rw_targets.py

"""



from __future__ import annotations



from pathlib import Path

from typing import Iterable, Optional

import re

import unicodedata



import numpy as np

import pandas as pd





# ======================================================================================

# PATHS

# ======================================================================================



ROOT = Path(__file__).resolve().parents[3]

SHORTLIST_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"

OUT_DIR = SHORTLIST_DIR / "final_recruitment_focus"



CB_CANDIDATES = [

    SHORTLIST_DIR / "young_cb_recruitment.csv",

    SHORTLIST_DIR / "center_back_jaissle_detailed.csv",

]



WINGER_CANDIDATES = [

    SHORTLIST_DIR / "bona_fide_winger_recruitment.csv",

    SHORTLIST_DIR / "winger_jaissle_archetypes.csv",

    SHORTLIST_DIR / "winger_jaissle_extended.csv",

]





# ======================================================================================

# SEARCH RULES

# ======================================================================================



CB_MIN_AGE = 18

CB_MAX_AGE = 23



RW_MIN_AGE = 22

RW_MAX_AGE = 28



CB_TOP_N = 20
RW_TOP_N = 10





# Players we currently know are unavailable / should not be returned.

EXCLUDED_PLAYERS = {
    "Chibuike Nwaiwu",      # recently transferred
    "Ahmetcan Kaplan",      # recently transferred
    "Julio Enciso",         # recently joined Ipswich

    # Exclude players who completed a transfer / loan move during calendar year 2026.
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



# Project-specific fee knowledge.

# Used only where we have a stronger project quote than the source CSV.

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

}





# ======================================================================================

# HELPERS

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

    lookup = {

        normalize(col): col

        for col in df.columns

    }



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

        + "\n".join(f"  {p}" for p in paths)

    )





def numeric(

    df: pd.DataFrame,

    candidates: list[str],

    default: float = np.nan,

) -> pd.Series:

    col = choose_column(df, candidates)



    if col is None:

        return pd.Series(

            default,

            index=df.index,

            dtype=float,

        )



    return pd.to_numeric(

        df[col],

        errors="coerce",

    )





def text_series(

    df: pd.DataFrame,

    candidates: list[str],

) -> pd.Series:

    col = choose_column(df, candidates)



    if col is None:

        return pd.Series(

            "",

            index=df.index,

            dtype=object,

        )



    return df[col].astype(str)





def percentile(series: pd.Series) -> pd.Series:

    values = pd.to_numeric(

        series,

        errors="coerce",

    )



    if values.notna().sum() <= 1:

        return pd.Series(

            50.0,

            index=series.index,

        )



    return values.rank(

        pct=True,

        method="average",

    ) * 100.0





def canonical_player_name(name: object) -> str:

    n = normalize(name)



    for canonical, aliases in ALIASES.items():

        choices = {

            normalize(canonical),

            *[

                normalize(alias)

                for alias in aliases

            ],

        }



        if n in choices:

            return canonical



    return str(name).strip()





def excluded_mask(players: pd.Series) -> pd.Series:

    excluded = {

        normalize(player)

        for player in EXCLUDED_PLAYERS

    }



    # include aliases too

    for canonical in EXCLUDED_PLAYERS:

        for alias in ALIASES.get(canonical, []):

            excluded.add(

                normalize(alias)

            )



    return (

        players.astype(str)

        .map(normalize)

        .isin(excluded)

    )





def fee_override(player: str, source_value: float) -> float:

    canonical = canonical_player_name(

        player

    )



    if canonical in PROJECT_FEE_OVERRIDES:

        return float(

            PROJECT_FEE_OVERRIDES[canonical]

        )



    return source_value





def value_score(

    fee: pd.Series,

    quality: pd.Series,

) -> pd.Series:

    """

    Light value bonus.



    This should never dominate the football score. Cheap players receive a

    modest bump, but high-end players can still rank first if their quality is

    clearly superior.

    """

    fee_num = pd.to_numeric(

        fee,

        errors="coerce",

    )



    value_rank = (

        100.0

        - percentile(

            fee_num

        )

    )



    value_rank = value_rank.where(

        fee_num.notna(),

        50.0,

    )



    return (

        0.75 * quality

        + 0.25 * value_rank

    )





# ======================================================================================

# CENTER BACK SEARCH

# ======================================================================================



def build_cb_targets() -> tuple[pd.DataFrame, Path]:

    path = choose_existing(

        CB_CANDIDATES

    )



    raw = pd.read_csv(

        path

    )



    player = text_series(

        raw,

        ["Player", "player_name", "Name"],

    )

    age = numeric(

        raw,

        ["Age", "age"],

    )

    club = text_series(

        raw,

        ["Club", "Current Club", "Squad", "Team"],

    )

    league = text_series(

        raw,

        ["League", "league"],

    )

    fee = numeric(

        raw,

        [

            "Value €M",

            "Current Value €M",

            "market_value_eur_m",

            "Value",

        ],

    )



    # Existing role model dimensions.

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

        [

            "Build-Up",

            "Build Up",

            "Build-Up Score",

            "Progression",

        ],

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

        [

            "Physical",

            "Physical Score",

            "Athleticism",

        ],

    )



    # Raw supporting metrics where present.

    line_breaking = numeric(

        raw,

        [

            "Line-Breaking Passes/90",

            "line_breaking_passes_per90",

        ],

    )

    pass_pct = numeric(

        raw,

        [

            "Pass %",

            "Pass Accuracy %",

            "pass_accuracy_pct",

        ],

    )

    duel_pct = numeric(

        raw,

        [

            "Duel Win %",

            "Duels Won %",

            "duel_win_pct",

        ],

    )

    aerial_pct = numeric(

        raw,

        [

            "Aerial Win %",

            "Aerial Duels Won %",

            "aerial_win_pct",

        ],

    )

    interceptions = numeric(

        raw,

        [

            "Interceptions/90",

            "interceptions_per90",

        ],

    )

    recoveries = numeric(

        raw,

        [

            "Recoveries/90",

            "recoveries_per90",

        ],

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
        [
            "Duels Won/90",
            "duels_won_per90",
        ],
    )

    aerials_won = numeric(
        raw,
        [
            "Aerials Won/90",
            "Aerial Duels Won/90",
            "aerials_won_per90",
        ],
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



    df["Player"] = (

        df["Player"]

        .map(canonical_player_name)

    )



    # Strict age focus requested by Naeem.

    df = df[

        df["Age"]

        .between(

            CB_MIN_AGE,

            CB_MAX_AGE,

            inclusive="both",

        )

    ].copy()



    df = df[

        ~excluded_mask(

            df["Player"]

        )

    ].copy()



    # Apply project quotes.

    df["Project Fee €M"] = [

        fee_override(

            player_name,

            source_fee,

        )

        for player_name, source_fee in zip(

            df["Player"],

            df["Value €M"],

        )

    ]



    # --------------------------------------------------------------

    # Focused CB score

    #

    # Prefer the existing model dimensions when available. If a dimension

    # is missing, raw metrics contribute through percentile ranks.

    # --------------------------------------------------------------



    build_rank = percentile(

        df["Build-Up"]

    )



    # Add raw progression signal if the model build-up column is incomplete.

    raw_progression = (

        0.55 * percentile(df["Line-Breaking/90"])

        + 0.45 * percentile(df["Pass %"])

    )



    build_rank = (

        build_rank.where(

            df["Build-Up"].notna(),

            raw_progression,

        )

    )



    defensive_rank = percentile(

        df["Front-Foot"]

    )



    raw_defense = (

        0.55 * percentile(df["Interceptions/90"])

        + 0.45 * percentile(df["Recoveries/90"])

    )



    defensive_rank = (

        defensive_rank.where(

            df["Front-Foot"].notna(),

            raw_defense,

        )

    )



    ground_rank = percentile(

        df["Duel Win %"]

    )



    physical_rank = percentile(

        df["Physical"]

    )



    aerial_rank = percentile(

        df["Aerial Win %"]

    )



    # If raw ground/aerial data are absent, Physical can serve as fallback.

    ground_rank = ground_rank.where(

        df["Duel Win %"].notna(),

        physical_rank,

    )

    aerial_rank = aerial_rank.where(

        df["Aerial Win %"].notna(),

        physical_rank,

    )



    age_rank = (

        100.0

        - percentile(

            df["Age"]

        )

    )



    fee_rank = (

        100.0

        - percentile(

            df["Project Fee €M"]

        )

    ).where(

        df["Project Fee €M"].notna(),

        50.0,

    )



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

            [

                "CB Final Score",

                "CB Football Score",

            ],

            ascending=False,

        )

        .drop_duplicates(

            "Player"

        )

        .head(CB_TOP_N)

        .reset_index(drop=True)

    )



    df.insert(

        0,

        "Rank",

        np.arange(

            1,

            len(df) + 1,

        ),

    )



    return df, path





# ======================================================================================

# WINGER SEARCH

# ======================================================================================



def build_rw_targets() -> tuple[pd.DataFrame, Path]:

    path = choose_existing(

        WINGER_CANDIDATES

    )



    raw = pd.read_csv(

        path

    )



    player = text_series(

        raw,

        ["Player", "player_name", "Name"],

    )

    age = numeric(

        raw,

        ["Age", "age"],

    )

    club = text_series(

        raw,

        ["Club", "Current Club", "Squad", "Team"],

    )

    league = text_series(

        raw,

        ["League", "league"],

    )

    fee = numeric(

        raw,

        [

            "Value €M",

            "Current Value €M",

            "market_value_eur_m",

            "Value",

        ],

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

        [

            "Creation",

            "Creator",

            "wide_creator_score",

        ],

    )



    one_v_one = numeric(

        raw,

        [

            "1v1 / Progression",

            "1v1/Progression",

            "Progression",

        ],

    )



    goal_threat = numeric(

        raw,

        [

            "Goal Threat",

            "Vertical/Box",

            "vertical_box_score",

        ],

    )



    # Raw performance signals.

    xg = numeric(

        raw,

        [

            "xG/90",

            "xg_per90",

        ],

    )

    xa = numeric(

        raw,

        [

            "xA/90",

            "xa_per90",

        ],

    )

    shots = numeric(

        raw,

        [

            "Shots/90",

            "shots_per90",

        ],

    )

    dribbles = numeric(

        raw,

        [

            "Successful Dribbles/90",

            "successful_dribbles_per90",

        ],

    )

    box_touches = numeric(

        raw,

        [

            "Touches In Box/90",

            "Box Touches/90",

            "box_touches_per90",

        ],

    )

    key_passes = numeric(

        raw,

        [

            "Chances Created/90",

            "Key Passes/90",

            "key_passes_per90",

        ],

    )

    recoveries = numeric(

        raw,

        [

            "Recoveries/90",

            "recoveries_per90",

        ],

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



    df["Player"] = (

        df["Player"]

        .map(canonical_player_name)

    )



    # Strict winger age band requested by Naeem.

    df = df[

        df["Age"]

        .between(

            RW_MIN_AGE,

            RW_MAX_AGE,

            inclusive="both",

        )

    ].copy()



    df = df[

        ~excluded_mask(

            df["Player"]

        )

    ].copy()



    df["Project Fee €M"] = [

        fee_override(

            player_name,

            source_fee,

        )

        for player_name, source_fee in zip(

            df["Player"],

            df["Value €M"],

        )

    ]



    # --------------------------------------------------------------

    # Elite / complete winger score

    #

    # We do not decide "creator" vs "vertical" in advance.

    # Instead, we reward players who are strong across ALL of:

    #   1. Creation

    #   2. 1v1 / progression

    #   3. Goal threat

    #

    # This is the correct complement to Elanga: a winger who can make the

    # attack more technically complete rather than merely duplicate pace.

    # --------------------------------------------------------------



    creation_rank = percentile(

        df["Creation"]

    )

    one_v_one_rank = percentile(

        df["1v1 / Progression"]

    )

    threat_rank = percentile(

        df["Goal Threat"]

    )

    overall_rank = percentile(

        df["Winger Score"]

    )



    # Raw fallback / support.

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



    # Core upper-echelon score.

    df["Elite RW Football Score"] = (

        0.34 * creation_rank

        + 0.31 * one_v_one_rank

        + 0.25 * threat_rank

        + 0.10 * overall_rank

    )



    # Balance bonus:

    # We want complete players, not someone who is elite in one category and

    # mediocre in the others.

    component_floor = pd.concat(

        [

            creation_rank,

            one_v_one_rank,

            threat_rank,

        ],

        axis=1,

    ).min(

        axis=1

    )



    df["Complete Profile Floor"] = component_floor



    # Upper-echelon boost for players with no major weakness.

    df["Elite RW Score"] = (

        0.85 * df["Elite RW Football Score"]

        + 0.15 * component_floor

    )



    # Helpful profile label, but ranking does NOT depend on forcing one archetype.

    strongest = pd.concat(

        {

            "Creator": creation_rank,

            "1v1": one_v_one_rank,

            "Goal Threat": threat_rank,

        },

        axis=1,

    )



    df["Primary Strength"] = strongest.idxmax(

        axis=1

    )



    # High-level filters.

    #

    # Don't make these too strict because source coverage is imperfect.

    # Use them when the raw metric exists; otherwise do not eliminate.

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



    qualified = df[

        raw_quality_mask

    ].copy()



    # If the data coverage makes the filters too restrictive, fall back to

    # the full age-qualified market rather than producing fewer than 10 names.

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

        .drop_duplicates(

            "Player"

        )

        .head(RW_TOP_N)

        .reset_index(drop=True)

    )



    df.insert(

        0,

        "Rank",

        np.arange(

            1,

            len(df) + 1,

        ),

    )



    return df, path





# ======================================================================================

# OUTPUT

# ======================================================================================



def useful_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """Keep only requested columns that contain at least one useful value."""
    kept = []

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce")
            if vals.notna().any():
                kept.append(col)
        else:
            vals = series.fillna("").astype(str).str.strip()
            if vals.ne("").any():
                kept.append(col)

    return kept


def print_table(

    title: str,

    df: pd.DataFrame,

    columns: list[str],

) -> None:

    print()

    print("=" * 150)

    print(title)

    print("=" * 150)



    shown = useful_columns(df, columns)



    print(

        df[shown]

        .to_string(

            index=False,

            float_format=lambda x: f"{x:.2f}",

        )

    )





def main() -> None:

    OUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )



    cb, cb_source = build_cb_targets()

    rw, rw_source = build_rw_targets()



    cb_out = (

        OUT_DIR

        / "top_20_high_line_succession_cb.csv"

    )

    rw_out = (

        OUT_DIR

        / "top_10_elite_complete_rw.csv"

    )



    cb.to_csv(

        cb_out,

        index=False,

    )

    rw.to_csv(

        rw_out,

        index=False,

    )



    print()

    print("FINAL NEWCASTLE RECRUITMENT FOCUS")

    print(f"CB source: {cb_source.name}")

    print(f"RW source: {rw_source.name}")



    print_table(

        "TOP 20 HIGH-LINE SUCCESSION CB TARGETS — AGE 18-23",

        cb,

        [

            "Rank",

            "Player",

            "Club",

            "Age",

            "Project Fee €M",

            "CB Final Score",

            "CB Football Score",

            "Existing CB Score",

            "Build-Up",

            "Front-Foot",

            "Physical",

            "Line-Breaking/90",

            "Duel Win %",

            "Aerial Win %",

        ],

    )



    print_table(

        "TOP 10 UPPER-ECHELON / COMPLETE RW TARGETS — AGE 22-28",

        rw,

        [

            "Rank",

            "Player",

            "Club",

            "Age",

            "Project Fee €M",

            "Elite RW Score",

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

        ],

    )



    print()

    print("=" * 150)

    print("FILES SAVED")

    print("=" * 150)

    print(f"CB -> {cb_out}")

    print(f"RW -> {rw_out}")





if __name__ == "__main__":

    main()