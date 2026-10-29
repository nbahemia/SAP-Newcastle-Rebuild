from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

PLAYER_FILE = RAW_DIR / "players.parquet"
LINEUPS_FILE = RAW_DIR / "lineups.parquet"

SEASON_FILE = (
    PROCESSED_DIR
    / "player_season_metrics.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "player_season_metrics_enriched.parquet"
)

POSITION_HISTORY_FILE = (
    PROCESSED_DIR
    / "player_position_history.parquet"
)


# ---------------------------------------------------------
# Position mappings
# ---------------------------------------------------------

DIRECT_POSITION_MAP = {
    # Goalkeeper
    "G": "GK",
    "GK": "GK",
    "GOALKEEPER": "GK",

    # Centre backs
    "CB": "CB",
    "LCB": "CB",
    "RCB": "CB",
    "CENTRE BACK": "CB",
    "CENTER BACK": "CB",
    "CENTRE-BACK": "CB",
    "CENTER-BACK": "CB",

    # Fullbacks / wingbacks
    "LB": "FB/WB",
    "RB": "FB/WB",
    "LWB": "FB/WB",
    "RWB": "FB/WB",
    "LEFT BACK": "FB/WB",
    "RIGHT BACK": "FB/WB",
    "LEFT-BACK": "FB/WB",
    "RIGHT-BACK": "FB/WB",
    "LEFT WING BACK": "FB/WB",
    "RIGHT WING BACK": "FB/WB",
    "LEFT WING-BACK": "FB/WB",
    "RIGHT WING-BACK": "FB/WB",

    # Defensive / central midfield
    "DM": "DM/CM",
    "CDM": "DM/CM",
    "CM": "DM/CM",
    "LCM": "DM/CM",
    "RCM": "DM/CM",
    "DMC": "DM/CM",
    "MC": "DM/CM",
    "MIDFIELD": "DM/CM",
    "CENTRAL MIDFIELD": "DM/CM",
    "CENTRE MIDFIELD": "DM/CM",
    "DEFENSIVE MIDFIELD": "DM/CM",

    # Attacking midfield
    "AM": "AM",
    "CAM": "AM",
    "LAM": "AM",
    "RAM": "AM",
    "AMC": "AM",
    "ATTACKING MIDFIELD": "AM",
    "ATTACKING MIDFIELDER": "AM",

    # Wingers
    "LW": "W",
    "RW": "W",
    "LM": "W",
    "RM": "W",
    "LWF": "W",
    "RWF": "W",
    "LEFT WING": "W",
    "RIGHT WING": "W",
    "LEFT MIDFIELD": "W",
    "RIGHT MIDFIELD": "W",
    "WINGER": "W",

    # Strikers
    "ST": "ST",
    "CF": "ST",
    "FW": "ST",
    "STRIKER": "ST",
    "CENTRE FORWARD": "ST",
    "CENTER FORWARD": "ST",
    "CENTRE-FORWARD": "ST",
    "CENTER-FORWARD": "ST",
    "FORWARD": "ST",
}


PROFILE_POSITION_MAP = {
    "G": "GK",
    "GK": "GK",
    "GOALKEEPER": "GK",

    "D": "DEF",
    "DEF": "DEF",
    "DEFENDER": "DEF",

    "M": "MID",
    "MID": "MID",
    "MIDFIELDER": "MID",

    "F": "FWD",
    "FW": "FWD",
    "FWD": "FWD",
    "FORWARD": "FWD",
    "ATTACKER": "FWD",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_position_text(value):
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if not text:
        return None

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def classify_observed_position(value):
    text = clean_position_text(
        value
    )

    if text is None:
        return None

    if text in DIRECT_POSITION_MAP:
        return DIRECT_POSITION_MAP[text]

    # Goalkeeper
    if (
        "GOALKEEP" in text
        or text == "GK"
    ):
        return "GK"

    # Wingbacks
    if (
        "WING BACK" in text
        or "WING-BACK" in text
        or text.endswith("WB")
    ):
        return "FB/WB"

    # Fullbacks
    if (
        "LEFT BACK" in text
        or "RIGHT BACK" in text
        or text in {"LB", "RB"}
    ):
        return "FB/WB"

    # Centre backs
    if (
        "CENTRE BACK" in text
        or "CENTER BACK" in text
        or "CENTRE-BACK" in text
        or "CENTER-BACK" in text
        or text.endswith("CB")
    ):
        return "CB"

    # Attacking midfield
    if (
        "ATTACKING MID" in text
        or text in {"AM", "CAM", "AMC"}
    ):
        return "AM"

    # DM / CM
    if (
        "DEFENSIVE MID" in text
        or "CENTRAL MID" in text
        or "CENTRE MID" in text
        or text in {
            "DM",
            "CDM",
            "CM",
            "DMC",
            "MC",
            "LCM",
            "RCM",
        }
    ):
        return "DM/CM"

    # Wingers
    if (
        "LEFT WING" in text
        or "RIGHT WING" in text
        or text in {
            "LW",
            "RW",
            "LM",
            "RM",
            "LWF",
            "RWF",
        }
    ):
        return "W"

    # Strikers
    if (
        "STRIKER" in text
        or "CENTRE FORWARD" in text
        or "CENTER FORWARD" in text
        or text in {
            "ST",
            "CF",
        }
    ):
        return "ST"

    return "OTHER"


def classify_profile_position(value):
    text = clean_position_text(
        value
    )

    if text is None:
        return None

    return PROFILE_POSITION_MAP.get(
        text,
        "OTHER",
    )


# ---------------------------------------------------------
# Build observed position history
# ---------------------------------------------------------

def summarize_position_history(
    lineups: pd.DataFrame,
):

    history = lineups.copy()

    history[
        "observed_position_clean"
    ] = history[
        "observed_position"
    ].apply(
        clean_position_text
    )

    history[
        "observed_position_group"
    ] = history[
        "observed_position"
    ].apply(
        classify_observed_position
    )

    history = history[
        history[
            "observed_position_group"
        ].notna()
    ].copy()

    known = history[
        history[
            "observed_position_group"
        ]
        != "OTHER"
    ].copy()

    rows = []

    for player_id, group in known.groupby(
        "player_id",
        sort=False,
    ):

        counts = (
            group[
                "observed_position_group"
            ]
            .value_counts()
        )

        if counts.empty:
            continue

        max_count = counts.max()

        tied_roles = (
            counts[
                counts == max_count
            ]
            .index
            .tolist()
        )

        if len(tied_roles) == 1:
            primary_group = tied_roles[0]

        else:
            tied_group = group[
                group[
                    "observed_position_group"
                ].isin(
                    tied_roles
                )
            ]

            primary_group = (
                tied_group
                .sort_values(
                    "event_id"
                )
                .iloc[-1][
                    "observed_position_group"
                ]
            )

        raw_counts = (
            group[
                "observed_position_clean"
            ]
            .value_counts()
        )

        primary_raw_position = (
            raw_counts.index[0]
            if not raw_counts.empty
            else None
        )

        rows.append(
            {
                "player_id": int(player_id),
                "primary_observed_position": (
                    primary_raw_position
                ),
                "primary_position_group": (
                    primary_group
                ),
                "position_matches_observed": (
                    len(group)
                ),
                "position_group_share": (
                    max_count
                    / len(group)
                ),
                "observed_position_groups": ",".join(
                    f"{role}:{count}"
                    for role, count
                    in counts.items()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return history, summary


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    for path in [
        PLAYER_FILE,
        LINEUPS_FILE,
        SEASON_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n"
                f"{path}"
            )

    players = pd.read_parquet(
        PLAYER_FILE
    )

    lineups = pd.read_parquet(
        LINEUPS_FILE
    )

    season = pd.read_parquet(
        SEASON_FILE
    )

    print()
    print("=" * 88)
    print(
        "BUILDING BSD OBSERVED PLAYER POSITIONS"
        .center(88)
    )
    print("=" * 88)

    print(
        f"Players: "
        f"{len(players):,}"
    )

    print(
        f"Lineup rows: "
        f"{len(lineups):,}"
    )

    print(
        f"Season rows: "
        f"{len(season):,}"
    )

    history, position_summary = (
        summarize_position_history(
            lineups
        )
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_parquet(
        POSITION_HISTORY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Metadata merge
    # -----------------------------------------------------

    metadata_columns = [
        "player_id",
        "player_name",
        "position",
        "date_of_birth",
        "height_cm",
        "preferred_foot",
        "nationality",
        "nationality_code",
        "market_value_eur",
        "team_id",
        "team_name",
        "contract_until",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in players.columns
    ]

    enriched = season.merge(
        players[
            metadata_columns
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_profile",
        ),
    )

    # -----------------------------------------------------
    # Position merge
    # -----------------------------------------------------

    enriched = enriched.merge(
        position_summary,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    enriched[
        "profile_position_group"
    ] = enriched[
        "position"
    ].apply(
        classify_profile_position
    )

    enriched[
        "position_group"
    ] = enriched[
        "primary_position_group"
    ].fillna(
        enriched[
            "profile_position_group"
        ]
    )

    enriched[
        "position_source"
    ] = np.where(
        enriched[
            "primary_position_group"
        ].notna(),
        "observed_lineup",
        "profile_fallback",
    )

    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    if "date_of_birth" in enriched.columns:

        dob = pd.to_datetime(
            enriched[
                "date_of_birth"
            ],
            errors="coerce",
        )

        reference_date = pd.Timestamp(
            "2026-07-01"
        )

        enriched[
            "age_at_season_start"
        ] = (
            (
                reference_date
                - dob
            )
            .dt.days
            / 365.2425
        ).round(
            1
        )

    # -----------------------------------------------------
    # Column ordering
    # -----------------------------------------------------

    preferred_columns = [
        "player_id",
        "player_name",
        "position",
        "profile_position_group",
        "primary_observed_position",
        "primary_position_group",
        "position_group",
        "position_source",
        "position_matches_observed",
        "position_group_share",
        "observed_position_groups",
        "current_team_id",
        "team_name",
        "age_at_season_start",
        "date_of_birth",
        "height_cm",
        "preferred_foot",
        "nationality",
        "nationality_code",
        "market_value_eur",
        "contract_until",
        "minutes",
        "appearances",
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column in enriched.columns
    ]

    remaining_columns = [
        column
        for column in enriched.columns
        if column not in preferred_columns
    ]

    enriched = enriched[
        preferred_columns
        + remaining_columns
    ]

    enriched.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 88)
    print(
        "BSD OBSERVED POSITION SUMMARY"
        .center(88)
    )
    print("=" * 88)

    print(
        f"Season players: "
        f"{len(enriched):,}"
    )

    print(
        f"Players with observed role: "
        f"{enriched['primary_position_group'].notna().sum():,}"
    )

    print(
        f"Profile fallbacks: "
        f"{enriched['primary_position_group'].isna().sum():,}"
    )

    print()
    print(
        "Final position groups:"
    )

    print(
        enriched[
            "position_group"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Position source:"
    )

    print(
        enriched[
            "position_source"
        ]
        .value_counts()
    )

    print()
    print(
        "Lowest observed-role certainty "
        "(minimum 2 observed matches):"
    )

    certainty_columns = [
        "player_name",
        "primary_observed_position",
        "primary_position_group",
        "position_matches_observed",
        "position_group_share",
    ]

    uncertain = enriched[
        enriched[
            "position_matches_observed"
        ]
        .fillna(0)
        >= 2
    ]

    if not uncertain.empty:

        print(
            uncertain
            .sort_values(
                "position_group_share"
            )
            [
                certainty_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {POSITION_HISTORY_FILE}"
    )


if __name__ == "__main__":
    main()