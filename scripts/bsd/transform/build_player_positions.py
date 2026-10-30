from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

PLAYER_FILE = RAW_DIR / "players.parquet"
LINEUPS_FILE = RAW_DIR / "lineups.parquet"
AVERAGE_POSITIONS_FILE = RAW_DIR / "average_positions.parquet"

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
# Broad position mappings
# ---------------------------------------------------------

BROAD_POSITION_MAP = {
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
# Tunable role thresholds
# ---------------------------------------------------------

DEFENDER_WIDE_DISTANCE = 24.0
FULLBACK_MIN_X = 38

MIDFIELDER_WIDE_DISTANCE = 20.0
ATTACKING_MID_X = 60.0

FORWARD_WIDE_DISTANCE = 17.0

SIDE_LEFT_MAX = 38.0
SIDE_RIGHT_MIN = 62.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_broad_position(value):
    if pd.isna(value):
        return None

    key = str(value).strip().upper()

    return BROAD_POSITION_MAP.get(key)


def infer_attack_direction(
    group: pd.DataFrame,
) -> str:
    """
    Infer team attacking direction from average-position data.

    If forwards have a higher mean x than defenders,
    increasing x is treated as attacking.
    Otherwise, decreasing x is attacking.
    """

    defenders = group[
        group["broad_position"]
        == "DEF"
    ]

    forwards = group[
        group["broad_position"]
        == "FWD"
    ]

    defenders = defenders[
        defenders["avg_x"].notna()
    ]

    forwards = forwards[
        forwards["avg_x"].notna()
    ]

    if (
        not defenders.empty
        and not forwards.empty
    ):
        defender_x = defenders[
            "avg_x"
        ].mean()

        forward_x = forwards[
            "avg_x"
        ].mean()

        if forward_x > defender_x:
            return "right"

        return "left"

    return "right"


def classify_tactical_role(
    broad_position,
    x_attacking,
    y_attacking,
):
    if pd.isna(broad_position):
        return None

    if (
        pd.isna(x_attacking)
        or pd.isna(y_attacking)
    ):
        return None

    x_attacking = float(
        x_attacking
    )

    y_attacking = float(
        y_attacking
    )

    wide_distance = abs(
        y_attacking - 50.0
    )

    if broad_position == "GK":
        return "GK"

    if broad_position == "DEF":

        if (
            wide_distance >= DEFENDER_WIDE_DISTANCE
            and x_attacking >= FULLBACK_MIN_X
        ):
            return "FB/WB"

        return "CB"

    if broad_position == "MID":

        if (
            wide_distance
            >= MIDFIELDER_WIDE_DISTANCE
        ):
            return "W"

        if (
            x_attacking
            >= ATTACKING_MID_X
        ):
            return "AM"

        return "DM/CM"

    if broad_position == "FWD":

        if (
            wide_distance
            >= FORWARD_WIDE_DISTANCE
        ):
            return "W"

        return "ST"

    return None


def classify_position_side(
    position_group,
    y_attacking,
):
    """
    Convert normalized lateral position into LEFT/CENTRAL/RIGHT.

    This is kept separate from position_group so percentiles can still
    be calculated over broader role populations.
    """

    if pd.isna(position_group):
        return None

    if pd.isna(y_attacking):
        return None

    y_attacking = float(
        y_attacking
    )

    if position_group in {
        "GK",
        "ST",
        "AM",
    }:
        return "CENTRAL"

    if y_attacking < SIDE_LEFT_MAX:
        return "RIGHT"

    if y_attacking > SIDE_RIGHT_MIN:
        return "LEFT"

    return "CENTRAL"


def build_tactical_position(
    position_group,
    position_side,
):
    """
    Create a presentation/recruitment-friendly position label.
    """

    if pd.isna(position_group):
        return None

    if position_group == "GK":
        return "GK"

    if position_group == "ST":
        return "ST"

    if position_group == "AM":
        return "AM"

    if position_group == "FB/WB":

        if position_side == "LEFT":
            return "LB/WB"

        if position_side == "RIGHT":
            return "RB/WB"

        return "FB/WB"

    if position_group == "CB":

        if position_side == "LEFT":
            return "LCB"

        if position_side == "RIGHT":
            return "RCB"

        return "CB"

    if position_group == "DM/CM":

        if position_side == "LEFT":
            return "LCM/DM"

        if position_side == "RIGHT":
            return "RCM/DM"

        return "CM/DM"

    if position_group == "W":

        if position_side == "LEFT":
            return "LW"

        if position_side == "RIGHT":
            return "RW"

        return "W"

    return position_group


# ---------------------------------------------------------
# Build match-level tactical positions
# ---------------------------------------------------------

def build_player_match_positions(
    lineups: pd.DataFrame,
    average_positions: pd.DataFrame,
) -> pd.DataFrame:

    required_lineup_columns = [
        "event_id",
        "player_id",
        "team_id",
        "observed_position",
    ]

    missing_lineup_columns = [
        column
        for column
        in required_lineup_columns
        if column
        not in lineups.columns
    ]

    if missing_lineup_columns:
        raise ValueError(
            "lineups.parquet is missing required columns: "
            f"{missing_lineup_columns}"
        )

    required_average_columns = [
        "event_id",
        "player_id",
        "avg_x",
        "avg_y",
    ]

    missing_average_columns = [
        column
        for column
        in required_average_columns
        if column
        not in average_positions.columns
    ]

    if missing_average_columns:
        raise ValueError(
            "average_positions.parquet is missing required columns: "
            f"{missing_average_columns}"
        )

    lineup = (
        lineups[
            required_lineup_columns
        ]
        .copy()
        .drop_duplicates(
            subset=[
                "event_id",
                "player_id",
            ],
            keep="first",
        )
    )

    lineup[
        "broad_position"
    ] = lineup[
        "observed_position"
    ].apply(
        normalize_broad_position
    )

    avg_columns = [
        "event_id",
        "player_id",
        "avg_x",
        "avg_y",
    ]

    if (
        "team_id"
        in average_positions.columns
    ):
        avg_columns.append(
            "team_id"
        )

    avg = (
        average_positions[
            avg_columns
        ]
        .copy()
        .drop_duplicates(
            subset=[
                "event_id",
                "player_id",
            ],
            keep="first",
        )
    )

    for column in [
        "event_id",
        "player_id",
        "team_id",
        "avg_x",
        "avg_y",
    ]:
        if (
            column
            in avg.columns
        ):
            avg[column] = pd.to_numeric(
                avg[column],
                errors="coerce",
            )

        if (
            column
            in lineup.columns
        ):
            lineup[column] = pd.to_numeric(
                lineup[column],
                errors="coerce",
            )

    merged = lineup.merge(
        avg,
        on=[
            "event_id",
            "player_id",
        ],
        how="left",
        suffixes=(
            "_lineup",
            "_avg",
        ),
        validate="one_to_one",
    )

    if (
        "team_id_lineup"
        in merged.columns
    ):
        merged[
            "team_id"
        ] = merged[
            "team_id_lineup"
        ]

        if (
            "team_id_avg"
            in merged.columns
        ):
            merged[
                "team_id"
            ] = merged[
                "team_id"
            ].fillna(
                merged[
                    "team_id_avg"
                ]
            )

    elif (
        "team_id"
        not in merged.columns
    ):
        merged[
            "team_id"
        ] = np.nan

    merged[
        "attack_direction"
    ] = None

    merged[
        "x_attacking"
    ] = np.nan

    merged[
        "y_attacking"
    ] = np.nan

    merged[
        "inferred_position_group"
    ] = None

    merged[
        "position_side"
    ] = None

    merged[
        "tactical_position"
    ] = None

    grouped = (
        merged
        .dropna(
            subset=[
                "team_id",
            ]
        )
        .groupby(
            [
                "event_id",
                "team_id",
            ]
        )
    )

    for (
        event_id,
        team_id
    ), group in grouped:

        indices = group.index

        direction = (
            infer_attack_direction(
                group
            )
        )

        merged.loc[
            indices,
            "attack_direction",
        ] = direction

        if direction == "right":

            normalized_x = (
                group[
                    "avg_x"
                ]
            )

            normalized_y = (
                group[
                    "avg_y"
                ]
            )

        else:

            normalized_x = (
                100.0
                - group[
                    "avg_x"
                ]
            )

            normalized_y = (
                100.0
                - group[
                    "avg_y"
                ]
            )

        merged.loc[
            indices,
            "x_attacking",
        ] = normalized_x

        merged.loc[
            indices,
            "y_attacking",
        ] = normalized_y

        for index in indices:

            role = classify_tactical_role(
                broad_position=merged.loc[
                    index,
                    "broad_position",
                ],
                x_attacking=merged.loc[
                    index,
                    "x_attacking",
                ],
                y_attacking=merged.loc[
                    index,
                    "y_attacking",
                ],
            )

            side = classify_position_side(
                position_group=role,
                y_attacking=merged.loc[
                    index,
                    "y_attacking",
                ],
            )

            tactical_position = (
                build_tactical_position(
                    position_group=role,
                    position_side=side,
                )
            )

            merged.loc[
                index,
                "inferred_position_group",
            ] = role

            merged.loc[
                index,
                "position_side",
            ] = side

            merged.loc[
                index,
                "tactical_position",
            ] = tactical_position

    return merged


# ---------------------------------------------------------
# Summarize player roles across matches
# ---------------------------------------------------------

def summarize_player_positions(
    history: pd.DataFrame,
) -> pd.DataFrame:

    valid = history[
        history[
            "inferred_position_group"
        ].notna()
    ].copy()

    rows = []

    for player_id, group in (
        valid.groupby(
            "player_id",
            sort=False,
        )
    ):

        role_counts = (
            group[
                "inferred_position_group"
            ]
            .value_counts()
        )

        if role_counts.empty:
            continue

        role_max = (
            role_counts.max()
        )

        tied_roles = (
            role_counts[
                role_counts
                == role_max
            ]
            .index
            .tolist()
        )

        if (
            len(tied_roles)
            == 1
        ):
            primary_role = (
                tied_roles[0]
            )

        else:

            primary_role = (
                group[
                    group[
                        "inferred_position_group"
                    ].isin(
                        tied_roles
                    )
                ]
                .sort_values(
                    "event_id"
                )
                .iloc[-1][
                    "inferred_position_group"
                ]
            )

        role_group = group[
            group[
                "inferred_position_group"
            ]
            == primary_role
        ].copy()

        side_counts = (
            role_group[
                "position_side"
            ]
            .dropna()
            .value_counts()
        )

        if side_counts.empty:
            primary_side = None
            side_share = np.nan

        else:

            side_max = (
                side_counts.max()
            )

            tied_sides = (
                side_counts[
                    side_counts
                    == side_max
                ]
                .index
                .tolist()
            )

            if (
                len(tied_sides)
                == 1
            ):
                primary_side = (
                    tied_sides[0]
                )

            else:

                primary_side = (
                    role_group[
                        role_group[
                            "position_side"
                        ].isin(
                            tied_sides
                        )
                    ]
                    .sort_values(
                        "event_id"
                    )
                    .iloc[-1][
                        "position_side"
                    ]
                )

            side_share = (
                side_max
                / len(role_group)
            )

        tactical_counts = (
            role_group[
                "tactical_position"
            ]
            .dropna()
            .value_counts()
        )

        if tactical_counts.empty:
            primary_tactical_position = (
                build_tactical_position(
                    primary_role,
                    primary_side,
                )
            )

        else:
            primary_tactical_position = (
                tactical_counts.index[0]
            )

        rows.append(
            {
                "player_id": int(
                    player_id
                ),

                "primary_position_group": (
                    primary_role
                ),

                "position_matches_observed": (
                    len(group)
                ),

                "position_group_share": (
                    role_max
                    / len(group)
                ),

                "observed_position_groups": (
                    ",".join(
                        f"{role}:{count}"
                        for role, count
                        in role_counts.items()
                    )
                ),

                "primary_position_side": (
                    primary_side
                ),

                "position_side_share": (
                    side_share
                ),

                "observed_position_sides": (
                    ",".join(
                        f"{side}:{count}"
                        for side, count
                        in side_counts.items()
                    )
                    if not side_counts.empty
                    else None
                ),

                "tactical_position": (
                    primary_tactical_position
                ),

                "average_x_attacking": (
                    group[
                        "x_attacking"
                    ]
                    .mean()
                ),

                "average_y_attacking": (
                    group[
                        "y_attacking"
                    ]
                    .mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    for path in [
        PLAYER_FILE,
        LINEUPS_FILE,
        AVERAGE_POSITIONS_FILE,
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

    average_positions = (
        pd.read_parquet(
            AVERAGE_POSITIONS_FILE
        )
    )

    season = pd.read_parquet(
        SEASON_FILE
    )

    print()
    print("=" * 96)
    print(
        "BUILDING BSD TACTICAL PLAYER POSITIONS + SIDES"
        .center(96)
    )
    print("=" * 96)

    print(
        f"Players: "
        f"{len(players):,}"
    )

    print(
        f"Lineup rows: "
        f"{len(lineups):,}"
    )

    print(
        f"Average-position rows: "
        f"{len(average_positions):,}"
    )

    print(
        f"Season rows: "
        f"{len(season):,}"
    )

    history = (
        build_player_match_positions(
            lineups=lineups,
            average_positions=average_positions,
        )
    )

    position_summary = (
        summarize_player_positions(
            history
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
        for column
        in metadata_columns
        if column
        in players.columns
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
        normalize_broad_position
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
        "average_position_inference",
        "profile_fallback",
    )

    if (
        "date_of_birth"
        in enriched.columns
    ):

        dob = pd.to_datetime(
            enriched[
                "date_of_birth"
            ],
            errors="coerce",
        )

        reference_date = (
            pd.Timestamp(
                "2026-07-01"
            )
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

    preferred_columns = [
        "player_id",
        "player_name",
        "position",
        "profile_position_group",
        "primary_position_group",
        "position_group",
        "primary_position_side",
        "position_side_share",
        "tactical_position",
        "position_source",
        "position_matches_observed",
        "position_group_share",
        "observed_position_groups",
        "observed_position_sides",
        "average_x_attacking",
        "average_y_attacking",
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
        for column
        in preferred_columns
        if column
        in enriched.columns
    ]

    remaining_columns = [
        column
        for column
        in enriched.columns
        if column
        not in preferred_columns
    ]

    enriched = enriched[
        preferred_columns
        + remaining_columns
    ]

    enriched.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 96)
    print(
        "BSD TACTICAL POSITION + SIDE SUMMARY"
        .center(96)
    )
    print("=" * 96)

    print(
        f"Season players: "
        f"{len(enriched):,}"
    )

    print(
        "Players with inferred tactical role: "
        f"{enriched['primary_position_group'].notna().sum():,}"
    )

    print(
        "Profile fallbacks: "
        f"{enriched['primary_position_group'].isna().sum():,}"
    )

    print()
    print(
        "Position groups:"
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
        "Tactical positions:"
    )

    print(
        enriched[
            "tactical_position"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Primary sides:"
    )

    print(
        enriched[
            "primary_position_side"
        ]
        .value_counts(
            dropna=False
        )
    )

    print()
    print(
        "Newcastle United position check:"
    )

    newcastle_columns = [
        "player_name",
        "position_group",
        "primary_position_side",
        "tactical_position",
        "position_group_share",
        "position_side_share",
        "observed_position_groups",
        "observed_position_sides",
        "minutes",
    ]

    newcastle_columns = [
        column
        for column
        in newcastle_columns
        if column
        in enriched.columns
    ]

    newcastle = enriched[
        enriched["team_name"]
        == "Newcastle United"
    ].copy()

    if not newcastle.empty:

        print(
            newcastle[
                newcastle_columns
            ]
            .sort_values(
                [
                    "position_group",
                    "minutes",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Lowest role certainty "
        "(minimum 2 observed matches):"
    )

    uncertainty = enriched[
        enriched[
            "position_matches_observed"
        ]
        .fillna(0)
        >= 2
    ]

    uncertainty_columns = [
        "player_name",
        "position",
        "primary_position_group",
        "primary_position_side",
        "tactical_position",
        "position_matches_observed",
        "position_group_share",
        "position_side_share",
        "observed_position_groups",
        "observed_position_sides",
    ]

    uncertainty_columns = [
        column
        for column
        in uncertainty_columns
        if column
        in uncertainty.columns
    ]

    if not uncertainty.empty:

        print(
            uncertainty
            .sort_values(
                [
                    "position_group_share",
                    "position_side_share",
                    "position_matches_observed",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            [
                uncertainty_columns
            ]
            .head(30)
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
