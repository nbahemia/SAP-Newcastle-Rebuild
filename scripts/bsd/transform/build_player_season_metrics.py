from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bsd"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bsd"
)

PLAYER_STATS_FILE = (
    RAW_DIR
    / "player_match_stats.parquet"
)

SHOTS_FILE = (
    RAW_DIR
    / "shots.parquet"
)

MATCHES_FILE = (
    RAW_DIR
    / "matches.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "player_season_metrics.parquet"
)

PLAYER_TEAM_OUTPUT_FILE = (
    PROCESSED_DIR
    / "player_team_season_metrics.parquet"
)


# ---------------------------------------------------------
# Season configuration
# ---------------------------------------------------------

LEAGUE_ID = 1
SEASON_ID = 1058


# ---------------------------------------------------------
# 0-100 pitch geometry
# ---------------------------------------------------------
#
# BSD documents shot origin "pos" on a 0-100 x 0-100 pitch.
#
# For shots-in-box, use standard penalty-area dimensions:
#
#   pitch length = 105 m
#   pitch width  = 68 m
#   penalty area depth = 16.5 m
#   penalty area width = 40.32 m
#
# Converted to BSD's 0-100 coordinate system:
#
#   x >= ~84.29
#   y between ~20.35 and ~79.65
#
# This is a geometric derived metric, not a native BSD field.
# ---------------------------------------------------------

PENALTY_AREA_X_MAX = (
    16.5
    / 105.0
    * 100.0
)

PENALTY_AREA_Y_MIN = (
    (68.0 - 40.32)
    / 2.0
    / 68.0
    * 100.0
)

PENALTY_AREA_Y_MAX = (
    100.0
    - PENALTY_AREA_Y_MIN
)


# ---------------------------------------------------------
# Player-stat metrics to aggregate
# ---------------------------------------------------------

SUM_COLUMNS = [
    # Output
    "goals",
    "goal_assist",
    "expected_assists",

    # Shooting feed values
    # NOTE: total_shots and expected_goals are retained
    # diagnostically, but shotmap values become canonical
    # season shot/xG metrics later.
    "total_shots",
    "shots_on_target",
    "expected_goals",
    "expected_goals_on_target",
    "blocked_scoring_attempt",
    "hit_woodwork",
    "big_chance_created",
    "big_chance_missed",

    # Passing
    "total_pass",
    "accurate_pass",
    "key_pass",
    "total_long_balls",
    "accurate_long_balls",
    "total_cross",
    "accurate_cross",
    "accurate_opposition_half_passes",
    "total_opposition_half_passes",
    "accurate_own_half_passes",
    "total_own_half_passes",

    # Carrying / dribbling
    "total_contest",
    "won_contest",
    "ball_carries_count",
    "progressive_ball_carries_count",
    "total_ball_carries_distance",
    "total_progressive_ball_carries_distance",
    "total_progression",

    # Defensive / physical
    "duel_won",
    "duel_lost",
    "aerial_won",
    "aerial_lost",
    "total_tackle",
    "won_tackle",
    "total_clearance",
    "interception",
    "ball_recovery",
    "challenge_lost",

    # Possession / discipline
    "touches",
    "dispossessed",
    "possession_lost",
    "unsuccessful_touch",
    "was_fouled",
    "fouls",
    "yellow_card",
    "red_card",

    # Goalkeeping
    "saves",
    "goals_conceded",
    "punches",
    "high_claims",
    "good_high_claim",
    "accurate_keeper_sweeper",
    "total_keeper_sweeper",
    "saved_shots_from_inside_the_box",

    # Errors / special defensive actions
    "last_man_tackle",
    "outfielder_block",
    "clearance_off_line",
    "error_lead_to_a_goal",
    "error_lead_to_a_shot",
]


# Metrics for which /90 values are useful.
PER90_COLUMNS = [
    # Canonical shotmap-derived
    "shots",
    "shots_on_target_canonical",
    "shots_in_box",
    "goals_shotmap",
    "xg",

    # Player output
    "goals",
    "goal_assist",
    "expected_assists",
    "key_pass",

    # Passing
    "total_pass",
    "accurate_pass",
    "total_long_balls",
    "accurate_long_balls",
    "total_cross",
    "accurate_cross",
    "accurate_opposition_half_passes",
    "total_opposition_half_passes",
    "accurate_own_half_passes",
    "total_own_half_passes",

    # Carrying
    "total_contest",
    "won_contest",
    "ball_carries_count",
    "progressive_ball_carries_count",
    "total_ball_carries_distance",
    "total_progressive_ball_carries_distance",
    "total_progression",

    # Defensive / physical
    "duel_won",
    "duel_lost",
    "aerial_won",
    "aerial_lost",
    "total_tackle",
    "won_tackle",
    "total_clearance",
    "interception",
    "ball_recovery",

    # Possession
    "touches",
    "dispossessed",
    "possession_lost",
    "unsuccessful_touch",
    "was_fouled",
    "fouls",

    # Goalkeeping
    "saves",
    "goals_conceded",
    "high_claims",
    "good_high_claim",
    "saved_shots_from_inside_the_box",
]


# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------

def require_file(
    path: Path,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n"
            f"{path}"
        )


def ensure_numeric(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    for column in columns:

        if column in dataframe.columns:

            dataframe[column] = (
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )
            )

    return dataframe


# ---------------------------------------------------------
# Match metadata
# ---------------------------------------------------------

def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


def build_match_metadata(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a minimal event_id -> match date mapping.

    The exact normalized matches schema may vary, so this
    deliberately checks several common date fields.
    """

    id_column = find_first_existing_column(
        matches,
        [
            "id",
            "event_id",
        ],
    )

    if id_column is None:
        raise ValueError(
            "Could not find match id column "
            "in matches.parquet."
        )

    date_column = find_first_existing_column(
        matches,
        [
            "event_date",
            "start_time",
            "start_timestamp",
            "kickoff",
            "date",
        ],
    )

    metadata = pd.DataFrame(
        {
            "event_id": pd.to_numeric(
                matches[id_column],
                errors="coerce",
            )
        }
    )

    if date_column is not None:

        metadata["match_date"] = pd.to_datetime(
            matches[date_column],
            errors="coerce",
            utc=True,
        )

    else:

        metadata["match_date"] = pd.NaT

    return (
        metadata
        .dropna(
            subset=["event_id"]
        )
        .drop_duplicates(
            subset=["event_id"]
        )
    )


# ---------------------------------------------------------
# Shotmap transformation
# ---------------------------------------------------------

def build_shot_metrics(
    shots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build canonical per-player shooting totals from BSD shotmaps.

    BSD recommends the shotmap as the per-shot source of truth.
    Shootout attempts are excluded.
    """

    shots = shots.copy()

    required = [
        "event_id",
        "player_id",
    ]

    missing = [
        column
        for column in required
        if column not in shots.columns
    ]

    if missing:
        raise ValueError(
            "shots.parquet is missing required "
            f"columns: {missing}"
        )

    # Exclude shootout attempts.
    if "sit" in shots.columns:

        shots = shots[
            shots["sit"]
            .fillna("")
            .str.casefold()
            .ne("shootout")
        ].copy()

    # Numeric normalization.
    numeric_columns = [
        "event_id",
        "player_id",
        "xg",
        "xgot",
        "pos_x",
        "pos_y",
    ]

    shots = ensure_numeric(
        shots,
        numeric_columns,
    )

    shots = shots.dropna(
        subset=[
            "event_id",
            "player_id",
        ]
    )

    # One row = one shot.
    shots["shots"] = 1

    # BSD: on target = goal or save.
    if "type" in shots.columns:

        shot_type = (
            shots["type"]
            .fillna("")
            .str.casefold()
        )

        shots[
            "shots_on_target_canonical"
        ] = (
            shot_type
            .isin(
                [
                    "goal",
                    "save",
                ]
            )
            .astype(int)
        )

        shots[
            "goals_shotmap"
        ] = (
            shot_type
            .eq("goal")
            .astype(int)
        )

    else:

        shots[
            "shots_on_target_canonical"
        ] = 0

        shots[
            "goals_shotmap"
        ] = 0

    # Derived penalty-box geometry.
    if (
        "pos_x" in shots.columns
        and "pos_y" in shots.columns
    ):

        shots[
            "shots_in_box"
        ] = (
            (
                shots["pos_x"]
                <= PENALTY_AREA_X_MAX
            )
            &
            (
                shots["pos_y"]
                >= PENALTY_AREA_Y_MIN
            )
            &
            (
                shots["pos_y"]
                <= PENALTY_AREA_Y_MAX
            )
        ).astype(int)

    else:

        shots[
            "shots_in_box"
        ] = 0

    # Preserve only measured/served xG values.
    # Do not replace missing xG with zero before aggregation.
    if "xg" not in shots.columns:
        shots["xg"] = np.nan

    shot_metrics = (
        shots
        .groupby(
            [
                "event_id",
                "player_id",
            ],
            as_index=False,
        )
        .agg(
            shots=(
                "shots",
                "sum",
            ),
            shots_on_target_canonical=(
                "shots_on_target_canonical",
                "sum",
            ),
            shots_in_box=(
                "shots_in_box",
                "sum",
            ),
            goals_shotmap=(
                "goals_shotmap",
                "sum",
            ),
            xg=(
                "xg",
                lambda series: (
                    series.sum(
                        min_count=1
                    )
                ),
            ),
        )
    )

    print()
    print("Shot coordinate ranges:")
    print(
        f"  x: "
        f"{shots['pos_x'].min():.1f} "
        f"to "
        f"{shots['pos_x'].max():.1f}"
    )
    print(
        f"  y: "
        f"{shots['pos_y'].min():.1f} "
        f"to "
        f"{shots['pos_y'].max():.1f}"
    )

    print(
        f"Shots classified in box: "
        f"{shots['shots_in_box'].sum():,} "
        f"/ {len(shots):,}"
    )
    
    return shot_metrics


# ---------------------------------------------------------
# Player-match cleanup
# ---------------------------------------------------------

def prepare_player_match_stats(
    player_stats: pd.DataFrame,
    match_metadata: pd.DataFrame,
    shot_metrics: pd.DataFrame,
) -> pd.DataFrame:

    player_stats = player_stats.copy()

    required = [
        "event_id",
        "player_id",
        "team_id",
        "minutes_played",
    ]

    missing = [
        column
        for column in required
        if column not in player_stats.columns
    ]

    if missing:
        raise ValueError(
            "player_match_stats.parquet is "
            f"missing required columns: {missing}"
        )

    player_stats = ensure_numeric(
        player_stats,
        [
            "event_id",
            "player_id",
            "team_id",
            "minutes_played",
            "rating",
            *SUM_COLUMNS,
        ],
    )

    # BSD player-stats includes unused bench players.
    player_stats = player_stats[
        player_stats[
            "minutes_played"
        ].fillna(0)
        > 0
    ].copy()

    player_stats = player_stats.merge(
        match_metadata,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    player_stats = player_stats.merge(
        shot_metrics,
        on=[
            "event_id",
            "player_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # No shotmap row means zero shots, but xG stays
    # zero only when the player had zero shots.
    for column in [
        "shots",
        "shots_on_target_canonical",
        "shots_in_box",
        "goals_shotmap",
    ]:

        if column not in player_stats.columns:
            player_stats[column] = 0

        player_stats[column] = (
            player_stats[column]
            .fillna(0)
        )

    if "xg" not in player_stats.columns:
        player_stats["xg"] = np.nan

    player_stats["xg"] = np.where(
        player_stats["shots"] == 0,
        0.0,
        player_stats["xg"],
    )

    return player_stats
    
# ---------------------------------------------------------
# Season aggregation
# ---------------------------------------------------------

def aggregate_player_team_season(
    player_match: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per player-team combination.
    """

    existing_sum_columns = [
        column
        for column in SUM_COLUMNS
        if column in player_match.columns
    ]

    canonical_columns = [
        "shots",
        "shots_on_target_canonical",
        "shots_in_box",
        "goals_shotmap",
        "xg",
    ]

    aggregation = {
        "minutes_played": "sum",
        "event_id": "nunique",
    }

    for column in (
        existing_sum_columns
        + canonical_columns
    ):

        aggregation[column] = "sum"

    grouped = (
        player_match
        .groupby(
            [
                "player_id",
                "team_id",
            ],
            as_index=False,
        )
        .agg(
            aggregation
        )
        .rename(
            columns={
                "minutes_played": "minutes",
                "event_id": "appearances",
            }
        )
    )

    # Minutes-weighted average rating.
    if "rating" in player_match.columns:

        rating_source = (
            player_match[
                [
                    "player_id",
                    "team_id",
                    "minutes_played",
                    "rating",
                ]
            ]
            .dropna(
                subset=["rating"]
            )
            .copy()
        )

        if not rating_source.empty:

            rating_source[
                "rating_weighted"
            ] = (
                rating_source[
                    "rating"
                ]
                *
                rating_source[
                    "minutes_played"
                ]
            )

            ratings = (
                rating_source
                .groupby(
                    [
                        "player_id",
                        "team_id",
                    ],
                    as_index=False,
                )
                .agg(
                    rating_weighted_sum=(
                        "rating_weighted",
                        "sum",
                    ),
                    rating_minutes=(
                        "minutes_played",
                        "sum",
                    ),
                )
            )

            ratings[
                "average_rating"
            ] = (
                ratings[
                    "rating_weighted_sum"
                ]
                /
                ratings[
                    "rating_minutes"
                ]
            )

            grouped = grouped.merge(
                ratings[
                    [
                        "player_id",
                        "team_id",
                        "average_rating",
                    ]
                ],
                on=[
                    "player_id",
                    "team_id",
                ],
                how="left",
            )

    grouped = add_derived_metrics(
        grouped
    )

    return grouped


def aggregate_player_season(
    player_match: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per player across the Premier League season.

    For a player who changed EPL clubs mid-season, all EPL
    minutes are combined. `current_team_id` is the team from
    the most recent appearance and `team_ids` preserves all
    clubs represented in the data.
    """

    existing_sum_columns = [
        column
        for column in SUM_COLUMNS
        if column in player_match.columns
    ]

    canonical_columns = [
        "shots",
        "shots_on_target_canonical",
        "shots_in_box",
        "goals_shotmap",
        "xg",
    ]

    aggregation = {
        "minutes_played": "sum",
        "event_id": "nunique",
    }

    for column in (
        existing_sum_columns
        + canonical_columns
    ):

        aggregation[column] = "sum"

    grouped = (
        player_match
        .groupby(
            "player_id",
            as_index=False,
        )
        .agg(
            aggregation
        )
        .rename(
            columns={
                "minutes_played": "minutes",
                "event_id": "appearances",
            }
        )
    )

    # Team history.
    team_history = (
        player_match
        .groupby(
            "player_id"
        )["team_id"]
        .agg(
            lambda series: ",".join(
                str(int(value))
                for value in pd.unique(
                    series.dropna()
                )
            )
        )
        .rename(
            "team_ids"
        )
        .reset_index()
    )

    grouped = grouped.merge(
        team_history,
        on="player_id",
        how="left",
    )

    # Most recent team.
    latest_team_source = (
        player_match
        .sort_values(
            [
                "player_id",
                "match_date",
                "event_id",
            ],
            na_position="first",
        )
        .groupby(
            "player_id",
            as_index=False,
        )
        .tail(1)
        [
            [
                "player_id",
                "team_id",
            ]
        ]
        .rename(
            columns={
                "team_id": "current_team_id",
            }
        )
    )

    grouped = grouped.merge(
        latest_team_source,
        on="player_id",
        how="left",
    )

    # Weighted rating.
    if "rating" in player_match.columns:

        rating_source = (
            player_match[
                [
                    "player_id",
                    "minutes_played",
                    "rating",
                ]
            ]
            .dropna(
                subset=["rating"]
            )
            .copy()
        )

        if not rating_source.empty:

            rating_source[
                "rating_weighted"
            ] = (
                rating_source[
                    "rating"
                ]
                *
                rating_source[
                    "minutes_played"
                ]
            )

            ratings = (
                rating_source
                .groupby(
                    "player_id",
                    as_index=False,
                )
                .agg(
                    rating_weighted_sum=(
                        "rating_weighted",
                        "sum",
                    ),
                    rating_minutes=(
                        "minutes_played",
                        "sum",
                    ),
                )
            )

            ratings[
                "average_rating"
            ] = (
                ratings[
                    "rating_weighted_sum"
                ]
                /
                ratings[
                    "rating_minutes"
                ]
            )

            grouped = grouped.merge(
                ratings[
                    [
                        "player_id",
                        "average_rating",
                    ]
                ],
                on="player_id",
                how="left",
            )

    grouped = add_derived_metrics(
        grouped
    )

    return grouped


# ---------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------

def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    return np.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )


def add_derived_metrics(
    season: pd.DataFrame,
) -> pd.DataFrame:

    season = season.copy()

    # -----------------------------------------------------
    # Per-90 values
    # -----------------------------------------------------

    for column in PER90_COLUMNS:

        if column not in season.columns:
            continue

        season[
            f"{column}_per90"
        ] = np.where(
            season["minutes"] > 0,
            season[column]
            / season["minutes"]
            * 90.0,
            np.nan,
        )

    # Friendly aliases used by scouting models.
    alias_map = {
        "xg_per90": "xg_per90",
        "shots_per90": "shots_per90",
        "shots_in_box_per90": "shots_in_box_per90",
        "goals_per90": "goals_per90",
        "assists_per90": "goal_assist_per90",
        "xa_per90": "expected_assists_per90",
        "passes_per90": "total_pass_per90",
        "successful_passes_per90": "accurate_pass_per90",
        "progressive_carries_per90": (
            "progressive_ball_carries_count_per90"
        ),
        "recoveries_per90": "ball_recovery_per90",
        "interceptions_per90": "interception_per90",
        "duels_won_per90": "duel_won_per90",
        "duels_lost_per90": "duel_lost_per90",
        "aerials_won_per90": "aerial_won_per90",
        "aerials_lost_per90": "aerial_lost_per90",
        "tackles_per90": "total_tackle_per90",
        "tackles_won_per90": "won_tackle_per90",
        "touches_per90": "touches_per90",
    }

    for alias, source in alias_map.items():

        if source in season.columns:
            season[alias] = season[source]

    # -----------------------------------------------------
    # Percent / efficiency metrics
    # -----------------------------------------------------

    if {
        "accurate_pass",
        "total_pass",
    }.issubset(
        season.columns
    ):

        season[
            "pass_accuracy_pct"
        ] = (
            safe_ratio(
                season["accurate_pass"],
                season["total_pass"],
            )
            * 100.0
        )

    if {
        "accurate_long_balls",
        "total_long_balls",
    }.issubset(
        season.columns
    ):

        season[
            "long_ball_accuracy_pct"
        ] = (
            safe_ratio(
                season["accurate_long_balls"],
                season["total_long_balls"],
            )
            * 100.0
        )

    if {
        "accurate_cross",
        "total_cross",
    }.issubset(
        season.columns
    ):

        season[
            "cross_accuracy_pct"
        ] = (
            safe_ratio(
                season["accurate_cross"],
                season["total_cross"],
            )
            * 100.0
        )

    if {
        "won_contest",
        "total_contest",
    }.issubset(
        season.columns
    ):

        season[
            "dribble_success_pct"
        ] = (
            safe_ratio(
                season["won_contest"],
                season["total_contest"],
            )
            * 100.0
        )

    if {
        "duel_won",
        "duel_lost",
    }.issubset(
        season.columns
    ):

        duel_total = (
            season["duel_won"]
            + season["duel_lost"]
        )

        season[
            "duel_win_pct"
        ] = (
            safe_ratio(
                season["duel_won"],
                duel_total,
            )
            * 100.0
        )

    if {
        "aerial_won",
        "aerial_lost",
    }.issubset(
        season.columns
    ):

        aerial_total = (
            season["aerial_won"]
            + season["aerial_lost"]
        )

        season[
            "aerial_win_pct"
        ] = (
            safe_ratio(
                season["aerial_won"],
                aerial_total,
            )
            * 100.0
        )

    if {
        "won_tackle",
        "total_tackle",
    }.issubset(
        season.columns
    ):

        season[
            "tackle_win_pct"
        ] = (
            safe_ratio(
                season["won_tackle"],
                season["total_tackle"],
            )
            * 100.0
        )

    if {
        "shots_on_target_canonical",
        "shots",
    }.issubset(
        season.columns
    ):

        season[
            "shot_on_target_pct"
        ] = (
            safe_ratio(
                season[
                    "shots_on_target_canonical"
                ],
                season["shots"],
            )
            * 100.0
        )

    if {
        "shots_in_box",
        "shots",
    }.issubset(
        season.columns
    ):

        season[
            "shots_in_box_pct"
        ] = (
            safe_ratio(
                season[
                    "shots_in_box"
                ],
                season["shots"],
            )
            * 100.0
        )

    # -----------------------------------------------------
    # Diagnostics comparing player-stats vs shotmap feeds
    # -----------------------------------------------------

    if {
        "total_shots",
        "shots",
    }.issubset(
        season.columns
    ):

        season[
            "shot_count_feed_difference"
        ] = (
            season["total_shots"]
            - season["shots"]
        )

    if {
        "expected_goals",
        "xg",
    }.issubset(
        season.columns
    ):

        season[
            "xg_feed_difference"
        ] = (
            season["expected_goals"]
            - season["xg"]
        )

    return season


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    require_file(
        PLAYER_STATS_FILE
    )

    require_file(
        SHOTS_FILE
    )

    require_file(
        MATCHES_FILE
    )

    print()
    print("=" * 88)
    print(
        "BUILDING BSD PLAYER SEASON METRICS"
        .center(88)
    )
    print("=" * 88)

    player_stats = pd.read_parquet(
        PLAYER_STATS_FILE
    )

    shots = pd.read_parquet(
        SHOTS_FILE
    )

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    print(
        f"Loaded "
        f"{len(player_stats):,} "
        f"player-match rows"
    )

    print(
        f"Loaded "
        f"{len(shots):,} "
        f"shot rows"
    )

    print(
        f"Loaded "
        f"{len(matches):,} "
        f"matches"
    )

    match_metadata = (
        build_match_metadata(
            matches
        )
    )

    shot_metrics = (
        build_shot_metrics(
            shots
        )
    )

    print()
    print(
        f"Shotmap player-match groups: "
        f"{len(shot_metrics):,}"
    )

    player_match = (
        prepare_player_match_stats(
            player_stats=player_stats,
            match_metadata=match_metadata,
            shot_metrics=shot_metrics,
        )
    )

    print(
        f"Played player-match rows: "
        f"{len(player_match):,}"
    )

    player_team_season = (
        aggregate_player_team_season(
            player_match
        )
    )

    player_season = (
        aggregate_player_season(
            player_match
        )
    )

    player_team_season = (
        player_team_season
        .sort_values(
            [
                "minutes",
                "player_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    player_season = (
        player_season
        .sort_values(
            [
                "minutes",
                "player_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_season.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    player_team_season.to_parquet(
        PLAYER_TEAM_OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 88)
    print(
        "BSD SEASON METRICS SUMMARY"
        .center(88)
    )
    print("=" * 88)

    print(
        f"Player-season rows:      "
        f"{len(player_season):,}"
    )

    print(
        f"Player-team rows:        "
        f"{len(player_team_season):,}"
    )

    print(
        f"Total minutes:           "
        f"{player_season['minutes'].sum():,.0f}"
    )

    print(
        f"Total canonical shots:   "
        f"{player_season['shots'].sum():,.0f}"
    )

    print(
        f"Total canonical xG:      "
        f"{player_season['xg'].sum():,.2f}"
    )

    print(
        f"Total shots in box:      "
        f"{player_season['shots_in_box'].sum():,.0f}"
    )

    if (
        "shot_count_feed_difference"
        in player_season.columns
    ):

        mismatch_players = (
            player_season[
                "shot_count_feed_difference"
            ]
            .fillna(0)
            .ne(0)
            .sum()
        )

        print(
            f"Players with shot-feed "
            f"difference: {mismatch_players:,}"
        )

    print()
    print("Outputs:")
    print(f"  {OUTPUT_FILE}")
    print(
        f"  {PLAYER_TEAM_OUTPUT_FILE}"
    )

    display_columns = [
        "player_id",
        "current_team_id",
        "minutes",
        "appearances",
        "goals",
        "xg",
        "shots",
        "shots_in_box",
        "aerial_won",
        "duel_won",
        "progressive_ball_carries_count",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in player_season.columns
    ]

    print()
    print(
        "Top 15 by minutes:"
    )

    print(
        player_season[
            display_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()