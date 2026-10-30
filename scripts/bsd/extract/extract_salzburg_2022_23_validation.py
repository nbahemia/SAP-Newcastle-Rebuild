from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("BSD_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "BSD_API_KEY not found. Add BSD_API_KEY=... to the project .env file."
    )

BASE_URL = "https://sports.bzzoiro.com/api/v2"

LEAGUE_ID = 96
TEAM_NAME = "Red Bull Salzburg"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bsd" / "salzburg_2022_23"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_FILE = OUTPUT_DIR / "matches.parquet"
AVERAGE_POSITIONS_FILE = OUTPUT_DIR / "average_positions.parquet"
PLAYERS_FILE = OUTPUT_DIR / "players.parquet"
PLAYERS_RAW_FILE = OUTPUT_DIR / "players_raw.json"
POSITION_HISTORY_FILE = OUTPUT_DIR / "player_position_history.parquet"
ROLE_SUMMARY_FILE = OUTPUT_DIR / "role_summary.parquet"
MATCH_ROLE_COUNTS_FILE = OUTPUT_DIR / "match_role_counts.parquet"
RAW_META_FILE = OUTPUT_DIR / "validation_metadata.json"

HEADERS = {
    "Authorization": f"Token {API_KEY}",
}

REQUEST_SLEEP_SECONDS = 0.08


# ---------------------------------------------------------------------
# Same tactical thresholds as Newcastle pipeline
# ---------------------------------------------------------------------

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

DEFENDER_WIDE_DISTANCE = 24.0
FULLBACK_MIN_X = 38.0

MIDFIELDER_WIDE_DISTANCE = 27.0
FORWARD_WIDE_DISTANCE = 32.0
ATTACKING_MID_X = 60.0

SIDE_LEFT_MAX = 38.0
SIDE_RIGHT_MIN = 62.0

POSITION_ORDER = [
    "GK",
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
]


# ---------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------

def api_get(path: str, params: dict | None = None):
    url = f"{BASE_URL}{path}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"BSD request failed: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"Body: {response.text[:1000]}"
        )

    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.json()


def paginated_get(
    path: str,
    params: dict | None = None,
):
    params = dict(params or {})
    params["limit"] = 200
    params["offset"] = 0

    all_rows = []

    while True:
        payload = api_get(
            path,
            params=params,
        )

        if isinstance(payload, list):
            all_rows.extend(payload)
            break

        results = (
            payload.get("results", [])
            if isinstance(payload, dict)
            else []
        )

        all_rows.extend(results)

        next_url = (
            payload.get("next")
            if isinstance(payload, dict)
            else None
        )

        if not next_url:
            break

        params["offset"] += 200

    return all_rows


# ---------------------------------------------------------------------
# Season / team discovery
# ---------------------------------------------------------------------

def find_season():
    payload = api_get(
        f"/leagues/{LEAGUE_ID}/seasons/"
    )

    if isinstance(payload, dict):
        seasons = (
            payload.get("results")
            or payload.get("seasons")
            or [payload]
        )
    else:
        seasons = payload

    print("\nAvailable Austrian Bundesliga seasons:")

    for season in seasons:
        print(
            f"  id={season.get('id')} | "
            f"name={season.get('name')} | "
            f"year={season.get('year')} | "
            f"{season.get('start_date')} -> {season.get('end_date')}"
        )

    candidates = []

    for season in seasons:
        name = str(
            season.get("name", "")
        ).lower()

        year = str(
            season.get("year", "")
        )

        if (
            "22/23" in name
            or (
                year == "2022"
                and "bundesliga" in name
            )
        ):
            candidates.append(
                season
            )

    if not candidates:
        raise RuntimeError(
            "Could not identify Bundesliga 22/23."
        )

    selected = candidates[0]

    print(
        "\nSelected season:",
        selected.get("id"),
        selected.get("name"),
    )

    return selected


def find_team(season_id: int):
    teams = paginated_get(
        "/teams/",
        params={
            "league_id": LEAGUE_ID,
            "season_id": season_id,
        },
    )

    matching = [
        team
        for team in teams
        if "salzburg" in str(
            team.get("name", "")
        ).lower()
    ]

    if not matching:
        raise RuntimeError(
            "Could not find Red Bull Salzburg "
            f"in league {LEAGUE_ID}, season {season_id}."
        )

    selected = matching[0]

    print(
        "Selected team:",
        selected.get("id"),
        selected.get("name"),
    )

    return selected


# ---------------------------------------------------------------------
# Match extraction
# ---------------------------------------------------------------------

def extract_matches(
    season_id: int,
    team_id: int,
):
    matches = paginated_get(
        "/events/",
        params={
            "league_id": LEAGUE_ID,
            "season_id": season_id,
            "team_id": team_id,
            "status": "finished",
        },
    )

    rows = []

    for event in matches:
        home_team = event.get(
            "home_team"
        )

        away_team = event.get(
            "away_team"
        )

        home_team_id = (
            event.get("home_team_id")
        )

        away_team_id = (
            event.get("away_team_id")
        )

        home_team_name = (
            event.get("home_team_name")
        )

        away_team_name = (
            event.get("away_team_name")
        )

        if isinstance(
            home_team,
            dict,
        ):
            home_team_id = (
                home_team_id
                or home_team.get("id")
            )

            home_team_name = (
                home_team_name
                or home_team.get("name")
            )

        elif home_team_name is None:
            home_team_name = home_team

        if isinstance(
            away_team,
            dict,
        ):
            away_team_id = (
                away_team_id
                or away_team.get("id")
            )

            away_team_name = (
                away_team_name
                or away_team.get("name")
            )

        elif away_team_name is None:
            away_team_name = away_team

        rows.append(
            {
                "event_id":
                    event.get("id"),

                "event_date":
                    event.get("event_date"),

                "league_id":
                    event.get("league_id")
                    or LEAGUE_ID,

                "season_id":
                    event.get("season_id")
                    or season_id,

                "home_team_id":
                    home_team_id,

                "away_team_id":
                    away_team_id,

                "home_team":
                    home_team_name,

                "away_team":
                    away_team_name,

                "home_score":
                    event.get("home_score"),

                "away_score":
                    event.get("away_score"),

                "status":
                    event.get("status"),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Average-position extraction
# ---------------------------------------------------------------------

def recursive_find_average_positions(
    obj,
    event_id,
    match,
):
    """
    Pull rows from any nested list/dict that looks like an
    average-position record. This is intentionally permissive
    because historical payload shape can differ from current EPL.
    """

    rows = []

    def walk(
        node,
        context_key=None,
        inherited_team_id=None,
        inherited_home=None,
    ):
        if isinstance(
            node,
            dict,
        ):
            current_team_id = (
                node.get("team_id")
                or inherited_team_id
            )

            current_home = (
                node.get("home")
                if isinstance(
                    node.get("home"),
                    bool,
                )
                else inherited_home
            )

            if (
                context_key is not None
                and str(
                    context_key
                ).lower()
                in {
                    "home",
                    "home_team",
                }
            ):
                current_home = True

                if pd.notna(
                    match.get(
                        "home_team_id"
                    )
                ):
                    current_team_id = int(
                        match[
                            "home_team_id"
                        ]
                    )

            elif (
                context_key is not None
                and str(
                    context_key
                ).lower()
                in {
                    "away",
                    "away_team",
                }
            ):
                current_home = False

                if pd.notna(
                    match.get(
                        "away_team_id"
                    )
                ):
                    current_team_id = int(
                        match[
                            "away_team_id"
                        ]
                    )

            player_obj = (
                node.get("player")
                if isinstance(
                    node.get("player"),
                    dict,
                )
                else {}
            )

            player_id = (
                node.get("player_id")
                or player_obj.get("id")
            )

            player_name = (
                node.get("player_name")
                or node.get("name")
                or player_obj.get("name")
            )

            pos = node.get("pos")

            avg_x = None
            avg_y = None

            if isinstance(
                pos,
                dict,
            ):
                avg_x = (
                    pos.get("x")
                )
                avg_y = (
                    pos.get("y")
                )

            if avg_x is None:
                avg_x = (
                    node.get("avg_x")
                    or node.get("x")
                )

            if avg_y is None:
                avg_y = (
                    node.get("avg_y")
                    or node.get("y")
                )

            looks_like_position_row = (
                player_id is not None
                and avg_x is not None
                and avg_y is not None
            )

            if looks_like_position_row:
                rows.append(
                    {
                        "event_id":
                            event_id,

                        "team_id":
                            current_team_id,

                        "is_home":
                            current_home,

                        "player_id":
                            player_id,

                        "player_name":
                            player_name,

                        "avg_x":
                            avg_x,

                        "avg_y":
                            avg_y,

                        "touches":
                            node.get("n")
                            or node.get(
                                "touches"
                            ),
                    }
                )

            for key, value in (
                node.items()
            ):
                walk(
                    value,
                    context_key=key,
                    inherited_team_id=(
                        current_team_id
                    ),
                    inherited_home=(
                        current_home
                    ),
                )

        elif isinstance(
            node,
            list,
        ):
            for item in node:
                walk(
                    item,
                    context_key=(
                        context_key
                    ),
                    inherited_team_id=(
                        inherited_team_id
                    ),
                    inherited_home=(
                        inherited_home
                    ),
                )

    walk(obj)

    if not rows:
        return []

    frame = pd.DataFrame(
        rows
    )

    frame["player_id"] = pd.to_numeric(
        frame["player_id"],
        errors="coerce",
    )

    frame["avg_x"] = pd.to_numeric(
        frame["avg_x"],
        errors="coerce",
    )

    frame["avg_y"] = pd.to_numeric(
        frame["avg_y"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "player_id",
            "avg_x",
            "avg_y",
        ]
    )

    frame[
        "player_id"
    ] = frame[
        "player_id"
    ].astype(int)

    # De-duplicate recursive discoveries.
    frame = frame.drop_duplicates(
        subset=[
            "event_id",
            "player_id",
        ],
        keep="first",
    )

    return frame.to_dict(
        orient="records"
    )


def extract_average_positions(
    matches: pd.DataFrame,
):
    rows = []

    coverage = {
        "matches":
            len(matches),

        "average_positions_nonempty":
            0,
    }

    for i, match in (
        matches.iterrows()
    ):
        event_id = int(
            match["event_id"]
        )

        print(
            f"[{i + 1}/{len(matches)}] "
            f"event {event_id}",
            end=" ",
        )

        stats_payload = api_get(
            f"/events/{event_id}/stats/"
        )

        event_rows = (
            recursive_find_average_positions(
                stats_payload,
                event_id,
                match,
            )
        )

        if event_rows:
            coverage[
                "average_positions_nonempty"
            ] += 1

            rows.extend(
                event_rows
            )

        print(
            f"| avg-pos rows="
            f"{len(event_rows)}"
        )

    return (
        pd.DataFrame(
            rows
        ),
        coverage,
    )


# ---------------------------------------------------------------------
# Player-profile extraction
# ---------------------------------------------------------------------

def extract_player_profiles(
    average_positions: pd.DataFrame,
):
    player_ids = sorted(
        average_positions[
            "player_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print()
    print(
        f"Unique historical players "
        f"with average-position rows: "
        f"{len(player_ids)}"
    )

    raw_payloads = {}
    rows = []

    for i, player_id in (
        enumerate(
            player_ids,
            start=1,
        )
    ):
        print(
            f"[{i}/{len(player_ids)}] "
            f"player {player_id}",
            end=" ",
        )

        payload = api_get(
            f"/players/{player_id}/"
        )

        raw_payloads[
            str(player_id)
        ] = payload

        if not isinstance(
            payload,
            dict,
        ):
            print("| invalid payload")
            continue

        team = (
            payload.get("team")
            if isinstance(
                payload.get("team"),
                dict,
            )
            else {}
        )

        nationality = (
            payload.get("nationality")
            if isinstance(
                payload.get(
                    "nationality"
                ),
                dict,
            )
            else {}
        )

        rows.append(
            {
                "player_id":
                    player_id,

                "player_name":
                    payload.get("name")
                    or payload.get(
                        "player_name"
                    ),

                "position":
                    payload.get("position")
                    or payload.get(
                        "pos"
                    ),

                "date_of_birth":
                    payload.get(
                        "date_of_birth"
                    )
                    or payload.get("dob"),

                "height_cm":
                    payload.get(
                        "height_cm"
                    )
                    or payload.get("height"),

                "preferred_foot":
                    payload.get(
                        "preferred_foot"
                    )
                    or payload.get("foot"),

                "nationality":
                    nationality.get(
                        "name"
                    )
                    or payload.get(
                        "nationality_name"
                    ),

                "nationality_code":
                    nationality.get(
                        "code"
                    )
                    or payload.get(
                        "nationality_code"
                    ),

                "profile_team_id":
                    team.get("id")
                    or payload.get(
                        "team_id"
                    ),

                "profile_team_name":
                    team.get("name")
                    or payload.get(
                        "team_name"
                    ),
            }
        )

        print(
            f"| "
            f"{rows[-1]['player_name']} "
            f"| position="
            f"{rows[-1]['position']}"
        )

    players = pd.DataFrame(
        rows
    )

    PLAYERS_RAW_FILE.write_text(
        json.dumps(
            raw_payloads,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return players


# ---------------------------------------------------------------------
# Tactical inference
# ---------------------------------------------------------------------

def normalize_broad_position(
    value,
):
    if pd.isna(
        value
    ):
        return None

    return BROAD_POSITION_MAP.get(
        str(
            value
        )
        .strip()
        .upper()
    )


def infer_attack_direction(
    group: pd.DataFrame,
):
    defenders = group[
        group[
            "broad_position"
        ]
        == "DEF"
    ]

    forwards = group[
        group[
            "broad_position"
        ]
        == "FWD"
    ]

    defenders = defenders[
        defenders[
            "avg_x"
        ].notna()
    ]

    forwards = forwards[
        forwards[
            "avg_x"
        ].notna()
    ]

    if (
        not defenders.empty
        and not forwards.empty
    ):
        if (
            forwards[
                "avg_x"
            ].mean()
            > defenders[
                "avg_x"
            ].mean()
        ):
            return "right"

        return "left"

    return "right"


def classify_tactical_role(
    broad_position,
    x_attacking,
    y_attacking,
):
    if (
        pd.isna(
            broad_position
        )
        or pd.isna(
            x_attacking
        )
        or pd.isna(
            y_attacking
        )
    ):
        return None

    x = float(
        x_attacking
    )

    y = float(
        y_attacking
    )

    wide_distance = abs(
        y - 50.0
    )

    if broad_position == "GK":
        return "GK"

    if broad_position == "DEF":
        if (
            wide_distance
            >= DEFENDER_WIDE_DISTANCE
            and x
            >= FULLBACK_MIN_X
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
            x
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
    if (
        pd.isna(
            position_group
        )
        or pd.isna(
            y_attacking
        )
    ):
        return None

    if position_group in {
        "GK",
        "ST",
        "AM",
    }:
        return "CENTRAL"

    y = float(
        y_attacking
    )

    if (
        y
        < SIDE_LEFT_MAX
    ):
        return "RIGHT"

    if (
        y
        > SIDE_RIGHT_MIN
    ):
        return "LEFT"

    return "CENTRAL"


def build_tactical_position(
    position_group,
    position_side,
):
    if pd.isna(
        position_group
    ):
        return None

    if position_group == "GK":
        return "GK"

    if position_group == "ST":
        return "ST"

    if position_group == "AM":
        return "AM"

    if position_group == "FB/WB":
        if (
            position_side
            == "LEFT"
        ):
            return "LB/WB"

        if (
            position_side
            == "RIGHT"
        ):
            return "RB/WB"

        return "FB/WB"

    if position_group == "CB":
        if (
            position_side
            == "LEFT"
        ):
            return "LCB"

        if (
            position_side
            == "RIGHT"
        ):
            return "RCB"

        return "CB"

    if position_group == "DM/CM":
        if (
            position_side
            == "LEFT"
        ):
            return "LCM/DM"

        if (
            position_side
            == "RIGHT"
        ):
            return "RCM/DM"

        return "CM/DM"

    if position_group == "W":
        if (
            position_side
            == "LEFT"
        ):
            return "LW"

        if (
            position_side
            == "RIGHT"
        ):
            return "RW"

        return "W"

    return position_group


def build_position_history(
    average_positions: pd.DataFrame,
    players: pd.DataFrame,
):
    history = average_positions.merge(
        players[
            [
                "player_id",
                "player_name",
                "position",
            ]
        ],
        on="player_id",
        how="left",
        suffixes=(
            "_avg",
            "_profile",
        ),
        validate="many_to_one",
    )

    history[
        "player_name"
    ] = history[
        "player_name_avg"
    ].fillna(
        history[
            "player_name_profile"
        ]
    )

    history[
        "broad_position"
    ] = history[
        "position"
    ].apply(
        normalize_broad_position
    )

    history[
        "attack_direction"
    ] = None

    history[
        "x_attacking"
    ] = np.nan

    history[
        "y_attacking"
    ] = np.nan

    history[
        "position_group"
    ] = None

    history[
        "position_side"
    ] = None

    history[
        "tactical_position"
    ] = None

    for (
        event_id,
        team_id
    ), group in (
        history
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
    ):
        direction = (
            infer_attack_direction(
                group
            )
        )

        idx = group.index

        history.loc[
            idx,
            "attack_direction",
        ] = direction

        if (
            direction
            == "right"
        ):
            history.loc[
                idx,
                "x_attacking",
            ] = group[
                "avg_x"
            ]

            history.loc[
                idx,
                "y_attacking",
            ] = group[
                "avg_y"
            ]

        else:
            history.loc[
                idx,
                "x_attacking",
            ] = (
                100.0
                - group[
                    "avg_x"
                ]
            )

            history.loc[
                idx,
                "y_attacking",
            ] = (
                100.0
                - group[
                    "avg_y"
                ]
            )

    history[
        "position_group"
    ] = history.apply(
        lambda row:
            classify_tactical_role(
                row[
                    "broad_position"
                ],
                row[
                    "x_attacking"
                ],
                row[
                    "y_attacking"
                ],
            ),
        axis=1,
    )

    history[
        "position_side"
    ] = history.apply(
        lambda row:
            classify_position_side(
                row[
                    "position_group"
                ],
                row[
                    "y_attacking"
                ],
            ),
        axis=1,
    )

    history[
        "tactical_position"
    ] = history.apply(
        lambda row:
            build_tactical_position(
                row[
                    "position_group"
                ],
                row[
                    "position_side"
                ],
            ),
        axis=1,
    )

    return history


# ---------------------------------------------------------------------
# Role-share analysis
# ---------------------------------------------------------------------

def build_role_analysis(
    history: pd.DataFrame,
    salzburg_team_id: int,
):
    salzburg = history[
        history[
            "team_id"
        ]
        == salzburg_team_id
    ].copy()

    salzburg = salzburg[
        salzburg[
            "position_group"
        ].notna()
    ].copy()

    if salzburg.empty:
        raise RuntimeError(
            "No Salzburg tactical-position rows were produced."
        )

    role_summary = (
        salzburg[
            "position_group"
        ]
        .value_counts()
        .rename_axis(
            "position_group"
        )
        .reset_index(
            name=(
                "player_match_observations"
            )
        )
    )

    total = role_summary[
        "player_match_observations"
    ].sum()

    role_summary[
        "role_share_pct"
    ] = (
        role_summary[
            "player_match_observations"
        ]
        / total
        * 100.0
    ).round(
        1
    )

    match_role_counts = (
        salzburg
        .groupby(
            [
                "event_id",
                "position_group",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for role in (
        POSITION_ORDER
    ):
        if (
            role
            not in match_role_counts.columns
        ):
            match_role_counts[
                role
            ] = 0

    average_counts = pd.DataFrame(
        [
            {
                "position_group":
                    role,

                "average_players_per_match":
                    round(
                        match_role_counts[
                            role
                        ].mean(),
                        2,
                    ),

                "median_players_per_match":
                    round(
                        match_role_counts[
                            role
                        ].median(),
                        2,
                    ),
            }
            for role in (
                POSITION_ORDER
            )
        ]
    )

    role_summary = role_summary.merge(
        average_counts,
        on="position_group",
        how="outer",
    )

    order_map = {
        role: i
        for i, role in (
            enumerate(
                POSITION_ORDER
            )
        )
    }

    role_summary[
        "_order"
    ] = role_summary[
        "position_group"
    ].map(
        order_map
    )

    role_summary = (
        role_summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        salzburg,
        role_summary,
        match_role_counts,
    )


def print_player_role_summary(
    salzburg_history: pd.DataFrame,
):
    print()
    print("=" * 100)
    print(
        "SALZBURG PLAYER ROLE DISTRIBUTION"
        .center(100)
    )
    print("=" * 100)

    summary = (
        salzburg_history
        .groupby(
            [
                "player_id",
                "player_name",
            ],
            dropna=False,
        )
        ["position_group"]
        .value_counts()
        .rename(
            "count"
        )
        .reset_index()
    )

    if summary.empty:
        print(
            "No player-role rows."
        )
        return

    totals = (
        summary
        .groupby(
            [
                "player_id",
                "player_name",
            ],
            dropna=False,
        )
        ["count"]
        .sum()
        .rename(
            "total"
        )
        .reset_index()
    )

    summary = summary.merge(
        totals,
        on=[
            "player_id",
            "player_name",
        ],
        how="left",
    )

    summary[
        "share_pct"
    ] = (
        summary[
            "count"
        ]
        / summary[
            "total"
        ]
        * 100.0
    ).round(
        1
    )

    summary = (
        summary
        .sort_values(
            [
                "total",
                "player_name",
                "count",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 100)
    print(
        "RB SALZBURG 2022/23 — JAISSLE ROLE VALIDATION"
        .center(100)
    )
    print("=" * 100)

    season = find_season()

    season_id = int(
        season[
            "id"
        ]
    )

    team = find_team(
        season_id
    )

    team_id = int(
        team[
            "id"
        ]
    )

    matches = extract_matches(
        season_id,
        team_id,
    )

    if matches.empty:
        raise RuntimeError(
            "No finished Salzburg league matches returned."
        )

    matches.to_parquet(
        MATCHES_FILE,
        index=False,
    )

    print(
        f"\nFinished Salzburg league matches: "
        f"{len(matches)}"
    )

    (
        average_positions,
        coverage,
    ) = extract_average_positions(
        matches
    )

    if average_positions.empty:
        raise RuntimeError(
            "No historical Salzburg average-position rows were returned."
        )

    average_positions.to_parquet(
        AVERAGE_POSITIONS_FILE,
        index=False,
    )

    covered_event_ids = sorted(
        average_positions[
            "event_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print()
    print("=" * 100)
    print(
        "HISTORICAL COVERAGE CHECK"
        .center(100)
    )
    print("=" * 100)

    print(
        f"league matches: "
        f"{len(matches)}"
    )

    print(
        f"matches with average positions: "
        f"{len(covered_event_ids)}"
    )

    print(
        "average-position coverage: "
        f"{len(covered_event_ids) / len(matches):.1%}"
    )

    players = (
        extract_player_profiles(
            average_positions
        )
    )

    if players.empty:
        raise RuntimeError(
            "No player profiles were returned."
        )

    players.to_parquet(
        PLAYERS_FILE,
        index=False,
    )

    position_coverage = (
        players[
            "position"
        ].notna().mean()
    )

    print()
    print(
        "Player-profile broad-position "
        f"coverage: {position_coverage:.1%}"
    )

    history = (
        build_position_history(
            average_positions,
            players,
        )
    )

    unknown_positions = history[
        history[
            "broad_position"
        ].isna()
    ]

    if (
        not unknown_positions.empty
    ):
        print()
        print(
            "WARNING: historical rows missing "
            "usable profile broad position:"
        )

        print(
            unknown_positions[
                [
                    "player_id",
                    "player_name",
                    "position",
                ]
            ]
            .drop_duplicates()
            .to_string(
                index=False
            )
        )

    (
        salzburg_history,
        role_summary,
        match_role_counts,
    ) = build_role_analysis(
        history,
        team_id,
    )

    salzburg_history.to_parquet(
        POSITION_HISTORY_FILE,
        index=False,
    )

    role_summary.to_parquet(
        ROLE_SUMMARY_FILE,
        index=False,
    )

    match_role_counts.to_parquet(
        MATCH_ROLE_COUNTS_FILE,
        index=False,
    )

    metadata = {
        "league_id":
            LEAGUE_ID,

        "season_id":
            season_id,

        "season_name":
            season.get(
                "name"
            ),

        "team_id":
            team_id,

        "team_name":
            team.get(
                "name"
            ),

        "finished_matches":
            len(matches),

        "matches_with_average_positions":
            len(
                covered_event_ids
            ),

        "average_position_coverage_pct":
            round(
                (
                    len(
                        covered_event_ids
                    )
                    / len(
                        matches
                    )
                    * 100.0
                ),
                1,
            ),

        "player_profile_position_coverage_pct":
            round(
                position_coverage
                * 100.0,
                1,
            ),

        "thresholds": {
            "defender_wide_distance":
                DEFENDER_WIDE_DISTANCE,

            "fullback_min_x":
                FULLBACK_MIN_X,

            "midfielder_wide_distance":
                MIDFIELDER_WIDE_DISTANCE,

            "forward_wide_distance":
                FORWARD_WIDE_DISTANCE,

            "attacking_mid_x":
                ATTACKING_MID_X,

            "side_left_max":
                SIDE_LEFT_MAX,

            "side_right_min":
                SIDE_RIGHT_MIN,
        },
    }

    RAW_META_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "SALZBURG 2022/23 ROLE DISTRIBUTION"
        .center(100)
    )
    print("=" * 100)

    print(
        role_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "MATCH-BY-MATCH ROLE COUNTS"
        .center(100)
    )
    print("=" * 100)

    display_columns = (
        [
            "event_id",
        ]
        + POSITION_ORDER
    )

    print(
        match_role_counts[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print_player_role_summary(
        salzburg_history
    )

    print()
    print("=" * 100)
    print(
        "SAVED"
        .center(100)
    )
    print("=" * 100)

    for path in [
        MATCHES_FILE,
        AVERAGE_POSITIONS_FILE,
        PLAYERS_FILE,
        PLAYERS_RAW_FILE,
        POSITION_HISTORY_FILE,
        ROLE_SUMMARY_FILE,
        MATCH_ROLE_COUNTS_FILE,
        RAW_META_FILE,
    ]:
        print(
            path
        )


if __name__ == "__main__":
    main()
