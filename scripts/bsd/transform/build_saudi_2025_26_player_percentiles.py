from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("BSD_API_KEY")
if not API_KEY:
    raise RuntimeError("BSD_API_KEY not found in .env")

BASE_URL = "https://sports.bzzoiro.com/api/v2"
HEADERS = {"Authorization": f"Token {API_KEY}"}
REQUEST_SLEEP_SECONDS = 0.08

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd" / "saudi_2025_26"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_FILE = RAW_DIR / "matches.parquet"
POSITION_HISTORY_FILE = RAW_DIR / "player_position_history.parquet"
PLAYER_MATCH_FILE = RAW_DIR / "player_match_stats.parquet"

MINUTES_FILE = RAW_DIR / "player_minutes_v3.parquet"
MINUTES_RAW_DIR = RAW_DIR / "minute_source_raw"
MINUTES_RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASON_METRICS_FILE = (
    PROCESSED_DIR / "saudi_2025_26_player_season_metrics.parquet"
)
PERCENTILES_FILE = (
    PROCESSED_DIR / "saudi_2025_26_player_percentiles.parquet"
)

MIN_MINUTES = 270

POSITION_ORDER = [
    "GK",
    "CB",
    "FB/WB",
    "DM/CM",
    "AM",
    "W",
    "ST",
]

ALIASES = {
    "minutes": ["minutes", "mins", "min"],
    "goals": ["goals", "goal"],
    "assists": ["assists", "assist"],
    "xg": ["xg", "expected_goals"],
    "xa": ["xa", "expected_assists"],
    "shots": ["shots", "total_shots"],
    "shots_on_target": ["shots_on_target", "sot"],
    "key_passes": ["key_passes", "key_pass"],
    "touches": ["touches"],
    "successful_passes": [
        "successful_passes",
        "accurate_passes",
        "passes_completed",
    ],
    "total_passes": [
        "total_passes",
        "passes",
        "pass_attempts",
    ],
    "accurate_opposition_half_passes": [
        "accurate_opposition_half_passes",
        "successful_opposition_half_passes",
    ],
    "total_long_balls": [
        "total_long_balls",
        "long_balls",
    ],
    "accurate_long_balls": [
        "accurate_long_balls",
        "successful_long_balls",
    ],
    "total_crosses": [
        "total_crosses",
        "crosses",
    ],
    "accurate_crosses": [
        "accurate_crosses",
        "successful_crosses",
    ],
    "progressive_carries": [
        "progressive_carries",
    ],
    "total_progressive_ball_carries_distance": [
        "total_progressive_ball_carries_distance",
        "progressive_carry_distance",
        "progressive_carries_distance",
    ],
    "recoveries": [
        "recoveries",
        "ball_recoveries",
    ],
    "interceptions": [
        "interceptions",
    ],
    "duels_won": [
        "duels_won",
        "won_duels",
    ],
    "total_duels": [
        "total_duels",
        "duels",
    ],
    "aerials_won": [
        "aerials_won",
        "won_aerials",
    ],
    "total_aerials": [
        "total_aerials",
        "aerials",
    ],
    "tackles_won": [
        "tackles_won",
        "successful_tackles",
    ],
    "total_tackles": [
        "total_tackles",
        "tackles",
    ],
    "possession_lost": [
        "possession_lost",
        "poss_lost",
    ],
}


# ============================================================
# API
# ============================================================

def api_get(path: str):
    r = requests.get(
        f"{BASE_URL}{path}",
        headers=HEADERS,
        timeout=30,
    )

    if not r.ok:
        raise RuntimeError(
            f"BSD request failed {r.status_code}\n"
            f"{r.url}\n"
            f"{r.text[:1000]}"
        )

    time.sleep(REQUEST_SLEEP_SECONDS)
    return r.json()


def try_api_get(path: str):
    try:
        return api_get(path)
    except Exception:
        return None


# ============================================================
# MINUTE RECONSTRUCTION — BSD HISTORICAL PAYLOAD SHAPE
# ============================================================

def _to_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if "+" in value:
            base, extra = value.split("+", 1)
            try:
                return int(float(base)) + int(float(extra))
            except ValueError:
                return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _effective_incident_minute(incident: dict):
    minute = _to_int(incident.get("minute"))
    if minute is None:
        return None

    added_time = _to_int(incident.get("added_time"))

    if (
        added_time is not None
        and added_time > 0
        and minute in {45, 90, 105, 120}
    ):
        return minute + added_time

    return minute


def _extract_bsd_lineups(payload: dict):
    lineups_payload = None

    for key, value in payload.items():
        if key.endswith("/lineups/"):
            lineups_payload = value
            break

    if not isinstance(lineups_payload, dict):
        return [], []

    lineups = lineups_payload.get("lineups", {})
    if not isinstance(lineups, dict):
        return [], []

    starters = []
    substitutes = []

    for side in ["home", "away"]:
        side_data = lineups.get(side, {})
        if not isinstance(side_data, dict):
            continue

        team_id = _to_int(side_data.get("team_id"))

        for player in side_data.get("players", []) or []:
            if not isinstance(player, dict):
                continue

            player_id = _to_int(player.get("id"))
            if player_id is None:
                continue

            starters.append(
                {
                    "player_id": player_id,
                    "team_id": team_id,
                    "player_name": player.get("name"),
                    "side": side,
                }
            )

        for player in side_data.get("substitutes", []) or []:
            if not isinstance(player, dict):
                continue

            player_id = _to_int(player.get("id"))
            if player_id is None:
                continue

            substitutes.append(
                {
                    "player_id": player_id,
                    "team_id": team_id,
                    "player_name": player.get("name"),
                    "side": side,
                }
            )

    return starters, substitutes


def _extract_bsd_substitutions(payload: dict):
    incidents_payload = None

    for key, value in payload.items():
        if key.endswith("/incidents/"):
            incidents_payload = value
            break

    if not isinstance(incidents_payload, dict):
        return []

    incidents = incidents_payload.get("incidents", []) or []
    substitutions = []

    for incident in incidents:
        if not isinstance(incident, dict):
            continue

        if str(incident.get("type", "")).lower() != "substitution":
            continue

        minute = _effective_incident_minute(incident)
        if minute is None:
            continue

        player_in_id = _to_int(incident.get("player_in_id"))
        player_out_id = _to_int(incident.get("player_out_id"))

        if player_in_id is None and player_out_id is None:
            continue

        substitutions.append(
            {
                "player_in_id": player_in_id,
                "player_out_id": player_out_id,
                "minute": minute,
                "is_home": incident.get("is_home"),
            }
        )

    return substitutions


def _extract_bsd_match_length(payload: dict):
    candidates = [90]

    event_payload = None
    for key, value in payload.items():
        if (
            key.startswith("/events/")
            and key.endswith("/")
            and "/lineups/" not in key
            and "/incidents/" not in key
            and "/timeline/" not in key
            and isinstance(value, dict)
        ):
            event_payload = value
            break

    if isinstance(event_payload, dict):
        current_minute = _to_int(event_payload.get("current_minute"))
        if current_minute is not None:
            candidates.append(current_minute)

    incidents_payload = None
    for key, value in payload.items():
        if key.endswith("/incidents/"):
            incidents_payload = value
            break

    if isinstance(incidents_payload, dict):
        for incident in incidents_payload.get("incidents", []) or []:
            if not isinstance(incident, dict):
                continue

            minute = _effective_incident_minute(incident)
            if minute is not None:
                candidates.append(minute)

            if str(incident.get("type", "")).lower() == "injurytime":
                base = _to_int(incident.get("minute"))
                length = _to_int(incident.get("length"))
                if base is not None and length is not None:
                    candidates.append(base + length)

    match_length = max(candidates)

    if match_length < 90 or match_length > 130:
        return 90

    return match_length


def get_minute_source_payload(event_id: int):
    cache_file = MINUTES_RAW_DIR / f"{event_id}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    payload = {}

    for endpoint in [
        f"/events/{event_id}/",
        f"/events/{event_id}/lineups/",
        f"/events/{event_id}/incidents/",
    ]:
        result = try_api_get(endpoint)
        if result is not None:
            payload[endpoint] = result

    cache_file.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    return payload


def build_minutes_from_lineups_and_incidents(
    matches: pd.DataFrame,
    player_match: pd.DataFrame,
):
    """
    Reconstruct minutes TEAM-BY-TEAM.

    Important BSD historical behavior:
    player_match_stats contains squad/bench rows, not only players
    who actually appeared.

    Therefore:
      1. Identify actual participants from positive match-stat activity.
      2. Players who appeared AND were not subbed on are inferred starters.
      3. Accept the team-match only when that produces exactly 11 starters.
      4. Fall back to a complete 11-player lineup when BSD provides it.
    """

    rows = []

    stats = player_match.copy()

    for column in ["event_id", "player_id", "team_id"]:
        stats[column] = pd.to_numeric(
            stats[column],
            errors="coerce",
        )

    stats = stats.dropna(
        subset=[
            "event_id",
            "player_id",
            "team_id",
        ]
    ).copy()

    # Metrics that prove a player actually participated.
    # touches is the strongest signal; the rest protect against rare
    # rows where touches is unavailable but another event stat exists.
    activity_columns = [
        "touches",
        "shots",
        "shots_on_target",
        "goals",
        "assists",
        "xg",
        "xa",
        "key_passes",
        "successful_passes",
        "total_passes",
        "accurate_opposition_half_passes",
        "total_long_balls",
        "accurate_long_balls",
        "total_crosses",
        "accurate_crosses",
        "progressive_carries",
        "recoveries",
        "interceptions",
        "duels_won",
        "total_duels",
        "aerials_won",
        "total_aerials",
        "tackles_won",
        "total_tackles",
        "possession_lost",
    ]

    activity_columns = [
        col
        for col in activity_columns
        if col in stats.columns
    ]

    for col in activity_columns:
        stats[col] = pd.to_numeric(
            stats[col],
            errors="coerce",
        )

    if "touches" in stats.columns:
        stats["appeared"] = (
            stats["touches"]
            .fillna(0)
            > 0
        )
    else:
        stats["appeared"] = False

    # Fallback: any positive event statistic means the player appeared.
    if activity_columns:
        positive_activity = (
            stats[
                activity_columns
            ]
            .fillna(0)
            .gt(0)
            .any(axis=1)
        )

        stats["appeared"] = (
            stats["appeared"]
            | positive_activity
        )

    valid_team_matches = 0
    skipped_team_matches = 0

    for i, match in matches.iterrows():
        event_id = int(
            match["event_id"]
        )

        event_stats = stats[
            stats["event_id"]
            == event_id
        ].copy()

        if event_stats.empty:
            continue

        payload = get_minute_source_payload(
            event_id
        )

        lineup_starters, _ = (
            _extract_bsd_lineups(
                payload
            )
        )

        substitutions = (
            _extract_bsd_substitutions(
                payload
            )
        )

        match_length = (
            _extract_bsd_match_length(
                payload
            )
        )

        home_team_id = _to_int(
            match.get(
                "home_team_id"
            )
        )

        away_team_id = _to_int(
            match.get(
                "away_team_id"
            )
        )

        subs_by_team = {}

        for sub in substitutions:
            if sub.get("is_home") is True:
                team_id = home_team_id
            elif sub.get("is_home") is False:
                team_id = away_team_id
            else:
                team_id = None

            if team_id is None:
                continue

            subs_by_team.setdefault(
                team_id,
                [],
            ).append(sub)

        lineup_starters_by_team = {}

        for player in lineup_starters:
            team_id = _to_int(
                player.get(
                    "team_id"
                )
            )

            player_id = _to_int(
                player.get(
                    "player_id"
                )
            )

            if (
                team_id is None
                or player_id is None
            ):
                continue

            lineup_starters_by_team.setdefault(
                team_id,
                set(),
            ).add(player_id)

        event_rows = 0
        event_valid_teams = 0

        for team_id, team_stats in event_stats.groupby(
            "team_id"
        ):
            team_id = int(
                team_id
            )

            all_stat_players = set(
                team_stats[
                    "player_id"
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            active_players = set(
                team_stats.loc[
                    team_stats[
                        "appeared"
                    ],
                    "player_id",
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            team_subs = subs_by_team.get(
                team_id,
                [],
            )

            subbed_in_ids = {
                int(
                    sub[
                        "player_in_id"
                    ]
                )
                for sub in team_subs
                if sub.get(
                    "player_in_id"
                )
                is not None
            }

            # A player listed as coming on definitely appeared, even if
            # the statistical row happens to show no positive activity.
            active_players |= subbed_in_ids

            inferred_starters = (
                active_players
                - subbed_in_ids
            )

            known_lineup_starters = (
                lineup_starters_by_team.get(
                    team_id,
                    set(),
                )
            )

            if len(inferred_starters) == 11:
                starters = set(
                    inferred_starters
                )

                starter_source = (
                    "active-stats-minus-sub-ins"
                )

            elif len(
                known_lineup_starters
            ) == 11:
                starters = set(
                    known_lineup_starters
                )

                starter_source = (
                    "lineup"
                )

                # Full lineup lets us safely include starters even if their
                # activity row is sparse.
                active_players |= starters

            else:
                skipped_team_matches += 1

                print(
                    f"[{i+1}/{len(matches)}] "
                    f"event {event_id} team {team_id} "
                    f"| squad rows={len(all_stat_players)} "
                    f"| active={len(active_players)} "
                    f"| sub-ins={len(subbed_in_ids)} "
                    f"| inferred starters={len(inferred_starters)} "
                    f"| lineup starters={len(known_lineup_starters)} "
                    f"| skipped"
                )

                continue

            state = {}

            for player_id in starters:
                state[
                    player_id
                ] = {
                    "event_id":
                        event_id,

                    "player_id":
                        player_id,

                    "team_id":
                        team_id,

                    "starter":
                        True,

                    "start_minute":
                        0,

                    "end_minute":
                        match_length,
                }

            # Only actual participants belong in the minute table.
            for player_id in active_players:
                if player_id in state:
                    continue

                state[
                    player_id
                ] = {
                    "event_id":
                        event_id,

                    "player_id":
                        player_id,

                    "team_id":
                        team_id,

                    "starter":
                        False,

                    "start_minute":
                        None,

                    "end_minute":
                        None,
                }

            for sub in sorted(
                team_subs,
                key=lambda x: x[
                    "minute"
                ],
            ):
                minute = min(
                    int(
                        sub[
                            "minute"
                        ]
                    ),
                    match_length,
                )

                out_id = sub.get(
                    "player_out_id"
                )

                in_id = sub.get(
                    "player_in_id"
                )

                if out_id is not None:
                    out_id = int(
                        out_id
                    )

                    if out_id not in state:
                        # A recorded player-off means they were on the pitch.
                        state[
                            out_id
                        ] = {
                            "event_id":
                                event_id,

                            "player_id":
                                out_id,

                            "team_id":
                                team_id,

                            "starter":
                                True,

                            "start_minute":
                                0,

                            "end_minute":
                                match_length,
                        }

                    if (
                        state[
                            out_id
                        ][
                            "start_minute"
                        ]
                        is None
                    ):
                        state[
                            out_id
                        ][
                            "start_minute"
                        ] = 0

                    state[
                        out_id
                    ][
                        "end_minute"
                    ] = minute

                if in_id is not None:
                    in_id = int(
                        in_id
                    )

                    if in_id not in state:
                        state[
                            in_id
                        ] = {
                            "event_id":
                                event_id,

                            "player_id":
                                in_id,

                            "team_id":
                                team_id,

                            "starter":
                                False,

                            "start_minute":
                                minute,

                            "end_minute":
                                match_length,
                        }

                    else:
                        state[
                            in_id
                        ][
                            "starter"
                        ] = False

                        state[
                            in_id
                        ][
                            "start_minute"
                        ] = minute

                        state[
                            in_id
                        ][
                            "end_minute"
                        ] = match_length

            team_rows = 0

            for player in state.values():
                start_minute = player[
                    "start_minute"
                ]

                end_minute = player[
                    "end_minute"
                ]

                if (
                    start_minute is None
                    or end_minute is None
                ):
                    continue

                minutes = max(
                    0,
                    min(
                        match_length,
                        end_minute,
                    )
                    - max(
                        0,
                        start_minute,
                    ),
                )

                if minutes <= 0:
                    continue

                rows.append(
                    {
                        "event_id":
                            event_id,

                        "player_id":
                            player[
                                "player_id"
                            ],

                        "team_id":
                            team_id,

                        "starter":
                            player[
                                "starter"
                            ],

                        "start_minute":
                            start_minute,

                        "end_minute":
                            end_minute,

                        "match_length":
                            match_length,

                        "minutes":
                            minutes,

                        "starter_source":
                            starter_source,
                    }
                )

                team_rows += 1
                event_rows += 1

            valid_team_matches += 1
            event_valid_teams += 1

        print(
            f"[{i+1}/{len(matches)}] "
            f"event {event_id} "
            f"| valid teams={event_valid_teams}/2 "
            f"| subs={len(substitutions)} "
            f"| length={match_length} "
            f"| minute rows={event_rows}"
        )

    result = pd.DataFrame(
        rows
    )

    print()
    print(
        "Valid team-matches reconstructed: "
        f"{valid_team_matches}"
    )

    print(
        "Skipped team-matches: "
        f"{skipped_team_matches}"
    )

    return result


# ============================================================
# ROLE / TEAM HELPERS
# ============================================================

def choose_primary_position(
    position_history: pd.DataFrame,
):
    data = position_history[
        position_history[
            "position_group"
        ].notna()
    ].copy()

    counts = (
        data
        .groupby(
            [
                "player_id",
                "position_group",
            ]
        )
        .size()
        .rename(
            "position_matches"
        )
        .reset_index()
    )

    totals = (
        counts
        .groupby(
            "player_id"
        )[
            "position_matches"
        ]
        .sum()
        .rename(
            "position_matches_observed"
        )
        .reset_index()
    )

    counts = counts.merge(
        totals,
        on="player_id",
        how="left",
    )

    counts[
        "position_group_share"
    ] = (
        counts[
            "position_matches"
        ]
        / counts[
            "position_matches_observed"
        ]
    )

    order = {
        role: i
        for i, role
        in enumerate(
            POSITION_ORDER
        )
    }

    counts[
        "_order"
    ] = (
        counts[
            "position_group"
        ]
        .map(order)
        .fillna(999)
    )

    primary = (
        counts
        .sort_values(
            [
                "player_id",
                "position_matches",
                "_order",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "player_id"
        )
        [
            [
                "player_id",
                "position_group",
                "position_matches_observed",
                "position_group_share",
            ]
        ]
    )

    return primary


def add_team_names(
    season: pd.DataFrame,
    matches: pd.DataFrame,
):
    home = (
        matches[
            [
                "home_team_id",
                "home_team",
            ]
        ]
        .rename(
            columns={
                "home_team_id":
                    "team_id",

                "home_team":
                    "team_name",
            }
        )
    )

    away = (
        matches[
            [
                "away_team_id",
                "away_team",
            ]
        ]
        .rename(
            columns={
                "away_team_id":
                    "team_id",

                "away_team":
                    "team_name",
            }
        )
    )

    lookup = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    lookup[
        "team_id"
    ] = pd.to_numeric(
        lookup[
            "team_id"
        ],
        errors="coerce",
    )

    lookup = (
        lookup
        .dropna(
            subset=[
                "team_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "team_id",
            ],
            keep="last",
        )
    )

    season[
        "team_id"
    ] = pd.to_numeric(
        season[
            "team_id"
        ],
        errors="coerce",
    )

    return season.merge(
        lookup,
        on="team_id",
        how="left",
    )


# ============================================================
# SEASON METRICS
# ============================================================

def build_season_metrics(
    player_match: pd.DataFrame,
    position_history: pd.DataFrame,
    matches: pd.DataFrame,
):
    pm = player_match.copy()

    for col in [
        "player_id",
        "event_id",
        "team_id",
        "minutes",
    ]:
        pm[col] = pd.to_numeric(
            pm[col],
            errors="coerce",
        )

    pm = pm.dropna(
        subset=[
            "player_id",
            "event_id",
            "team_id",
            "minutes",
        ]
    ).copy()

    pm["player_id"] = (
        pm["player_id"]
        .astype(int)
    )

    pm["event_id"] = (
        pm["event_id"]
        .astype(int)
    )

    pm["team_id"] = (
        pm["team_id"]
        .astype(int)
    )

    pm = pm[
        pm[
            "minutes"
        ] > 0
    ].copy()

    print()
    print(
        "Player-match team ID coverage: "
        f"{pm['team_id'].notna().mean():.1%}"
    )

    print(
        "Unique teams in player-match stats: "
        f"{pm['team_id'].nunique()}"
    )

    print(
        "Player-match minute coverage: "
        f"{pm['minutes'].notna().mean():.1%}"
    )

    # Primary team by reconstructed minutes.
    team_minutes = (
        pm
        .groupby(
            [
                "player_id",
                "team_id",
            ]
        )[
            "minutes"
        ]
        .sum()
        .reset_index()
    )

    primary_team = (
        team_minutes
        .sort_values(
            [
                "player_id",
                "minutes",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            "player_id"
        )
        [
            [
                "player_id",
                "team_id",
            ]
        ]
    )

    # Names: historical player stats often have null names.
    names_pm = (
        pm[
            [
                "player_id",
                "player_name",
            ]
        ]
        .dropna(
            subset=[
                "player_name",
            ]
        )
        .drop_duplicates(
            "player_id",
            keep="last",
        )
    )

    names_ph = (
        position_history[
            [
                "player_id",
                "player_name",
            ]
        ]
        .dropna(
            subset=[
                "player_name",
            ]
        )
        .drop_duplicates(
            "player_id",
            keep="last",
        )
        .rename(
            columns={
                "player_name":
                    "history_player_name"
            }
        )
    )

    positions = choose_primary_position(
        position_history
    )

    # Only aggregate genuinely populated stats.
    sum_cols = []

    for col in ALIASES:
        if (
            col != "minutes"
            and col in pm.columns
            and pm[col].notna().any()
        ):
            pm[col] = pd.to_numeric(
                pm[col],
                errors="coerce",
            )

            sum_cols.append(col)

    agg_dict = {
        "minutes": "sum",
    }

    for col in sum_cols:
        agg_dict[col] = "sum"

    season = (
        pm
        .groupby(
            "player_id"
        )
        .agg(
            agg_dict
        )
        .reset_index()
    )

    season = (
        season
        .merge(
            primary_team,
            on="player_id",
            how="left",
        )
        .merge(
            names_pm,
            on="player_id",
            how="left",
        )
        .merge(
            names_ph,
            on="player_id",
            how="left",
        )
        .merge(
            positions,
            on="player_id",
            how="left",
        )
    )

    if "player_name" not in season.columns:
        season[
            "player_name"
        ] = season[
            "history_player_name"
        ]
    else:
        season[
            "player_name"
        ] = season[
            "player_name"
        ].fillna(
            season[
                "history_player_name"
            ]
        )

    season = season.drop(
        columns=[
            "history_player_name",
        ],
        errors="ignore",
    )

    season = add_team_names(
        season,
        matches,
    )

    per90_base = [
        "goals",
        "assists",
        "xg",
        "xa",
        "shots",
        "shots_on_target",
        "key_passes",
        "touches",
        "successful_passes",
        "total_passes",
        "accurate_opposition_half_passes",
        "total_long_balls",
        "accurate_long_balls",
        "total_crosses",
        "accurate_crosses",
        "progressive_carries",
        "total_progressive_ball_carries_distance",
        "recoveries",
        "interceptions",
        "duels_won",
        "total_duels",
        "aerials_won",
        "total_aerials",
        "tackles_won",
        "total_tackles",
        "possession_lost",
    ]

    for col in per90_base:
        if (
            col in season.columns
            and season[
                col
            ].notna().any()
        ):
            season[
                f"{col}_per90"
            ] = (
                season[
                    col
                ]
                / season[
                    "minutes"
                ]
                * 90.0
            )

    if {
        "successful_passes",
        "total_passes",
    }.issubset(
        season.columns
    ):
        season[
            "pass_accuracy_pct"
        ] = np.where(
            season[
                "total_passes"
            ] > 0,
            season[
                "successful_passes"
            ]
            / season[
                "total_passes"
            ]
            * 100.0,
            np.nan,
        )

    if {
        "accurate_long_balls",
        "total_long_balls",
    }.issubset(
        season.columns
    ):
        season[
            "long_ball_accuracy_pct"
        ] = np.where(
            season[
                "total_long_balls"
            ] > 0,
            season[
                "accurate_long_balls"
            ]
            / season[
                "total_long_balls"
            ]
            * 100.0,
            np.nan,
        )

    if {
        "accurate_crosses",
        "total_crosses",
    }.issubset(
        season.columns
    ):
        season[
            "cross_accuracy_pct"
        ] = np.where(
            season[
                "total_crosses"
            ] > 0,
            season[
                "accurate_crosses"
            ]
            / season[
                "total_crosses"
            ]
            * 100.0,
            np.nan,
        )

    print()
    print("Season team coverage:")

    print(
        season[
            [
                "team_id",
                "team_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "team_id"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Season players built: "
        f"{len(season)}"
    )

    print(
        "Players with position group: "
        f"{season['position_group'].notna().sum()}"
    )

    return season


# ============================================================
# PERCENTILES
# ============================================================

def add_percentiles(
    season: pd.DataFrame,
):
    eligible = season[
        (
            season[
                "minutes"
            ]
            >= MIN_MINUTES
        )
        & season[
            "position_group"
        ].notna()
    ].copy()

    pct_metrics = [
        "goals_per90",
        "assists_per90",
        "xg_per90",
        "xa_per90",
        "shots_per90",
        "shots_on_target_per90",
        "key_passes_per90",
        "touches_per90",
        "successful_passes_per90",
        "total_passes_per90",
        "accurate_opposition_half_passes_per90",
        "total_long_balls_per90",
        "accurate_long_balls_per90",
        "total_crosses_per90",
        "accurate_crosses_per90",
        "progressive_carries_per90",
        "total_progressive_ball_carries_distance_per90",
        "recoveries_per90",
        "interceptions_per90",
        "duels_won_per90",
        "aerials_won_per90",
        "tackles_won_per90",
        "possession_lost_per90",
        "pass_accuracy_pct",
        "long_ball_accuracy_pct",
        "cross_accuracy_pct",
    ]

    pct_metrics = [
        metric
        for metric
        in pct_metrics
        if (
            metric
            in eligible.columns
            and eligible[
                metric
            ].notna().any()
        )
    ]

    for metric in pct_metrics:
        eligible[
            f"{metric}_percentile"
        ] = (
            eligible
            .groupby(
                "position_group"
            )[
                metric
            ]
            .rank(
                pct=True,
                method="average",
            )
            * 100.0
        )

    return eligible


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 104)
    print(
        "SAUDI PRO LEAGUE 2025/26 — PLAYER STATS + POSITION-GROUP PERCENTILES"
        .center(104)
    )
    print("=" * 104)

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    position_history = (
        pd.read_parquet(
            POSITION_HISTORY_FILE
        )
    )

    player_match = pd.read_parquet(
        PLAYER_MATCH_FILE
    )

    print(
        f"\nUsing cached player-match stats:\n"
        f"{PLAYER_MATCH_FILE}"
    )

    print(
        f"\nPlayer-match rows: "
        f"{len(player_match):,}"
    )

    print(
        f"Unique players: "
        f"{player_match['player_id'].nunique():,}"
    )

    available = [
        col
        for col
        in ALIASES
        if (
            col in player_match.columns
            and player_match[
                col
            ].notna().any()
        )
    ]

    print(
        "\nAvailable canonical stat columns:"
    )

    for col in available:
        print(
            f"  {col}"
        )

    # --------------------------------------------------------
    # Minutes
    # --------------------------------------------------------

    if MINUTES_FILE.exists():
        print(
            f"\nUsing cached reconstructed minutes:\n"
            f"{MINUTES_FILE}"
        )

        player_minutes = pd.read_parquet(
            MINUTES_FILE
        )

    else:
        print(
            "\nNo reconstructed minute cache found."
        )

        print(
            "Pulling/caching match detail data and "
            "reconstructing minutes..."
        )

        player_minutes = (
            build_minutes_from_lineups_and_incidents(
                matches,
                player_match,
            )
        )

        if player_minutes.empty:
            raise RuntimeError(
                "\nNo player minutes could be reconstructed.\n"
                "Inspect one cached JSON file in:\n"
                f"{MINUTES_RAW_DIR}\n"
                "The BSD historical lineup/timeline payload "
                "shape differs from the generic parser."
            )

        player_minutes.to_parquet(
            MINUTES_FILE,
            index=False,
        )

    # Use reconstructed minutes, not the null minutes field
    # from historical player stats.
    player_match = (
        player_match
        .drop(
            columns=[
                "minutes",
            ],
            errors="ignore",
        )
        .merge(
            player_minutes[
                [
                    "event_id",
                    "player_id",
                    "minutes",
                ]
            ],
            on=[
                "event_id",
                "player_id",
            ],
            how="left",
        )
    )

    minute_coverage = (
        player_match[
            "minutes"
        ]
        .notna()
        .mean()
    )

    positive_minutes = (
        player_match[
            "minutes"
        ]
        .fillna(0)
        > 0
    ).sum()

    print()
    print(
        "Reconstructed minute coverage: "
        f"{minute_coverage:.1%}"
    )

    print(
        "Rows with positive minutes: "
        f"{positive_minutes:,}"
    )

    if minute_coverage < 0.35:
        raise RuntimeError(
            "\nReconstructed minute coverage is below 35%.\n"
            "Do not build per-90 statistics from this cache yet.\n"
            f"Inspect cached payloads in:\n{MINUTES_RAW_DIR}"
        )
        
    print()
    print(
        "WARNING: Saudi historical per-90 metrics are based only on "
        "team-matches with reconstructable minutes."
    )

    print(
        f"Validated minute coverage: {minute_coverage:.1%}"
    )

    # --------------------------------------------------------
    # Season build
    # --------------------------------------------------------

    season = build_season_metrics(
        player_match,
        position_history,
        matches,
    )

    season.to_parquet(
        SEASON_METRICS_FILE,
        index=False,
    )

    percentiles = add_percentiles(
        season
    )

    percentiles.to_parquet(
        PERCENTILES_FILE,
        index=False,
    )

    print()
    print("=" * 104)
    print(
        "SAUDI PERCENTILE SUMMARY"
        .center(104)
    )
    print("=" * 104)

    print(
        f"Season players: "
        f"{len(season)}"
    )

    print(
        f"Players >= {MIN_MINUTES} min "
        f"with inferred role: "
        f"{len(percentiles)}"
    )

    print()

    print(
        percentiles[
            "position_group"
        ]
        .value_counts()
        .reindex(
            POSITION_ORDER
        )
        .dropna()
        .to_string()
    )

    al_ahli = percentiles[
        pd.to_numeric(
            percentiles[
                "team_id"
            ],
            errors="coerce",
        )
        == 258
    ].copy()

    print()
    print("=" * 104)
    print(
        "AL-AHLI ELIGIBLE PLAYERS"
        .center(104)
    )
    print("=" * 104)

    show_cols = [
        "player_name",
        "position_group",
        "minutes",
        "position_group_share",
    ]

    for col in [
        "xg_per90_percentile",
        "xa_per90_percentile",
        "shots_per90_percentile",
        "key_passes_per90_percentile",
        "touches_per90_percentile",
        "accurate_opposition_half_passes_per90_percentile",
        "total_long_balls_per90_percentile",
        "possession_lost_per90_percentile",
    ]:
        if col in al_ahli.columns:
            show_cols.append(
                col
            )

    if al_ahli.empty:
        print(
            "No eligible Al-Ahli players."
        )
    else:
        print(
            al_ahli[
                show_cols
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
    print("Saved:")
    print(SEASON_METRICS_FILE)
    print(PERCENTILES_FILE)
    print(MINUTES_FILE)


if __name__ == "__main__":
    main()
