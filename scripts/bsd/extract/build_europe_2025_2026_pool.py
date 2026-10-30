from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv


# ======================================================================================
# CONFIG
# ======================================================================================

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "bsd" / "europe_2025_26"
PROCESSED_DIR = ROOT / "data" / "processed" / "bsd"

RAW_ROOT.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PLAYER_MATCH = RAW_ROOT / "player_match_stats.parquet"
OUTPUT_AVG_POS = RAW_ROOT / "average_positions.parquet"
OUTPUT_MATCHES = RAW_ROOT / "matches.parquet"
OUTPUT_METADATA = RAW_ROOT / "metadata.json"

OUTPUT_SEASON = PROCESSED_DIR / "europe_2025_26_player_season_metrics.parquet"
OUTPUT_PERCENTILES = PROCESSED_DIR / "europe_2025_26_player_percentiles.parquet"

BASE_URL = "https://sports.bzzoiro.com/api/v2"
REQUEST_SLEEP = 0.08
REQUEST_TIMEOUT = 30
MIN_MINUTES = 900

# Current tactical position thresholds used in the Newcastle project.
DEFENDER_WIDE_DISTANCE = 24.0
FULLBACK_MIN_X = 38.0
MIDFIELDER_WIDE_DISTANCE = 20.0
FORWARD_WIDE_DISTANCE = 17.0
ATTACKING_MID_X = 60.0

TARGET_LEAGUES = [
    {
        "slug": "premier_league",
        "country": "England",
        "name_candidates": ["Premier League"],
    },
    {
        "slug": "bundesliga",
        "country": "Germany",
        "name_candidates": ["Bundesliga"],
    },
    {
        "slug": "la_liga",
        "country": "Spain",
        "name_candidates": ["LaLiga", "La Liga"],
    },
    {
        "slug": "serie_a",
        "country": "Italy",
        "name_candidates": ["Serie A"],
    },
    {
        "slug": "ligue_1",
        "country": "France",
        "name_candidates": ["Ligue 1"],
    },
    {
        "slug": "eredivisie",
        "country": "Netherlands",
        "name_candidates": ["Eredivisie"],
    },
]

# Canonical metric -> possible flattened BSD names.
#
# The parser recursively flattens nested player-stat categories, so aliases include
# both simple and category-prefixed forms. Missing metrics remain NULL.
STAT_ALIASES = {
    "goals": [
        "goals",
        "attacking_goals",
    ],
    "assists": [
        "assists",
        "attacking_assists",
    ],
    "xg": [
        "xg",
        "expected_goals",
        "attacking_expected_goals",
        "advanced_expected_goals",
    ],
    "xa": [
        "xa",
        "expected_assists",
        "attacking_expected_assists",
        "advanced_expected_assists",
    ],
    "xgot": [
        "xgot",
        "expected_goals_on_target",
        "advanced_expected_goals_on_target",
    ],
    "shots": [
        "shots",
        "total_shots",
        "attacking_total_shots",
    ],
    "shots_on_target": [
        "shots_on_target",
        "attacking_shots_on_target",
    ],
    "headed_shots": [
        "headed_shots",
        "attacking_headed_shots",
        "advanced_headed_shots",
    ],
    "key_passes": [
        "key_passes",
        "chances_created",
        "passing_key_passes",
        "passing_chances_created",
    ],
    "big_chances_created": [
        "big_chances_created",
        "attacking_big_chances_created",
        "advanced_big_chances_created",
    ],
    "successful_passes": [
        "successful_passes",
        "accurate_passes",
        "passing_accurate_passes",
    ],
    "total_passes": [
        "total_passes",
        "passes",
        "passing_total_passes",
    ],
    "accurate_opposition_half_passes": [
        "accurate_opposition_half_passes",
        "passing_accurate_opposition_half_passes",
    ],
    "total_long_balls": [
        "total_long_balls",
        "long_balls",
        "passing_total_long_balls",
    ],
    "accurate_long_balls": [
        "accurate_long_balls",
        "passing_accurate_long_balls",
    ],
    "total_crosses": [
        "total_crosses",
        "crosses",
        "passing_total_crosses",
    ],
    "accurate_crosses": [
        "accurate_crosses",
        "successful_crosses",
        "passing_accurate_crosses",
    ],
    "line_breaking_passes": [
        "line_breaking_passes",
        "passing_line_breaking_passes",
        "advanced_line_breaking_passes",
    ],
    "touches": [
        "touches",
        "possession_touches",
    ],
    "box_touches": [
        "touches_in_opposition_box",
        "touches_opposition_box",
        "box_touches",
        "possession_touches_in_opposition_box",
    ],
    "progressive_carries": [
        "progressive_carries",
        "ball_carries_progressive_carries",
        "advanced_progressive_carries",
    ],
    "progressive_carry_distance": [
        "total_progressive_ball_carries_distance",
        "progressive_carry_distance",
        "ball_carries_progressive_distance",
        "advanced_progressive_carry_distance",
    ],
    "successful_dribbles": [
        "successful_dribbles",
        "possession_successful_dribbles",
    ],
    "total_dribbles": [
        "total_dribbles",
        "dribbles_attempted",
        "possession_total_dribbles",
    ],
    "duels_won": [
        "duels_won",
        "possession_duels_won",
    ],
    "total_duels": [
        "total_duels",
        "duels",
        "possession_total_duels",
    ],
    "aerials_won": [
        "aerials_won",
        "aerial_duels_won",
        "possession_aerials_won",
    ],
    "total_aerials": [
        "total_aerials",
        "aerial_duels",
        "possession_total_aerials",
    ],
    "recoveries": [
        "recoveries",
        "defending_recoveries",
    ],
    "interceptions": [
        "interceptions",
        "defending_interceptions",
    ],
    "tackles_won": [
        "tackles_won",
        "successful_tackles",
        "defending_tackles_won",
    ],
    "total_tackles": [
        "total_tackles",
        "tackles",
        "defending_total_tackles",
    ],
    "possession_lost": [
        "possession_lost",
        "possession_possession_lost",
    ],
    "dispossessed": [
        "dispossessed",
        "possession_dispossessed",
    ],
    "fouls_won": [
        "fouls_won",
        "possession_fouls_won",
    ],
}


# ======================================================================================
# HTTP
# ======================================================================================

def get_session() -> requests.Session:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("BSD_API_KEY")

    if not api_key:
        raise RuntimeError(
            "BSD_API_KEY was not found. Put it in the project .env file."
        )

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Token {api_key}",
            "Accept": "application/json",
        }
    )

    return session


def get_json(
    session: requests.Session,
    path: str,
    params: dict[str, Any] | None = None,
    retries: int = 4,
) -> Any:
    url = f"{BASE_URL}{path}"

    for attempt in range(retries):
        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue

        response.raise_for_status()

        time.sleep(REQUEST_SLEEP)

        return response.json()

    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def paginated(
    session: requests.Session,
    path: str,
    params: dict[str, Any] | None = None,
    page_size: int = 200,
) -> list[dict]:
    params = dict(params or {})
    rows: list[dict] = []
    offset = 0

    while True:
        page_params = {
            **params,
            "limit": page_size,
            "offset": offset,
        }

        payload = get_json(
            session,
            path,
            page_params,
        )

        if isinstance(payload, list):
            chunk = payload
        elif isinstance(payload, dict):
            chunk = (
                payload.get("results")
                or payload.get("events")
                or payload.get("leagues")
                or payload.get("seasons")
                or []
            )
        else:
            chunk = []

        rows.extend(chunk)

        if len(chunk) < page_size:
            break

        offset += page_size

    return rows


# ======================================================================================
# LEAGUE / SEASON RESOLUTION
# ======================================================================================

def norm(text: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text or "").lower(),
    ).strip()


def resolve_league(
    session: requests.Session,
    target: dict,
) -> dict:
    leagues = paginated(
        session,
        "/leagues/",
        params={
            "country": target["country"],
            "include_inactive": "true",
        },
    )

    candidates = [
        norm(name)
        for name in target["name_candidates"]
    ]

    exact = [
        league
        for league in leagues
        if norm(league.get("name")) in candidates
    ]

    if exact:
        return exact[0]

    partial = [
        league
        for league in leagues
        if any(
            candidate in norm(league.get("name"))
            or norm(league.get("name")) in candidate
            for candidate in candidates
        )
    ]

    if partial:
        return partial[0]

    available = sorted(
        {
            str(league.get("name"))
            for league in leagues
        }
    )

    raise RuntimeError(
        f"Could not resolve {target['slug']} in {target['country']}.\n"
        f"Available leagues:\n  " + "\n  ".join(available)
    )


def season_is_2025_26(season: dict) -> bool:
    name = str(season.get("name") or "").lower()
    start_date = str(season.get("start_date") or "")
    end_date = str(season.get("end_date") or "")

    if (
        "2025/26" in name
        or "2025-26" in name
        or "2025 26" in name
    ):
        return True

    return (
        start_date.startswith("2025-")
        and end_date.startswith("2026-")
    )


def resolve_2025_26_season(
    session: requests.Session,
    league_id: int,
) -> dict:
    payload = get_json(
        session,
        f"/leagues/{league_id}/seasons/",
    )

    if isinstance(payload, list):
        seasons = payload
    else:
        seasons = (
            payload.get("results")
            or payload.get("seasons")
            or []
        )

    matches = [
        season
        for season in seasons
        if season_is_2025_26(season)
    ]

    if not matches:
        names = [
            str(season.get("name"))
            for season in seasons
        ]

        raise RuntimeError(
            f"No 2025/26 season found for league {league_id}.\n"
            f"Available seasons:\n  " + "\n  ".join(names)
        )

    return matches[0]


# ======================================================================================
# PARSING
# ======================================================================================

def flatten_dict(
    obj: dict,
    prefix: str = "",
) -> dict:
    out = {}

    for key, value in obj.items():
        key_norm = norm(key).replace(" ", "_")
        full_key = f"{prefix}_{key_norm}" if prefix else key_norm

        if isinstance(value, dict):
            out.update(
                flatten_dict(
                    value,
                    prefix=full_key,
                )
            )
        elif not isinstance(value, (list, tuple)):
            out[full_key] = value

    return out


def find_player_rows(
    node: Any,
    context: dict | None = None,
) -> list[dict]:
    """
    Recursively finds dicts containing player_id and minutes_played.

    This is intentionally tolerant of changes in the outer player-stats response
    shape while keeping each actual player record intact.
    """
    context = dict(context or {})
    rows: list[dict] = []

    if isinstance(node, list):
        for item in node:
            rows.extend(
                find_player_rows(
                    item,
                    context,
                )
            )
        return rows

    if not isinstance(node, dict):
        return rows

    local_context = dict(context)

    for key in [
        "team_id",
        "team_name",
        "is_home",
        "home",
        "side",
    ]:
        if key in node and not isinstance(node[key], (dict, list)):
            local_context[key] = node[key]

    if "player_id" in node and "minutes_played" in node:
        flat = flatten_dict(node)
        flat.update(
            {
                key: value
                for key, value in local_context.items()
                if key not in flat
            }
        )
        rows.append(flat)
        return rows

    for key, value in node.items():
        child_context = dict(local_context)

        key_norm = norm(key)

        if key_norm == "home":
            child_context["is_home"] = True
        elif key_norm == "away":
            child_context["is_home"] = False

        rows.extend(
            find_player_rows(
                value,
                child_context,
            )
        )

    return rows


def pick_value(
    row: dict,
    aliases: list[str],
) -> Any:
    for alias in aliases:
        if alias in row:
            return row[alias]

    # Fallback to suffix match for nested category prefixes.
    for alias in aliases:
        suffix = f"_{alias}"

        for key, value in row.items():
            if key.endswith(suffix):
                return value

    return None


def canonicalize_player_row(
    row: dict,
    event: dict,
    league_meta: dict,
) -> dict:
    player_id = (
        row.get("player_id")
        or row.get("player_player_id")
        or row.get("player_id_id")
    )

    player_name = (
        row.get("player_name")
        or row.get("name")
        or row.get("player_player_name")
        or row.get("player_name_name")
    )

    team_id = row.get("team_id")
    is_home = row.get("is_home")

    home_team_id = event.get("home_team_id")
    away_team_id = event.get("away_team_id")

    if team_id is None and is_home is True:
        team_id = home_team_id
    elif team_id is None and is_home is False:
        team_id = away_team_id

    team_name = row.get("team_name")

    if team_name is None and team_id == home_team_id:
        team_name = (
            event.get("home_team")
            or event.get("home_team_name")
        )
    elif team_name is None and team_id == away_team_id:
        team_name = (
            event.get("away_team")
            or event.get("away_team_name")
        )

    output = {
        "event_id": event["event_id"],
        "league_id": league_meta["league_id"],
        "league_name": league_meta["league_name"],
        "league_slug": league_meta["league_slug"],
        "season_id": league_meta["season_id"],
        "season_name": league_meta["season_name"],
        "team_id": team_id,
        "team_name": team_name,
        "player_id": player_id,
        "player_name": player_name,
        "position": (
            row.get("position")
            or row.get("player_position")
        ),
        "minutes": (
            row.get("minutes_played")
            or row.get("minutes")
            or 0
        ),
    }

    for canonical, aliases in STAT_ALIASES.items():
        output[canonical] = pick_value(
            row,
            aliases,
        )

    return output


def parse_average_positions(
    stats_payload: dict,
    event: dict,
    league_meta: dict,
) -> list[dict]:
    avg = stats_payload.get("average_positions") or {}

    rows = []

    for side, is_home in [
        ("home", True),
        ("away", False),
    ]:
        players = avg.get(side) or []

        team_id = (
            event.get("home_team_id")
            if is_home
            else event.get("away_team_id")
        )

        team_name = (
            event.get("home_team")
            or event.get("home_team_name")
            if is_home
            else event.get("away_team")
            or event.get("away_team_name")
        )

        for player in players:
            pos = player.get("pos")

            rows.append(
                {
                    "event_id": event["event_id"],
                    "league_id": league_meta["league_id"],
                    "league_name": league_meta["league_name"],
                    "league_slug": league_meta["league_slug"],
                    "season_id": league_meta["season_id"],
                    "team_id": team_id,
                    "team_name": team_name,
                    "is_home": is_home,
                    "player_id": player.get("player_id"),
                    "player_name": player.get("name"),
                    "broad_position": pos,
                    "avg_x": player.get("x"),
                    "avg_y": player.get("y"),
                    "touches": player.get("n"),
                }
            )

    return rows


# ======================================================================================
# POSITION INFERENCE
# ======================================================================================

def infer_position_group(
    broad_position: Any,
    x: Any,
    y: Any,
) -> str | None:
    if broad_position is None:
        return None

    position = str(broad_position).upper().strip()

    try:
        x_val = float(x)
        y_val = float(y)
    except (TypeError, ValueError):
        return None

    # Normalize direction by assuming average-position x increases toward attack.
    # BSD's current average-position feed uses the same 0-100 pitch frame for both sides.
    x_attacking = x_val
    y_attacking = y_val

    wide_distance = abs(
        y_attacking - 50.0
    )

    if position in {"D", "DEF"}:
        if (
            wide_distance >= DEFENDER_WIDE_DISTANCE
            and x_attacking >= FULLBACK_MIN_X
        ):
            return "FB/WB"
        return "CB"

    if position in {"M", "MID"}:
        if wide_distance >= MIDFIELDER_WIDE_DISTANCE:
            return "W"
        if x_attacking >= ATTACKING_MID_X:
            return "AM"
        return "DM/CM"

    if position in {"F", "FWD", "A"}:
        if wide_distance >= FORWARD_WIDE_DISTANCE:
            return "W"
        return "ST"

    if position in {"G", "GK"}:
        return "GK"

    return None


# ======================================================================================
# EXTRACTION
# ======================================================================================

def extract_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    session = get_session()

    all_matches = []
    all_player_match = []
    all_avg_positions = []
    metadata = []

    for league_number, target in enumerate(
        TARGET_LEAGUES,
        start=1,
    ):
        print()
        print("=" * 110)
        print(
            f"[{league_number}/{len(TARGET_LEAGUES)}] "
            f"Resolving {target['slug']}..."
        )
        print("=" * 110)

        league = resolve_league(
            session,
            target,
        )

        league_id = int(
            league["id"]
        )

        season = resolve_2025_26_season(
            session,
            league_id,
        )

        season_id = int(
            season["id"]
        )

        league_meta = {
            "league_slug": target["slug"],
            "league_id": league_id,
            "league_name": league.get("name"),
            "country": target["country"],
            "season_id": season_id,
            "season_name": season.get("name"),
        }

        print(
            f"League: {league_meta['league_name']} "
            f"(id={league_id})"
        )
        print(
            f"Season: {league_meta['season_name']} "
            f"(id={season_id})"
        )

        events = paginated(
            session,
            "/events/",
            params={
                "league_id": league_id,
                "season_id": season_id,
                "status": "finished",
            },
        )

        print(
            f"Finished matches: {len(events)}"
        )

        metadata.append(
            {
                **league_meta,
                "finished_matches": len(events),
            }
        )

        league_cache = (
            RAW_ROOT
            / target["slug"]
        )

        league_cache.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_event_dir = (
            league_cache
            / "events"
        )

        raw_event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, raw_event in enumerate(
            events,
            start=1,
        ):
            event_id = int(
                raw_event.get("id")
                or raw_event.get("event_id")
            )

            event = dict(raw_event)
            event["event_id"] = event_id

            all_matches.append(
                {
                    **league_meta,
                    **event,
                }
            )

            cache_file = (
                raw_event_dir
                / f"{event_id}.json"
            )

            if cache_file.exists():
                payload = json.loads(
                    cache_file.read_text(
                        encoding="utf-8"
                    )
                )
            else:
                player_stats = get_json(
                    session,
                    f"/events/{event_id}/player-stats/",
                )

                stats = get_json(
                    session,
                    f"/events/{event_id}/stats/",
                )

                payload = {
                    "player_stats": player_stats,
                    "stats": stats,
                }

                cache_file.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            player_rows = find_player_rows(
                payload["player_stats"]
            )

            canonical_rows = [
                canonicalize_player_row(
                    row,
                    event,
                    league_meta,
                )
                for row in player_rows
            ]

            all_player_match.extend(
                canonical_rows
            )

            all_avg_positions.extend(
                parse_average_positions(
                    payload["stats"],
                    event,
                    league_meta,
                )
            )

            if (
                index == 1
                or index % 25 == 0
                or index == len(events)
            ):
                positive_minutes = sum(
                    float(row.get("minutes") or 0) > 0
                    for row in canonical_rows
                )

                print(
                    f"  [{index:>3}/{len(events)}] "
                    f"event {event_id} | "
                    f"player rows={len(canonical_rows)} | "
                    f"played={positive_minutes}"
                )

    matches_df = pd.DataFrame(
        all_matches
    )

    player_match_df = pd.DataFrame(
        all_player_match
    )

    avg_pos_df = pd.DataFrame(
        all_avg_positions
    )

    return (
        matches_df,
        player_match_df,
        avg_pos_df,
        metadata,
    )


# ======================================================================================
# TRANSFORM / SQL
# ======================================================================================

def build_position_history(
    avg_positions: pd.DataFrame,
) -> pd.DataFrame:
    df = avg_positions.copy()

    df["position_group"] = df.apply(
        lambda row: infer_position_group(
            row.get("broad_position"),
            row.get("avg_x"),
            row.get("avg_y"),
        ),
        axis=1,
    )

    return df


def build_season_tables(
    player_match: pd.DataFrame,
    position_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_match = player_match.copy()

    numeric_cols = [
        "minutes",
        *STAT_ALIASES.keys(),
    ]

    for col in numeric_cols:
        if col in player_match.columns:
            player_match[col] = pd.to_numeric(
                player_match[col],
                errors="coerce",
            )

    # v2 gives minutes_played directly. Filter unused substitutes here.
    player_match = player_match[
        player_match["minutes"].fillna(0) > 0
    ].copy()

    if player_match.empty:
        raise RuntimeError(
            "No positive-minute player rows were parsed. "
            "Inspect one cached player-stats payload before continuing."
        )

    # Primary tactical role = modal inferred role across covered matches.
    ph = position_history.dropna(
        subset=[
            "player_id",
            "position_group",
        ]
    ).copy()

    if ph.empty:
        raise RuntimeError(
            "No tactical positions could be inferred from average_positions."
        )

    role_counts = (
        ph.groupby(
            [
                "league_slug",
                "player_id",
                "position_group",
            ],
            dropna=False,
        )
        .size()
        .rename("role_matches")
        .reset_index()
    )

    role_counts["role_total"] = (
        role_counts.groupby(
            [
                "league_slug",
                "player_id",
            ]
        )["role_matches"]
        .transform("sum")
    )

    role_counts["position_group_share"] = (
        role_counts["role_matches"]
        / role_counts["role_total"]
    )

    primary_roles = (
        role_counts.sort_values(
            [
                "league_slug",
                "player_id",
                "role_matches",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            [
                "league_slug",
                "player_id",
            ]
        )
        [
            [
                "league_slug",
                "player_id",
                "position_group",
                "position_group_share",
            ]
        ]
    )

    # Write temp parquet and use DuckDB for season aggregation.
    tmp_pm = RAW_ROOT / "_tmp_player_match.parquet"
    tmp_roles = RAW_ROOT / "_tmp_primary_roles.parquet"

    player_match.to_parquet(
        tmp_pm,
        index=False,
    )

    primary_roles.to_parquet(
        tmp_roles,
        index=False,
    )

    con = duckdb.connect()

    # Normalize Windows paths before inserting them into SQL f-strings.
    # Backslashes inside an f-string expression cause a SyntaxError.
    tmp_pm_sql = str(tmp_pm).replace("\\", "/")
    tmp_roles_sql = str(tmp_roles).replace("\\", "/")

    try:
        cols = set(
            player_match.columns
        )

        metric_sums = [
            metric
            for metric in STAT_ALIASES
            if metric in cols
        ]

        sum_sql = ",\n                ".join(
            [
                f"SUM(COALESCE({metric}, 0)) AS {metric}"
                for metric in metric_sums
            ]
        )

        season = con.execute(
            f"""
            WITH aggregated AS (
                SELECT
                    league_id,
                    league_name,
                    league_slug,
                    season_id,
                    season_name,
                    team_id,
                    ANY_VALUE(team_name) AS team_name,
                    player_id,
                    ANY_VALUE(player_name) AS player_name,
                    ANY_VALUE(position) AS broad_position,
                    SUM(minutes) AS minutes,
                    COUNT(*) AS appearances,
                    {sum_sql}
                FROM read_parquet('{tmp_pm_sql}')
                GROUP BY
                    league_id,
                    league_name,
                    league_slug,
                    season_id,
                    season_name,
                    team_id,
                    player_id
            )
            SELECT
                a.*,
                r.position_group,
                r.position_group_share
            FROM aggregated a
            LEFT JOIN read_parquet('{tmp_roles_sql}') r
              ON a.league_slug = r.league_slug
             AND a.player_id = r.player_id
            """
        ).df()

    finally:
        con.close()

        for path in [
            tmp_pm,
            tmp_roles,
        ]:
            if path.exists():
                path.unlink()

    # Derive per90 and percentages only from genuinely available raw columns.
    for metric in metric_sums:
        season[f"{metric}_per90"] = (
            season[metric]
            / season["minutes"]
            * 90.0
        )

    if {
        "successful_passes",
        "total_passes",
    }.issubset(season.columns):
        season["pass_accuracy_pct"] = (
            season["successful_passes"]
            / season["total_passes"].replace(0, pd.NA)
            * 100
        )

    if {
        "accurate_long_balls",
        "total_long_balls",
    }.issubset(season.columns):
        season["long_ball_accuracy_pct"] = (
            season["accurate_long_balls"]
            / season["total_long_balls"].replace(0, pd.NA)
            * 100
        )

    if {
        "accurate_crosses",
        "total_crosses",
    }.issubset(season.columns):
        season["cross_accuracy_pct"] = (
            season["accurate_crosses"]
            / season["total_crosses"].replace(0, pd.NA)
            * 100
        )

    if {
        "duels_won",
        "total_duels",
    }.issubset(season.columns):
        season["duel_win_pct"] = (
            season["duels_won"]
            / season["total_duels"].replace(0, pd.NA)
            * 100
        )

    if {
        "aerials_won",
        "total_aerials",
    }.issubset(season.columns):
        season["aerial_win_pct"] = (
            season["aerials_won"]
            / season["total_aerials"].replace(0, pd.NA)
            * 100
        )

    if {
        "successful_dribbles",
        "total_dribbles",
    }.issubset(season.columns):
        season["dribble_success_pct"] = (
            season["successful_dribbles"]
            / season["total_dribbles"].replace(0, pd.NA)
            * 100
        )

    # Percentiles: within league + tactical position group.
    eligible = season[
        (season["minutes"] >= MIN_MINUTES)
        & season["position_group"].notna()
    ].copy()

    metric_cols = [
        col
        for col in eligible.columns
        if (
            col.endswith("_per90")
            or col.endswith("_pct")
        )
    ]

    # Do not percentile-rank a metric that is entirely unavailable.
    metric_cols = [
        col
        for col in metric_cols
        if eligible[col].notna().any()
    ]

    for metric in metric_cols:
        percentile_col = f"{metric}_percentile"

        eligible[percentile_col] = (
            eligible.groupby(
                [
                    "league_slug",
                    "position_group",
                ]
            )[metric]
            .rank(
                pct=True,
                method="average",
            )
            * 100.0
        )

    return season, eligible


# ======================================================================================
# OUTPUT
# ======================================================================================

def main() -> None:
    print("=" * 120)
    print(
        "EUROPE 2025/26 — COMPLETED-SEASON BSD PLAYER POOL".center(120)
    )
    print("=" * 120)

    print(
        "Target leagues:"
    )

    for target in TARGET_LEAGUES:
        print(
            f"  {target['country']:12s}  {target['slug']}"
        )

    matches, player_match, avg_positions, metadata = extract_all()

    print()
    print("=" * 120)
    print(
        "RAW EXTRACTION SUMMARY".center(120)
    )
    print("=" * 120)

    print(
        f"Matches: {len(matches):,}"
    )
    print(
        f"Player-match squad rows: {len(player_match):,}"
    )

    if "minutes" in player_match.columns:
        positive = pd.to_numeric(
            player_match["minutes"],
            errors="coerce",
        ).fillna(0) > 0

        print(
            f"Positive-minute player rows: {positive.sum():,}"
        )

    print(
        f"Average-position rows: {len(avg_positions):,}"
    )

    matches.to_parquet(
        OUTPUT_MATCHES,
        index=False,
    )

    player_match.to_parquet(
        OUTPUT_PLAYER_MATCH,
        index=False,
    )

    position_history = build_position_history(
        avg_positions
    )

    position_history.to_parquet(
        OUTPUT_AVG_POS,
        index=False,
    )

    OUTPUT_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    season, percentiles = build_season_tables(
        player_match,
        position_history,
    )

    season.to_parquet(
        OUTPUT_SEASON,
        index=False,
    )

    percentiles.to_parquet(
        OUTPUT_PERCENTILES,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "2025/26 EUROPE POOL SUMMARY".center(120)
    )
    print("=" * 120)

    coverage = (
        percentiles.groupby(
            [
                "league_slug",
                "position_group",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        coverage.to_string()
    )

    print()
    print(
        f"Season player-team rows: {len(season):,}"
    )
    print(
        f"Eligible rows >= {MIN_MINUTES} min: {len(percentiles):,}"
    )

    print()
    print("Saved:")
    print(OUTPUT_PLAYER_MATCH)
    print(OUTPUT_AVG_POS)
    print(OUTPUT_MATCHES)
    print(OUTPUT_SEASON)
    print(OUTPUT_PERCENTILES)


if __name__ == "__main__":
    main()
