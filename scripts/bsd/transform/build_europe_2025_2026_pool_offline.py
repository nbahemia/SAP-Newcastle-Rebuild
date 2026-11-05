from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


# ======================================================================================
# PATHS / CONFIG
# ======================================================================================

ROOT = Path(__file__).resolve().parents[3]

RAW_ROOT = ROOT / "data" / "raw" / "bsd" / "europe_2025_26"
PROCESSED_DIR = ROOT / "data" / "processed" / "bsd"

OUTPUT_PLAYER_MATCH = RAW_ROOT / "player_match_stats.parquet"
OUTPUT_AVG_POS = RAW_ROOT / "average_positions.parquet"
OUTPUT_MATCHES = RAW_ROOT / "matches.parquet"
OUTPUT_METADATA = RAW_ROOT / "metadata.json"

OUTPUT_SEASON = PROCESSED_DIR / "europe_2025_26_player_season_metrics.parquet"
OUTPUT_PERCENTILES = PROCESSED_DIR / "europe_2025_26_player_percentiles.parquet"

MIN_MINUTES = 900

DEFENDER_WIDE_DISTANCE = 24.0
FULLBACK_MIN_X = 38.0
MIDFIELDER_WIDE_DISTANCE = 20.0
FORWARD_WIDE_DISTANCE = 17.0
ATTACKING_MID_X = 60.0


# Known metadata. Cache discovery is NOT restricted to this dictionary.
LEAGUE_METADATA = {
    "premier_league": {
        "country": "England",
        "league_id": 1,
        "league_name": "Premier League",
        "season_id": 337,
        "season_name": "Premier League 25/26",
    },
    "bundesliga": {
        "country": "Germany",
        "league_id": 5,
        "league_name": "Bundesliga",
        "season_id": 228,
        "season_name": "Bundesliga 25/26",
    },
    "la_liga": {
        "country": "Spain",
        "league_id": 3,
        "league_name": "La Liga",
        "season_id": 294,
        "season_name": "LaLiga 25/26",
    },
    "serie_a": {
        "country": "Italy",
        "league_id": 4,
        "league_name": "Serie A",
        "season_id": 358,
        "season_name": "Serie A 25/26",
    },
    "ligue_1": {
        "country": "France",
        "league_id": 6,
        "league_name": "Ligue 1",
        "season_id": 317,
        "season_name": "Ligue 1 25/26",
    },
    "eredivisie": {
        "country": "Netherlands",
        "league_id": 10,
        "league_name": "Eredivisie",
        "season_id": 279,
        "season_name": "VriendenLoterij Eredivisie 25/26",
    },
}


# Canonical metric -> raw BSD player-stats field(s).
#
# IMPORTANT:
# - These are intentionally exact first.
# - Missing source statistics stay NULL.
# - Genuine source zeros stay zero.
STAT_ALIASES = {
    "rating": ["rating"],
    "goals": ["goals"],
    "assists": ["goal_assist", "assists"],
    "xg": ["expected_goals", "xg"],
    "xa": ["expected_assists", "xa"],
    "xgot": ["expected_goals_on_target", "xgot"],
    "shots": ["total_shots", "shots"],
    "shots_on_target": ["shots_on_target"],
    "key_passes": ["key_pass", "key_passes"],
    "big_chances_created": ["big_chance_created", "big_chances_created"],
    "big_chances_missed": ["big_chance_missed", "big_chances_missed"],

    "successful_passes": ["accurate_pass", "successful_passes", "accurate_passes"],
    "total_passes": ["total_pass", "total_passes"],
    "accurate_opposition_half_passes": ["accurate_opposition_half_passes"],
    "total_opposition_half_passes": ["total_opposition_half_passes"],
    "accurate_own_half_passes": ["accurate_own_half_passes"],
    "total_own_half_passes": ["total_own_half_passes"],
    "total_long_balls": ["total_long_balls"],
    "accurate_long_balls": ["accurate_long_balls"],
    "total_crosses": ["total_cross", "total_crosses"],
    "accurate_crosses": ["accurate_cross", "accurate_crosses"],

    "touches": ["touches"],
    "ball_carries": ["ball_carries_count", "ball_carries"],
    "progressive_carries": ["progressive_ball_carries_count", "progressive_carries"],
    "ball_carry_distance": ["total_ball_carries_distance", "ball_carry_distance"],
    "progressive_carry_distance": [
        "total_progressive_ball_carries_distance",
        "progressive_carry_distance",
    ],
    "total_progression": ["total_progression"],
    "successful_dribbles": ["won_contest", "successful_dribbles"],
    "total_dribbles": ["total_contest", "total_dribbles"],
    "unsuccessful_touches": ["unsuccessful_touch", "unsuccessful_touches"],

    "duels_won": ["duel_won", "duels_won"],
    "aerials_won": ["aerial_won", "aerials_won"],
    "recoveries": ["ball_recovery", "recoveries"],
    "interceptions": ["interception", "interceptions"],
    "tackles_won": ["won_tackle", "tackles_won"],
    "total_tackles": ["total_tackle", "total_tackles"],
    "clearances": ["total_clearance", "clearances"],
    "blocked_shots": ["blocked_scoring_attempt", "blocked_shots"],
    "outfielder_blocks": ["outfielder_block", "outfielder_blocks"],
    "last_man_tackles": ["last_man_tackle", "last_man_tackles"],

    "possession_lost": ["possession_lost"],
    "dispossessed": ["dispossessed"],
    "fouls_won": ["was_fouled", "fouls_won"],
    "fouls_committed": ["fouls", "fouls_committed"],
    "offsides": ["total_offside", "offsides"],
    "yellow_cards": ["yellow_card", "yellow_cards"],
    "red_cards": ["red_card", "red_cards"],

    "saves": ["saves"],
    "goals_conceded": ["goals_conceded"],
    "punches": ["punches"],
    "high_claims": ["high_claims"],
    "good_high_claims": ["good_high_claim", "good_high_claims"],
    "accurate_keeper_sweeper": ["accurate_keeper_sweeper"],
    "total_keeper_sweeper": ["total_keeper_sweeper"],
    "goals_prevented": ["goals_prevented"],
}

DERIVED_SUM_METRICS = [
    "total_duels",
    "total_aerials",
]

ALL_SUM_METRICS = [*STAT_ALIASES.keys(), *DERIVED_SUM_METRICS]


# ======================================================================================
# HELPERS
# ======================================================================================

def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def humanize_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))


def to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pick_value(row: dict, aliases: list[str]) -> Any:
    """
    Prefer exact raw-field matches.

    Suffix matching remains only as a compatibility fallback for older flattened
    cache shapes. The current nested player_stats rows use exact BSD field names.
    """
    for alias in aliases:
        if alias in row:
            return row[alias]

    for alias in aliases:
        suffix = "_" + alias
        matches = [(key, value) for key, value in row.items() if key.endswith(suffix)]
        if len(matches) == 1:
            return matches[0][1]

    return None


def discover_leagues() -> list[dict]:
    """
    Discover leagues from actual cache folders.

    Metadata is recovered from cache_meta first, then supplemented by known metadata.
    A league is not excluded merely because it is absent from LEAGUE_METADATA.
    """
    leagues: list[dict] = []

    if not RAW_ROOT.exists():
        raise RuntimeError(f"Raw cache root does not exist: {RAW_ROOT}")

    for league_dir in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
        event_dir = league_dir / "events"
        if not event_dir.exists():
            continue

        files = sorted(
            event_dir.glob("*.json"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not files:
            continue

        slug = league_dir.name
        known = dict(LEAGUE_METADATA.get(slug, {}))

        cache_meta: dict[str, Any] = {}
        for file in files[:10]:
            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
                candidate = payload.get("cache_meta") or {}
                if isinstance(candidate, dict):
                    cache_meta.update({k: v for k, v in candidate.items() if v is not None})
                if cache_meta.get("league_id") is not None and cache_meta.get("season_id") is not None:
                    break
            except (OSError, json.JSONDecodeError):
                continue

        league_id = to_int_or_none(cache_meta.get("league_id"))
        season_id = to_int_or_none(cache_meta.get("season_id"))

        if league_id is None:
            league_id = to_int_or_none(known.get("league_id"))
        if season_id is None:
            season_id = to_int_or_none(known.get("season_id"))

        league_name = (
            cache_meta.get("league_name")
            or known.get("league_name")
            or humanize_slug(slug)
        )
        season_name = (
            cache_meta.get("season_name")
            or known.get("season_name")
            or f"{league_name} 25/26"
        )

        leagues.append(
            {
                "slug": slug,
                "country": cache_meta.get("country") or known.get("country"),
                "league_id": league_id,
                "league_name": league_name,
                "season_id": season_id,
                "season_name": season_name,
                "event_dir": event_dir,
                "cached_matches": len(files),
            }
        )

    if not leagues:
        raise RuntimeError(
            "No league event caches discovered under "
            f"{RAW_ROOT}/<league>/events/*.json"
        )

    return leagues


def get_raw_player_rows(payload: dict) -> list[dict]:
    """
    Current cache shape:
        payload["player_stats"] = {
            "event_id": ...,
            "count": ...,
            "player_stats": [...]
        }

    Also supports a direct list for compatibility.
    """
    block = payload.get("player_stats")

    if isinstance(block, list):
        return [row for row in block if isinstance(row, dict)]

    if isinstance(block, dict):
        rows = block.get("player_stats")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    return []


def build_cached_identity_maps(stats_payload: Any) -> tuple[dict[int, str], dict[int, str]]:
    player_names: dict[int, str] = {}
    team_names: dict[int, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return

        if not isinstance(node, dict):
            return

        player_id = node.get("player_id")
        player_name = node.get("player_name") or node.get("name")

        if player_id is not None and player_name:
            try:
                player_names[int(player_id)] = str(player_name)
            except (TypeError, ValueError):
                pass

        team_id = node.get("team_id")
        team_name = node.get("team_name")

        if team_id is not None and team_name:
            try:
                team_names[int(team_id)] = str(team_name)
            except (TypeError, ValueError):
                pass

        for value in node.values():
            walk(value)

    walk(stats_payload)
    return player_names, team_names


def parse_average_positions(
    stats_payload: dict,
    event_id: int,
    league: dict,
) -> list[dict]:
    avg = stats_payload.get("average_positions") or {}
    rows: list[dict] = []

    if not isinstance(avg, dict):
        return rows

    for side, is_home in [("home", True), ("away", False)]:
        players = avg.get(side) or []
        if not isinstance(players, list):
            continue

        for player in players:
            if not isinstance(player, dict):
                continue

            rows.append(
                {
                    "event_id": event_id,
                    "league_id": league["league_id"],
                    "league_name": league["league_name"],
                    "league_slug": league["slug"],
                    "season_id": league["season_id"],
                    "season_name": league["season_name"],
                    "is_home": is_home,
                    "player_id": player.get("player_id"),
                    "player_name": player.get("name"),
                    "broad_position": player.get("pos"),
                    "avg_x": player.get("x"),
                    "avg_y": player.get("y"),
                    "touches": player.get("n"),
                }
            )

    return rows


def infer_team_side_map(raw_players: list[dict], avg_rows: list[dict]) -> dict[int, bool]:
    """
    Infer team_id -> is_home by matching player IDs between player_stats and
    average_positions. This avoids assuming team ordering.
    """
    player_to_team: dict[int, int] = {}

    for row in raw_players:
        player_id = to_int_or_none(row.get("player_id"))
        team_id = to_int_or_none(row.get("team_id"))
        if player_id is not None and team_id is not None:
            player_to_team[player_id] = team_id

    votes: dict[int, list[bool]] = {}

    for row in avg_rows:
        player_id = to_int_or_none(row.get("player_id"))
        is_home = row.get("is_home")
        if player_id is None or not isinstance(is_home, bool):
            continue

        team_id = player_to_team.get(player_id)
        if team_id is not None:
            votes.setdefault(team_id, []).append(is_home)

    result: dict[int, bool] = {}

    for team_id, team_votes in votes.items():
        true_count = sum(team_votes)
        false_count = len(team_votes) - true_count
        if true_count != false_count:
            result[team_id] = true_count > false_count

    return result


def canonicalize_player_row(
    row: dict,
    event_id: int,
    league: dict,
    player_name_map: dict[int, str],
    team_name_map: dict[int, str],
) -> dict:
    player_id = to_int_or_none(row.get("player_id"))
    team_id = to_int_or_none(row.get("team_id"))

    player_name = row.get("player_name") or row.get("name")
    if not player_name and player_id is not None:
        player_name = player_name_map.get(player_id)

    team_name = row.get("team_name")
    if not team_name and team_id is not None:
        team_name = team_name_map.get(team_id)

    output = {
        "event_id": event_id,
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        "league_slug": league["slug"],
        "season_id": league["season_id"],
        "season_name": league["season_name"],
        "team_id": team_id,
        "team_name": team_name,
        "player_id": player_id,
        "player_name": player_name,
        "position": row.get("position") or row.get("pos"),
        "minutes": (
            row.get("minutes_played")
            if row.get("minutes_played") is not None
            else row.get("minutes")
        ),
    }

    for canonical, aliases in STAT_ALIASES.items():
        output[canonical] = pick_value(row, aliases)

    duel_won = row.get("duel_won")
    duel_lost = row.get("duel_lost")
    if duel_won is not None or duel_lost is not None:
        output["total_duels"] = (duel_won or 0) + (duel_lost or 0)
    else:
        output["total_duels"] = None

    aerial_won = row.get("aerial_won")
    aerial_lost = row.get("aerial_lost")
    if aerial_won is not None or aerial_lost is not None:
        output["total_aerials"] = (aerial_won or 0) + (aerial_lost or 0)
    else:
        output["total_aerials"] = None

    return output


# ======================================================================================
# OFFLINE CACHE LOAD
# ======================================================================================

def extract_from_cache():
    leagues = discover_leagues()

    all_matches: list[dict] = []
    all_player_match: list[dict] = []
    all_avg_positions: list[dict] = []
    metadata: list[dict] = []

    print()
    print(f"Discovered {len(leagues)} cached league folders.")

    for idx, league in enumerate(leagues, start=1):
        event_dir: Path = league["event_dir"]
        files = sorted(
            event_dir.glob("*.json"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )

        print()
        print("=" * 110)
        print(f"[{idx}/{len(leagues)}] Loading cached {league['slug']}...")
        print("=" * 110)
        print(f"Cached matches: {len(files)}")

        league_player_rows = 0
        league_positive_rows = 0
        empty_player_stat_matches = 0

        for i, file in enumerate(files, start=1):
            try:
                event_id = int(file.stem)
            except ValueError:
                print(f"WARNING: skipping non-numeric event file {file.name}")
                continue

            try:
                payload = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"WARNING: could not read {file}: {exc}")
                continue

            raw_players = get_raw_player_rows(payload)
            # Keep the outer stats payload. average_positions and identity
            # metadata live alongside the inner "stats" object.
            stats = payload.get("stats") or {}

            if not raw_players:
                empty_player_stat_matches += 1

            avg_rows = parse_average_positions(
                stats_payload=stats,
                event_id=event_id,
                league=league,
            )

            player_name_map, team_name_map = build_cached_identity_maps(stats)
            team_side_map = infer_team_side_map(raw_players, avg_rows)

            canonical = [
                canonicalize_player_row(
                    row=row,
                    event_id=event_id,
                    league=league,
                    player_name_map=player_name_map,
                    team_name_map=team_name_map,
                )
                for row in raw_players
            ]

            # Add team IDs to average-position rows when inferable.
            side_to_team = {side: team for team, side in team_side_map.items()}
            for avg_row in avg_rows:
                avg_row["team_id"] = side_to_team.get(avg_row.get("is_home"))
                team_id = avg_row.get("team_id")
                avg_row["team_name"] = (
                    team_name_map.get(team_id)
                    if team_id is not None
                    else None
                )

            home_team_id = side_to_team.get(True)
            away_team_id = side_to_team.get(False)

            all_player_match.extend(canonical)
            all_avg_positions.extend(avg_rows)

            all_matches.append(
                {
                    "event_id": event_id,
                    "league_id": league["league_id"],
                    "league_name": league["league_name"],
                    "league_slug": league["slug"],
                    "season_id": league["season_id"],
                    "season_name": league["season_name"],
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_team_name": team_name_map.get(home_team_id)
                    if home_team_id is not None else None,
                    "away_team_name": team_name_map.get(away_team_id)
                    if away_team_id is not None else None,
                }
            )

            played = sum(float(row.get("minutes") or 0) > 0 for row in canonical)
            league_player_rows += len(canonical)
            league_positive_rows += played

            if i == 1 or i % 50 == 0 or i == len(files):
                print(
                    f"  [{i:>3}/{len(files)}] "
                    f"event {event_id} | "
                    f"player rows={len(canonical)} | "
                    f"played={played}"
                )

        usable = league_player_rows > 0

        if not usable:
            print(
                f"WARNING: {league['slug']} has zero cached player-stat rows "
                f"across {len(files)} matches. It will not contribute players "
                "to season scouting tables."
            )

        metadata.append(
            {
                "slug": league["slug"],
                "country": league.get("country"),
                "league_id": league["league_id"],
                "league_name": league["league_name"],
                "season_id": league["season_id"],
                "season_name": league["season_name"],
                "cached_matches": len(files),
                "empty_player_stat_matches": empty_player_stat_matches,
                "player_rows": league_player_rows,
                "positive_minute_rows": league_positive_rows,
                "usable_for_player_scouting": usable,
                "source": "offline_cache",
            }
        )

    matches_df = pd.DataFrame(all_matches)
    player_match_df = pd.DataFrame(all_player_match)
    avg_pos_df = pd.DataFrame(all_avg_positions)

    if player_match_df.empty:
        raise RuntimeError(
            "No cached player-stat rows found in any league. "
            "The season tables cannot be rebuilt."
        )

    return matches_df, player_match_df, avg_pos_df, metadata


# ======================================================================================
# POSITION / SEASON TRANSFORM
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

    wide_distance = abs(y_val - 50.0)

    if position in {"D", "DEF"}:
        if wide_distance >= DEFENDER_WIDE_DISTANCE and x_val >= FULLBACK_MIN_X:
            return "FB/WB"
        return "CB"

    if position in {"M", "MID"}:
        if wide_distance >= MIDFIELDER_WIDE_DISTANCE:
            return "W"
        if x_val >= ATTACKING_MID_X:
            return "AM"
        return "DM/CM"

    if position in {"F", "FWD", "A"}:
        if wide_distance >= FORWARD_WIDE_DISTANCE:
            return "W"
        return "ST"

    if position in {"G", "GK"}:
        return "GK"

    return None


def build_position_history(avg_positions: pd.DataFrame) -> pd.DataFrame:
    if avg_positions.empty:
        return avg_positions.copy()

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


def build_primary_roles(position_history: pd.DataFrame) -> pd.DataFrame:
    if position_history.empty:
        return pd.DataFrame(
            columns=[
                "league_slug",
                "player_id",
                "position_group",
                "position_group_share",
            ]
        )

    ph = position_history.dropna(
        subset=["player_id", "position_group"]
    ).copy()

    if ph.empty:
        return pd.DataFrame(
            columns=[
                "league_slug",
                "player_id",
                "position_group",
                "position_group_share",
            ]
        )

    role_counts = (
        ph.groupby(["league_slug", "player_id", "position_group"])
        .size()
        .rename("role_matches")
        .reset_index()
    )

    role_counts["role_total"] = (
        role_counts.groupby(["league_slug", "player_id"])["role_matches"]
        .transform("sum")
    )

    role_counts["position_group_share"] = (
        role_counts["role_matches"] / role_counts["role_total"]
    )

    return (
        role_counts.sort_values(
            ["league_slug", "player_id", "role_matches"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["league_slug", "player_id"])
        [
            [
                "league_slug",
                "player_id",
                "position_group",
                "position_group_share",
            ]
        ]
    )


def build_season_tables(
    player_match: pd.DataFrame,
    position_history: pd.DataFrame,
):
    player_match = player_match.copy()

    numeric_cols = ["minutes", *ALL_SUM_METRICS]

    for col in numeric_cols:
        if col in player_match.columns:
            player_match[col] = pd.to_numeric(
                player_match[col],
                errors="coerce",
            )

    player_match = player_match[
        player_match["minutes"].fillna(0) > 0
    ].copy()

    print()
    print(f"Positive-minute rows retained: {len(player_match):,}")

    primary_roles = build_primary_roles(position_history)

    tmp_pm = RAW_ROOT / "_tmp_player_match.parquet"
    tmp_roles = RAW_ROOT / "_tmp_primary_roles.parquet"

    player_match.to_parquet(tmp_pm, index=False)

    metric_sums = [
        metric for metric in ALL_SUM_METRICS
        if metric in player_match.columns
    ]

    sum_sql = ",\n                    ".join(
        (
            f"CASE WHEN COUNT({metric}) > 0 "
            f"THEN SUM({metric}) ELSE NULL END AS {metric}"
        )
        for metric in metric_sums
    )

    tmp_pm_sql = str(tmp_pm).replace("\\", "/")

    con = duckdb.connect()

    try:
        season = con.execute(
            f"""
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
            """
        ).df()
    finally:
        con.close()
        if tmp_pm.exists():
            tmp_pm.unlink()

    if not primary_roles.empty:
        season = season.merge(
            primary_roles,
            how="left",
            on=["league_slug", "player_id"],
        )
    else:
        season["position_group"] = None
        season["position_group_share"] = pd.NA

    for metric in metric_sums:
        season[f"{metric}_per90"] = (
            season[metric] / season["minutes"] * 90.0
        )

    ratio_specs = {
        "pass_accuracy_pct": ("successful_passes", "total_passes"),
        "long_ball_accuracy_pct": ("accurate_long_balls", "total_long_balls"),
        "cross_accuracy_pct": ("accurate_crosses", "total_crosses"),
        "duel_win_pct": ("duels_won", "total_duels"),
        "aerial_win_pct": ("aerials_won", "total_aerials"),
        "dribble_success_pct": ("successful_dribbles", "total_dribbles"),
    }

    for output_col, (numerator, denominator) in ratio_specs.items():
        if {numerator, denominator}.issubset(season.columns):
            season[output_col] = (
                season[numerator]
                / season[denominator].replace(0, pd.NA)
                * 100.0
            )

    eligible = season[
        (season["minutes"] >= MIN_MINUTES)
        & season["position_group"].notna()
    ].copy()

    metric_cols = [
        col for col in eligible.columns
        if col.endswith("_per90") or col.endswith("_pct")
    ]

    for metric in metric_cols:
        if not eligible[metric].notna().any():
            continue

        eligible[f"{metric}_percentile"] = (
            eligible.groupby(["league_slug", "position_group"])[metric]
            .rank(pct=True, method="average")
            * 100.0
        )

    return season, eligible


# ======================================================================================
# VALIDATION
# ======================================================================================

def metric_coverage(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            rows.append(
                {
                    "metric": metric,
                    "non_null": 0,
                    "non_zero": 0,
                    "sum": None,
                    "max": None,
                    "status": "MISSING_COLUMN",
                }
            )
            continue

        s = pd.to_numeric(df[metric], errors="coerce")
        non_null = int(s.notna().sum())
        non_zero = int(s.fillna(0).ne(0).sum())

        if non_null == 0:
            status = "NO_SOURCE_VALUES"
        elif non_zero == 0:
            status = "ALL_ZERO"
        else:
            status = "OK"

        rows.append(
            {
                "metric": metric,
                "non_null": non_null,
                "non_zero": non_zero,
                "sum": float(s.sum()) if non_null else None,
                "max": float(s.max()) if non_null else None,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def validate_known_raw_record(player_match: pd.DataFrame) -> None:
    """
    Validate the exact Premier League event/player record inspected during debugging.

    Raw cache reference:
      league_slug = premier_league
      event_id = 1
      player_id = 325
      minutes = 90
      total_pass = 74
      accurate_pass = 63
      goal_assist = 0
      expected_assists = 0.00952033
      duel_won = 13
      duel_lost = 1
    """
    mask = (
        (player_match["league_slug"] == "premier_league")
        & (pd.to_numeric(player_match["event_id"], errors="coerce") == 1)
        & (pd.to_numeric(player_match["player_id"], errors="coerce") == 325)
    )

    rows = player_match.loc[mask]

    if rows.empty:
        print(
            "WARNING: known Premier League validation record "
            "(event 1, player 325) was not found."
        )
        return

    row = rows.iloc[0]

    expected = {
        "minutes": 90.0,
        "total_passes": 74.0,
        "successful_passes": 63.0,
        "assists": 0.0,
        "xa": 0.00952033,
        "duels_won": 13.0,
        "total_duels": 14.0,
    }

    failures = []

    for field, expected_value in expected.items():
        actual = pd.to_numeric(
            pd.Series([row.get(field)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(actual) or abs(float(actual) - expected_value) > 1e-8:
            failures.append(
                f"{field}: expected {expected_value}, got {row.get(field)!r}"
            )

    if failures:
        raise RuntimeError(
            "KNOWN RAW RECORD VALIDATION FAILED:\n  - "
            + "\n  - ".join(failures)
        )

    print(
        "Known-record validation PASSED: "
        "Premier League event 1 / player 325."
    )


def validate_player_match(player_match: pd.DataFrame) -> pd.DataFrame:
    print()
    print("=" * 120)
    print("PLAYER-MATCH VALIDATION".center(120))
    print("=" * 120)

    validate_known_raw_record(player_match)

    critical_metrics = [
        "goals",
        "assists",
        "xg",
        "xa",
        "key_passes",
        "successful_passes",
        "total_passes",
        "total_crosses",
        "accurate_crosses",
        "successful_dribbles",
        "total_dribbles",
        "duels_won",
        "total_duels",
        "aerials_won",
        "total_aerials",
        "recoveries",
        "interceptions",
        "progressive_carries",
        "progressive_carry_distance",
    ]

    coverage = metric_coverage(player_match, critical_metrics)
    print(coverage.to_string(index=False))

    # Hard corruption check: assists and xA must not be identical everywhere
    # when both fields have populated observations.
    if {"assists", "xa"}.issubset(player_match.columns):
        assists = pd.to_numeric(player_match["assists"], errors="coerce")
        xa = pd.to_numeric(player_match["xa"], errors="coerce")
        valid = assists.notna() & xa.notna()

        if valid.any():
            identical = (assists[valid] == xa[valid])
            pct = float(identical.mean() * 100.0)

            print()
            print(
                f"assists == xA: {identical.sum():,}/{valid.sum():,} "
                f"({pct:.2f}%)"
            )

            if pct >= 99.9 and valid.sum() >= 100:
                raise RuntimeError(
                    "VALIDATION FAILED: assists and xA are effectively identical "
                    "across all populated player-match rows. This reproduces the "
                    "corruption the rebuild is intended to fix."
                )

    # These are known to exist in the populated BSD player-stat cache.
    must_have_nonzero = [
        "total_passes",
        "successful_passes",
        "key_passes",
        "total_duels",
    ]

    for metric in must_have_nonzero:
        if metric not in player_match.columns:
            raise RuntimeError(
                f"VALIDATION FAILED: required metric column missing: {metric}"
            )

        s = pd.to_numeric(player_match[metric], errors="coerce")

        if not s.notna().any():
            raise RuntimeError(
                f"VALIDATION FAILED: {metric} has no populated values."
            )

        if not s.fillna(0).ne(0).any():
            raise RuntimeError(
                f"VALIDATION FAILED: {metric} is zero for every player-match row."
            )

    print()
    print("Player-match validation PASSED.")
    return coverage


def validate_season(season: pd.DataFrame) -> None:
    print()
    print("=" * 120)
    print("SEASON VALIDATION".center(120))
    print("=" * 120)

    required = [
        "total_passes",
        "successful_passes",
        "key_passes",
        "total_duels",
    ]

    for metric in required:
        if metric not in season.columns:
            raise RuntimeError(
                f"SEASON VALIDATION FAILED: missing {metric}"
            )

        s = pd.to_numeric(season[metric], errors="coerce")
        if not s.fillna(0).ne(0).any():
            raise RuntimeError(
                f"SEASON VALIDATION FAILED: {metric} is zero everywhere."
            )

    if {"assists", "xa"}.issubset(season.columns):
        assists = pd.to_numeric(season["assists"], errors="coerce")
        xa = pd.to_numeric(season["xa"], errors="coerce")
        valid = assists.notna() & xa.notna()

        if valid.any() and (assists[valid] == xa[valid]).mean() >= 0.999:
            raise RuntimeError(
                "SEASON VALIDATION FAILED: assists and xA remain effectively "
                "identical after aggregation."
            )

    print("Season validation PASSED.")


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("EUROPE 2025/26 — OFFLINE CACHE-ONLY PLAYER POOL REBUILD".center(120))
    print("=" * 120)
    print("ZERO API REQUESTS")
    print("Source: cached JSON event files only")

    matches, player_match, avg_positions, metadata = extract_from_cache()

    print()
    print("=" * 120)
    print("CACHE SUMMARY".center(120))
    print("=" * 120)

    print(f"Matches: {len(matches):,}")
    print(f"Player-match rows: {len(player_match):,}")
    print(f"Average-position rows: {len(avg_positions):,}")

    usable = [m for m in metadata if m["usable_for_player_scouting"]]
    unusable = [m for m in metadata if not m["usable_for_player_scouting"]]

    print(f"Usable player-stat leagues: {len(usable)}")
    print(f"Empty player-stat leagues: {len(unusable)}")

    if unusable:
        print(
            "Empty leagues: "
            + ", ".join(m["slug"] for m in unusable)
        )

    if "player_name" in player_match.columns:
        player_name_cov = player_match["player_name"].notna().mean() * 100
        print(f"Player-name coverage: {player_name_cov:.1f}%")

    if "team_name" in player_match.columns:
        team_name_cov = player_match["team_name"].notna().mean() * 100
        print(f"Team-name coverage: {team_name_cov:.1f}%")

    # Validate BEFORE overwriting the canonical Parquets.
    coverage = validate_player_match(player_match)

    position_history = build_position_history(avg_positions)

    season, percentiles = build_season_tables(
        player_match,
        position_history,
    )

    validate_season(season)

    # Only write final outputs after all hard validations pass.
    matches.to_parquet(OUTPUT_MATCHES, index=False)
    player_match.to_parquet(OUTPUT_PLAYER_MATCH, index=False)
    position_history.to_parquet(OUTPUT_AVG_POS, index=False)
    season.to_parquet(OUTPUT_SEASON, index=False)
    percentiles.to_parquet(OUTPUT_PERCENTILES, index=False)

    metadata_payload = {
        "generated_from": "offline_cache",
        "api_requests": 0,
        "minimum_minutes": MIN_MINUTES,
        "leagues": metadata,
        "metric_coverage": coverage.to_dict(orient="records"),
    }

    OUTPUT_METADATA.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FINAL PLAYER POOL".center(120))
    print("=" * 120)

    print(f"Season player-team rows: {len(season):,}")
    print(
        f"Eligible >= {MIN_MINUTES} minutes with inferred role: "
        f"{len(percentiles):,}"
    )

    if not percentiles.empty:
        summary = (
            percentiles.groupby(["league_slug", "position_group"])
            .size()
            .unstack(fill_value=0)
        )

        print()
        print(summary.to_string())

    print()
    print("Saved:")
    print(OUTPUT_MATCHES)
    print(OUTPUT_PLAYER_MATCH)
    print(OUTPUT_AVG_POS)
    print(OUTPUT_SEASON)
    print(OUTPUT_PERCENTILES)
    print(OUTPUT_METADATA)

    print()
    print("REBUILD COMPLETE — VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
