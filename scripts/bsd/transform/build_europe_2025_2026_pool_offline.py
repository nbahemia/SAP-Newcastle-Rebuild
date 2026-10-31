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

FALLBACK_LEAGUES = [
    {"slug": "premier_league", "country": "England", "league_id": 1, "league_name": "Premier League", "season_id": 337, "season_name": "Premier League 25/26"},
    {"slug": "bundesliga", "country": "Germany", "league_id": 5, "league_name": "Bundesliga", "season_id": 228, "season_name": "Bundesliga 25/26"},
    {"slug": "la_liga", "country": "Spain", "league_id": 3, "league_name": "La Liga", "season_id": 294, "season_name": "LaLiga 25/26"},
    {"slug": "serie_a", "country": "Italy", "league_id": 4, "league_name": "Serie A", "season_id": 358, "season_name": "Serie A 25/26"},
    {"slug": "ligue_1", "country": "France", "league_id": 6, "league_name": "Ligue 1", "season_id": 317, "season_name": "Ligue 1 25/26"},
    {"slug": "eredivisie", "country": "Netherlands", "league_id": 10, "league_name": "Eredivisie", "season_id": 279, "season_name": "VriendenLoterij Eredivisie 25/26"},
]

CACHE_MANIFEST = RAW_ROOT / "cache_manifest.json"


def discover_leagues() -> list[dict]:
    """Merge the known original six leagues with any leagues added by the cache extractor."""
    by_slug = {row["slug"]: dict(row) for row in FALLBACK_LEAGUES}

    if CACHE_MANIFEST.exists():
        try:
            payload = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
            for row in payload.get("leagues", []):
                slug = row.get("slug")
                if not slug:
                    continue
                by_slug[slug] = {
                    "slug": slug,
                    "country": row.get("country"),
                    "league_id": row.get("league_id"),
                    "league_name": row.get("league_name"),
                    "season_id": row.get("season_id"),
                    "season_name": row.get("season_name"),
                }
        except Exception as exc:
            print(f"WARNING: could not read cache manifest: {exc}")

    # Only include leagues that actually have cached event files.
    leagues = []
    for slug, row in by_slug.items():
        event_dir = RAW_ROOT / slug / "events"
        if event_dir.exists() and any(event_dir.glob("*.json")):
            leagues.append(row)

    # Keep the original six first, then expansion leagues alphabetically.
    original_order = {row["slug"]: i for i, row in enumerate(FALLBACK_LEAGUES)}
    leagues.sort(key=lambda r: (original_order.get(r["slug"], 999), r["slug"]))
    return leagues



STAT_ALIASES = {
    "goals": ["goals", "attacking_goals"],
    "assists": ["assists", "attacking_assists"],
    "xg": ["xg", "expected_goals", "attacking_expected_goals"],
    "xa": ["xa", "expected_assists", "attacking_expected_assists"],
    "xgot": ["xgot", "expected_goals_on_target"],
    "shots": ["shots", "total_shots", "attacking_total_shots"],
    "shots_on_target": ["shots_on_target", "attacking_shots_on_target"],
    "headed_shots": ["headed_shots", "attacking_headed_shots"],
    "key_passes": ["key_passes", "chances_created", "passing_key_passes"],
    "big_chances_created": ["big_chances_created", "big_chance_created", "attacking_big_chances_created"],
    "successful_passes": ["successful_passes", "accurate_passes", "passing_accurate_passes"],
    "total_passes": ["total_passes", "passes", "passing_total_passes"],
    "accurate_opposition_half_passes": [
        "accurate_opposition_half_passes",
        "passing_accurate_opposition_half_passes",
    ],
    "total_long_balls": ["total_long_balls", "long_balls", "passing_total_long_balls"],
    "accurate_long_balls": ["accurate_long_balls", "passing_accurate_long_balls"],
    "total_crosses": ["total_crosses", "crosses", "passing_total_crosses"],
    "accurate_crosses": ["accurate_crosses", "successful_crosses", "passing_accurate_crosses"],
    "line_breaking_passes": ["line_breaking_passes", "passing_line_breaking_passes"],
    "touches": ["touches", "possession_touches"],
    "box_touches": [
        "touches_in_opposition_box",
        "touches_opposition_box",
        "box_touches",
    ],
    "progressive_carries": ["progressive_carries", "progressive_ball_carries_count"],
    "progressive_carry_distance": [
        "total_progressive_ball_carries_distance",
        "progressive_carry_distance",
    ],
    "successful_dribbles": ["successful_dribbles", "won_contest"],
    "total_dribbles": ["total_dribbles", "dribbles_attempted", "total_contest"],
    "duels_won": ["duels_won", "duel_won"],
    "total_duels": ["total_duels", "duels"],
    "aerials_won": ["aerials_won", "aerial_duels_won", "aerial_won"],
    "total_aerials": ["total_aerials", "aerial_duels"],
    "recoveries": ["recoveries", "defending_recoveries", "ball_recovery"],
    "interceptions": ["interceptions", "defending_interceptions", "interception"],
    "tackles_won": ["tackles_won", "successful_tackles", "won_tackle"],
    "total_tackles": ["total_tackles", "tackles", "total_tackle"],
    "possession_won_final_third": [
        "possession_won_final_third",
        "possession_won_final_3rd",
        "won_possession_final_third",
        "won_possession_final_3rd",
        "defending_possession_won_final_third",
    ],
    "possession_lost": ["possession_lost"],
    "dispossessed": ["dispossessed"],
    "fouls_won": ["fouls_won"],
}


# ======================================================================================
# HELPERS
# ======================================================================================

def norm(text: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(text or "").lower(),
    ).strip("_")


def flatten_dict(obj: dict, prefix: str = "") -> dict:
    out = {}

    for key, value in obj.items():
        clean = norm(key)
        full = f"{prefix}_{clean}" if prefix else clean

        if isinstance(value, dict):
            out.update(flatten_dict(value, full))
        elif not isinstance(value, (list, tuple)):
            out[full] = value

    return out


def find_player_rows(node: Any, context: dict | None = None) -> list[dict]:
    context = dict(context or {})
    rows = []

    if isinstance(node, list):
        for item in node:
            rows.extend(find_player_rows(item, context))
        return rows

    if not isinstance(node, dict):
        return rows

    local = dict(context)

    for key in ["team_id", "team_name", "is_home", "side"]:
        if key in node and not isinstance(node[key], (dict, list)):
            local[key] = node[key]

    if "player_id" in node and (
        "minutes_played" in node or "minutes" in node
    ):
        flat = flatten_dict(node)

        for key, value in local.items():
            flat.setdefault(key, value)

        rows.append(flat)
        return rows

    for key, value in node.items():
        child = dict(local)
        key_n = norm(key)

        if key_n == "home":
            child["is_home"] = True
        elif key_n == "away":
            child["is_home"] = False

        rows.extend(find_player_rows(value, child))

    return rows


def pick_value(row: dict, aliases: list[str]) -> Any:
    for alias in aliases:
        if alias in row:
            return row[alias]

    for alias in aliases:
        suffix = "_" + alias
        for key, value in row.items():
            if key.endswith(suffix):
                return value

    return None


def derive_event_team_metadata(player_rows: list[dict]) -> dict:
    home_ids = []
    away_ids = []
    home_names = []
    away_names = []

    for row in player_rows:
        is_home = row.get("is_home")
        team_id = row.get("team_id")
        team_name = row.get("team_name")

        if is_home is True:
            if team_id is not None:
                home_ids.append(team_id)
            if team_name:
                home_names.append(team_name)

        elif is_home is False:
            if team_id is not None:
                away_ids.append(team_id)
            if team_name:
                away_names.append(team_name)

    return {
        "home_team_id": home_ids[0] if home_ids else None,
        "away_team_id": away_ids[0] if away_ids else None,
        "home_team_name": home_names[0] if home_names else None,
        "away_team_name": away_names[0] if away_names else None,
    }


def canonicalize_player_row(
    row: dict,
    event_id: int,
    event_meta: dict,
    league: dict,
) -> dict:
    team_id = row.get("team_id")

    if team_id is None:
        if row.get("is_home") is True:
            team_id = event_meta.get("home_team_id")
        elif row.get("is_home") is False:
            team_id = event_meta.get("away_team_id")

    team_name = row.get("team_name")

    if not team_name:
        if team_id == event_meta.get("home_team_id"):
            team_name = event_meta.get("home_team_name")
        elif team_id == event_meta.get("away_team_id"):
            team_name = event_meta.get("away_team_name")

    output = {
        "event_id": event_id,
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        "league_slug": league["slug"],
        "season_id": league["season_id"],
        "season_name": league["season_name"],
        "team_id": team_id,
        "team_name": team_name,
        "player_id": row.get("player_id"),
        "player_name": (
            row.get("player_name")
            or row.get("name")
        ),
        "position": row.get("position"),
        "minutes": (
            row.get("minutes_played")
            if row.get("minutes_played") is not None
            else row.get("minutes", 0)
        ),
    }

    for canonical, aliases in STAT_ALIASES.items():
        output[canonical] = pick_value(row, aliases)

    return output


def parse_average_positions(
    stats_payload: dict,
    event_id: int,
    event_meta: dict,
    league: dict,
) -> list[dict]:
    avg = stats_payload.get("average_positions") or {}
    rows = []

    for side, is_home in [("home", True), ("away", False)]:
        players = avg.get(side) or []

        team_id = (
            event_meta.get("home_team_id")
            if is_home
            else event_meta.get("away_team_id")
        )

        team_name = (
            event_meta.get("home_team_name")
            if is_home
            else event_meta.get("away_team_name")
        )

        for player in players:
            rows.append(
                {
                    "event_id": event_id,
                    "league_id": league["league_id"],
                    "league_name": league["league_name"],
                    "league_slug": league["slug"],
                    "season_id": league["season_id"],
                    "team_id": team_id,
                    "team_name": team_name,
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
        if (
            wide_distance >= DEFENDER_WIDE_DISTANCE
            and x_val >= FULLBACK_MIN_X
        ):
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


# ======================================================================================
# OFFLINE CACHE LOAD
# ======================================================================================

def extract_from_cache():
    all_matches = []
    all_player_match = []
    all_avg_positions = []
    metadata = []

    leagues = discover_leagues()
    if not leagues:
        raise RuntimeError("No cached leagues were discovered.")

    print("Discovered leagues:", ", ".join(x["slug"] for x in leagues))

    for idx, league in enumerate(leagues, start=1):
        event_dir = RAW_ROOT / league["slug"] / "events"

        print()
        print("=" * 110)
        print(
            f"[{idx}/{len(leagues)}] Loading cached {league['slug']}..."
        )
        print("=" * 110)

        if not event_dir.exists():
            print(f"WARNING: missing {event_dir}")
            continue

        files = sorted(
            event_dir.glob("*.json"),
            key=lambda p: int(p.stem),
        )

        print(f"Cached matches: {len(files)}")

        league_player_rows = 0
        league_positive_rows = 0

        for i, file in enumerate(files, start=1):
            event_id = int(file.stem)

            payload = json.loads(
                file.read_text(encoding="utf-8")
            )

            player_stats = payload.get("player_stats", {})
            stats = payload.get("stats", {})

            player_rows = find_player_rows(player_stats)

            event_meta = derive_event_team_metadata(player_rows)

            canonical = [
                canonicalize_player_row(
                    row,
                    event_id,
                    event_meta,
                    league,
                )
                for row in player_rows
            ]

            avg_rows = parse_average_positions(
                stats,
                event_id,
                event_meta,
                league,
            )

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
                    **event_meta,
                }
            )

            played = sum(
                float(row.get("minutes") or 0) > 0
                for row in canonical
            )

            league_player_rows += len(canonical)
            league_positive_rows += played

            if i == 1 or i % 50 == 0 or i == len(files):
                print(
                    f"  [{i:>3}/{len(files)}] "
                    f"event {event_id} | "
                    f"player rows={len(canonical)} | "
                    f"played={played}"
                )

        metadata.append(
            {
                **league,
                "cached_matches": len(files),
                "player_rows": league_player_rows,
                "positive_minute_rows": league_positive_rows,
                "source": "offline_cache",
            }
        )

    matches_df = pd.DataFrame(all_matches)
    player_match_df = pd.DataFrame(all_player_match)
    avg_pos_df = pd.DataFrame(all_avg_positions)

    if player_match_df.empty:
        raise RuntimeError(
            "No cached player rows found. "
            "Check data/raw/bsd/europe_2025_26/<league>/events/"
        )

    return matches_df, player_match_df, avg_pos_df, metadata


# ======================================================================================
# TRANSFORM
# ======================================================================================

def build_position_history(avg_positions: pd.DataFrame) -> pd.DataFrame:
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
):
    player_match = player_match.copy()

    numeric_cols = ["minutes", *STAT_ALIASES.keys()]

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

    ph = position_history.dropna(
        subset=["player_id", "position_group"]
    ).copy()

    role_counts = (
        ph.groupby(
            ["league_slug", "player_id", "position_group"]
        )
        .size()
        .rename("role_matches")
        .reset_index()
    )

    if role_counts.empty:
        raise RuntimeError(
            "No position groups were inferred. "
            "Check cached average_positions coverage."
        )

    role_counts["role_total"] = (
        role_counts.groupby(
            ["league_slug", "player_id"]
        )["role_matches"]
        .transform("sum")
    )

    role_counts["position_group_share"] = (
        role_counts["role_matches"]
        / role_counts["role_total"]
    )

    primary_roles = (
        role_counts.sort_values(
            ["league_slug", "player_id", "role_matches"],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            ["league_slug", "player_id"]
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

    tmp_pm = RAW_ROOT / "_tmp_player_match.parquet"
    tmp_roles = RAW_ROOT / "_tmp_primary_roles.parquet"

    player_match.to_parquet(tmp_pm, index=False)
    primary_roles.to_parquet(tmp_roles, index=False)

    tmp_pm_sql = str(tmp_pm).replace("\\", "/")
    tmp_roles_sql = str(tmp_roles).replace("\\", "/")

    metric_sums = [
        metric
        for metric in STAT_ALIASES
        if metric in player_match.columns
    ]

    sum_sql = ",\n                ".join(
        f"SUM(COALESCE({metric}, 0)) AS {metric}"
        for metric in metric_sums
    )

    con = duckdb.connect()

    try:
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

        if tmp_pm.exists():
            tmp_pm.unlink()

        if tmp_roles.exists():
            tmp_roles.unlink()

    for metric in metric_sums:
        season[f"{metric}_per90"] = (
            season[metric]
            / season["minutes"]
            * 90.0
        )

    if {"successful_passes", "total_passes"}.issubset(season.columns):
        season["pass_accuracy_pct"] = (
            season["successful_passes"]
            / season["total_passes"].replace(0, pd.NA)
            * 100
        )

    if {"accurate_long_balls", "total_long_balls"}.issubset(season.columns):
        season["long_ball_accuracy_pct"] = (
            season["accurate_long_balls"]
            / season["total_long_balls"].replace(0, pd.NA)
            * 100
        )

    if {"accurate_crosses", "total_crosses"}.issubset(season.columns):
        season["cross_accuracy_pct"] = (
            season["accurate_crosses"]
            / season["total_crosses"].replace(0, pd.NA)
            * 100
        )

    if {"duels_won", "total_duels"}.issubset(season.columns):
        season["duel_win_pct"] = (
            season["duels_won"]
            / season["total_duels"].replace(0, pd.NA)
            * 100
        )

    if {"aerials_won", "total_aerials"}.issubset(season.columns):
        season["aerial_win_pct"] = (
            season["aerials_won"]
            / season["total_aerials"].replace(0, pd.NA)
            * 100
        )

    if {"successful_dribbles", "total_dribbles"}.issubset(season.columns):
        season["dribble_success_pct"] = (
            season["successful_dribbles"]
            / season["total_dribbles"].replace(0, pd.NA)
            * 100
        )

    eligible = season[
        (season["minutes"] >= MIN_MINUTES)
        & season["position_group"].notna()
    ].copy()

    metric_cols = [
        col
        for col in eligible.columns
        if col.endswith("_per90")
        or col.endswith("_pct")
    ]

    for metric in metric_cols:
        if not eligible[metric].notna().any():
            continue

        eligible[f"{metric}_percentile"] = (
            eligible.groupby(
                ["league_slug", "position_group"]
            )[metric]
            .rank(
                pct=True,
                method="average",
            )
            * 100.0
        )

    return season, eligible


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(
        "EUROPE 2025/26 — OFFLINE CACHE-ONLY PLAYER POOL".center(120)
    )
    print("=" * 120)
    print("ZERO API REQUESTS")

    matches, player_match, avg_positions, metadata = extract_from_cache()

    print()
    print("=" * 120)
    print("CACHE SUMMARY".center(120))
    print("=" * 120)

    print(f"Matches: {len(matches):,}")
    print(f"Player-match rows: {len(player_match):,}")
    print(f"Average-position rows: {len(avg_positions):,}")

    matches.to_parquet(OUTPUT_MATCHES, index=False)
    player_match.to_parquet(OUTPUT_PLAYER_MATCH, index=False)

    position_history = build_position_history(avg_positions)
    position_history.to_parquet(OUTPUT_AVG_POS, index=False)

    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    season, percentiles = build_season_tables(
        player_match,
        position_history,
    )

    season.to_parquet(OUTPUT_SEASON, index=False)
    percentiles.to_parquet(OUTPUT_PERCENTILES, index=False)

    print()
    print("=" * 120)
    print("FINAL PLAYER POOL".center(120))
    print("=" * 120)

    print(f"Season player-team rows: {len(season):,}")
    print(
        f"Eligible >= {MIN_MINUTES} minutes: "
        f"{len(percentiles):,}"
    )

    if not percentiles.empty:
        summary = (
            percentiles.groupby(
                ["league_slug", "position_group"]
            )
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


if __name__ == "__main__":
    main()
