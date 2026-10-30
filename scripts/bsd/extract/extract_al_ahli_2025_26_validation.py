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

LEAGUE_ID = 17
SEASON_ID = 339
TARGET_TEAM_NAME = "Al-Ahli"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd" / "saudi_2025_26"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_FILE = RAW_DIR / "matches.parquet"
AVERAGE_POSITIONS_FILE = RAW_DIR / "average_positions.parquet"
PLAYERS_FILE = RAW_DIR / "players.parquet"
PLAYERS_RAW_FILE = RAW_DIR / "players_raw.json"
POSITION_HISTORY_FILE = RAW_DIR / "player_position_history.parquet"
AL_AHLI_HISTORY_FILE = RAW_DIR / "al_ahli_player_position_history.parquet"
META_FILE = RAW_DIR / "metadata.json"

HEADERS = {"Authorization": f"Token {API_KEY}"}
REQUEST_SLEEP_SECONDS = 0.08

BROAD_POSITION_MAP = {
    "G": "GK", "GK": "GK", "GOALKEEPER": "GK",
    "D": "DEF", "DEF": "DEF", "DEFENDER": "DEF",
    "M": "MID", "MID": "MID", "MIDFIELDER": "MID",
    "F": "FWD", "FW": "FWD", "FWD": "FWD",
    "FORWARD": "FWD", "ATTACKER": "FWD",
}

DEFENDER_WIDE_DISTANCE = 24.0
FULLBACK_MIN_X = 38.0
MIDFIELDER_WIDE_DISTANCE = 20.0
FORWARD_WIDE_DISTANCE = 17.0
ATTACKING_MID_X = 60.0
SIDE_LEFT_MAX = 38.0
SIDE_RIGHT_MIN = 62.0


def api_get(path: str, params: dict | None = None):
    r = requests.get(
        f"{BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(
            f"BSD request failed {r.status_code}\n"
            f"{r.url}\n{r.text[:1000]}"
        )
    time.sleep(REQUEST_SLEEP_SECONDS)
    return r.json()


def paginated_get(path: str, params: dict | None = None):
    params = dict(params or {})
    params["limit"] = 200
    params["offset"] = 0
    out = []

    while True:
        payload = api_get(path, params=params)

        if isinstance(payload, list):
            out.extend(payload)
            break

        rows = payload.get("results", []) if isinstance(payload, dict) else []
        out.extend(rows)

        if not isinstance(payload, dict) or not payload.get("next"):
            break

        params["offset"] += 200

    return out


def extract_matches():
    events = paginated_get(
        "/events/",
        params={
            "league_id": LEAGUE_ID,
            "season_id": SEASON_ID,
            "status": "finished",
        },
    )

    rows = []

    for event in events:
        home = event.get("home_team")
        away = event.get("away_team")

        home_id = event.get("home_team_id")
        away_id = event.get("away_team_id")
        home_name = event.get("home_team_name")
        away_name = event.get("away_team_name")

        if isinstance(home, dict):
            home_id = home_id or home.get("id")
            home_name = home_name or home.get("name")
        elif home_name is None:
            home_name = home

        if isinstance(away, dict):
            away_id = away_id or away.get("id")
            away_name = away_name or away.get("name")
        elif away_name is None:
            away_name = away

        rows.append(
            {
                "event_id": event.get("id"),
                "event_date": event.get("event_date"),
                "league_id": event.get("league_id") or LEAGUE_ID,
                "season_id": event.get("season_id") or SEASON_ID,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": home_name,
                "away_team": away_name,
                "status": event.get("status"),
            }
        )

    return pd.DataFrame(rows)


def recursive_find_average_positions(obj, event_id, match):
    rows = []

    def walk(node, context_key=None, inherited_team_id=None, inherited_home=None):
        if isinstance(node, dict):
            team_id = node.get("team_id") or inherited_team_id
            is_home = (
                node.get("home")
                if isinstance(node.get("home"), bool)
                else inherited_home
            )

            if context_key is not None and str(context_key).lower() in {"home", "home_team"}:
                is_home = True
                if pd.notna(match.get("home_team_id")):
                    team_id = int(match["home_team_id"])

            elif context_key is not None and str(context_key).lower() in {"away", "away_team"}:
                is_home = False
                if pd.notna(match.get("away_team_id")):
                    team_id = int(match["away_team_id"])

            player_obj = node.get("player") if isinstance(node.get("player"), dict) else {}
            player_id = node.get("player_id") or player_obj.get("id")
            player_name = node.get("player_name") or node.get("name") or player_obj.get("name")

            pos = node.get("pos")
            x = pos.get("x") if isinstance(pos, dict) else None
            y = pos.get("y") if isinstance(pos, dict) else None

            if x is None:
                x = node.get("avg_x")
                if x is None:
                    x = node.get("x")

            if y is None:
                y = node.get("avg_y")
                if y is None:
                    y = node.get("y")

            if player_id is not None and x is not None and y is not None:
                rows.append(
                    {
                        "event_id": event_id,
                        "team_id": team_id,
                        "is_home": is_home,
                        "player_id": player_id,
                        "player_name": player_name,
                        "avg_x": x,
                        "avg_y": y,
                        "touches": node.get("n") or node.get("touches"),
                    }
                )

            for key, value in node.items():
                walk(value, key, team_id, is_home)

        elif isinstance(node, list):
            for item in node:
                walk(item, context_key, inherited_team_id, inherited_home)

    walk(obj)

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["avg_x"] = pd.to_numeric(df["avg_x"], errors="coerce")
    df["avg_y"] = pd.to_numeric(df["avg_y"], errors="coerce")
    df = df.dropna(subset=["player_id", "avg_x", "avg_y"])
    df["player_id"] = df["player_id"].astype(int)
    df = df.drop_duplicates(["event_id", "player_id"], keep="first")

    return df.to_dict(orient="records")


def extract_average_positions(matches: pd.DataFrame):
    rows = []
    covered = 0

    for i, match in matches.iterrows():
        event_id = int(match["event_id"])
        payload = api_get(f"/events/{event_id}/stats/")
        event_rows = recursive_find_average_positions(payload, event_id, match)

        if event_rows:
            covered += 1
            rows.extend(event_rows)

        print(
            f"[{i+1}/{len(matches)}] event {event_id} "
            f"| avg-pos rows={len(event_rows)}"
        )

    return pd.DataFrame(rows), covered


def extract_player_profiles(avg: pd.DataFrame):
    ids = sorted(avg["player_id"].dropna().astype(int).unique().tolist())
    rows = []
    raw = {}

    for i, player_id in enumerate(ids, start=1):
        payload = api_get(f"/players/{player_id}/")
        raw[str(player_id)] = payload

        if not isinstance(payload, dict):
            continue

        team = payload.get("team") if isinstance(payload.get("team"), dict) else {}

        rows.append(
            {
                "player_id": player_id,
                "player_name": payload.get("name") or payload.get("player_name"),
                "position": payload.get("position") or payload.get("pos"),
                "profile_team_id": team.get("id") or payload.get("team_id"),
                "profile_team_name": team.get("name") or payload.get("team_name"),
            }
        )

        print(
            f"[{i}/{len(ids)}] {player_id} "
            f"| {rows[-1]['player_name']} "
            f"| pos={rows[-1]['position']}"
        )

    PLAYERS_RAW_FILE.write_text(
        json.dumps(raw, indent=2, default=str),
        encoding="utf-8",
    )

    return pd.DataFrame(rows)


def normalize_broad_position(value):
    if pd.isna(value):
        return None
    return BROAD_POSITION_MAP.get(str(value).strip().upper())


def infer_attack_direction(group: pd.DataFrame):
    defenders = group[
        (group["broad_position"] == "DEF")
        & group["avg_x"].notna()
    ]

    forwards = group[
        (group["broad_position"] == "FWD")
        & group["avg_x"].notna()
    ]

    if not defenders.empty and not forwards.empty:
        return "right" if forwards["avg_x"].mean() > defenders["avg_x"].mean() else "left"

    return "right"


def classify_tactical_role(broad_position, x_attacking, y_attacking):
    if pd.isna(broad_position) or pd.isna(x_attacking) or pd.isna(y_attacking):
        return None

    x = float(x_attacking)
    y = float(y_attacking)
    wide_distance = abs(y - 50.0)

    if broad_position == "GK":
        return "GK"

    if broad_position == "DEF":
        if wide_distance >= DEFENDER_WIDE_DISTANCE and x >= FULLBACK_MIN_X:
            return "FB/WB"
        return "CB"

    if broad_position == "MID":
        if wide_distance >= MIDFIELDER_WIDE_DISTANCE:
            return "W"
        if x >= ATTACKING_MID_X:
            return "AM"
        return "DM/CM"

    if broad_position == "FWD":
        if wide_distance >= FORWARD_WIDE_DISTANCE:
            return "W"
        return "ST"

    return None


def classify_side(position_group, y_attacking):
    if pd.isna(position_group) or pd.isna(y_attacking):
        return None

    if position_group in {"GK", "ST", "AM"}:
        return "CENTRAL"

    y = float(y_attacking)

    if y < SIDE_LEFT_MAX:
        return "RIGHT"
    if y > SIDE_RIGHT_MIN:
        return "LEFT"
    return "CENTRAL"


def build_tactical_position(position_group, side):
    if position_group == "GK":
        return "GK"
    if position_group == "ST":
        return "ST"
    if position_group == "AM":
        return "AM"

    if position_group == "CB":
        return "LCB" if side == "LEFT" else "RCB" if side == "RIGHT" else "CB"

    if position_group == "FB/WB":
        return "LB/WB" if side == "LEFT" else "RB/WB" if side == "RIGHT" else "FB/WB"

    if position_group == "DM/CM":
        return "LCM/DM" if side == "LEFT" else "RCM/DM" if side == "RIGHT" else "CM/DM"

    if position_group == "W":
        return "LW" if side == "LEFT" else "RW" if side == "RIGHT" else "W"

    return position_group


def build_position_history(avg: pd.DataFrame, players: pd.DataFrame):
    history = avg.merge(
        players[["player_id", "player_name", "position"]],
        on="player_id",
        how="left",
        suffixes=("_avg", "_profile"),
        validate="many_to_one",
    )

    history["player_name"] = history["player_name_avg"].fillna(history["player_name_profile"])
    history["broad_position"] = history["position"].apply(normalize_broad_position)

    history["attack_direction"] = None
    history["x_attacking"] = np.nan
    history["y_attacking"] = np.nan

    for (event_id, team_id), group in (
        history.dropna(subset=["team_id"])
        .groupby(["event_id", "team_id"])
    ):
        direction = infer_attack_direction(group)
        idx = group.index

        history.loc[idx, "attack_direction"] = direction

        if direction == "right":
            history.loc[idx, "x_attacking"] = group["avg_x"]
            history.loc[idx, "y_attacking"] = group["avg_y"]
        else:
            history.loc[idx, "x_attacking"] = 100.0 - group["avg_x"]
            history.loc[idx, "y_attacking"] = 100.0 - group["avg_y"]

    history["position_group"] = history.apply(
        lambda r: classify_tactical_role(
            r["broad_position"],
            r["x_attacking"],
            r["y_attacking"],
        ),
        axis=1,
    )

    history["position_side"] = history.apply(
        lambda r: classify_side(
            r["position_group"],
            r["y_attacking"],
        ),
        axis=1,
    )

    history["tactical_position"] = history.apply(
        lambda r: build_tactical_position(
            r["position_group"],
            r["position_side"],
        ),
        axis=1,
    )

    return history


def main():
    print("=" * 100)
    print("SAUDI PRO LEAGUE 2025/26 — POSITION HISTORY EXTRACTION".center(100))
    print("=" * 100)

    matches = extract_matches()
    matches.to_parquet(MATCHES_FILE, index=False)

    print(f"\nFinished league matches: {len(matches)}")

    avg, covered = extract_average_positions(matches)

    if avg.empty:
        raise RuntimeError("No average-position data returned.")

    avg.to_parquet(AVERAGE_POSITIONS_FILE, index=False)

    players = extract_player_profiles(avg)
    players.to_parquet(PLAYERS_FILE, index=False)

    history = build_position_history(avg, players)
    history.to_parquet(POSITION_HISTORY_FILE, index=False)

    team_lookup = pd.concat(
        [
            matches[["home_team_id", "home_team"]].rename(
                columns={
                    "home_team_id": "team_id",
                    "home_team": "team_name",
                }
            ),
            matches[["away_team_id", "away_team"]].rename(
                columns={
                    "away_team_id": "team_id",
                    "away_team": "team_name",
                }
            ),
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()

    history = history.merge(
        team_lookup,
        on="team_id",
        how="left",
    )

    history.to_parquet(POSITION_HISTORY_FILE, index=False)

    al_ahli = history[
        history["team_name"]
        .astype(str)
        .str.contains(
            "Al-Ahli",
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    if al_ahli.empty:
        # fallback for provider naming variants
        al_ahli = history[
            history["team_name"]
            .astype(str)
            .str.contains(
                "Ahli",
                case=False,
                na=False,
                regex=False,
            )
        ].copy()

    if al_ahli.empty:
        raise RuntimeError("Could not isolate Al-Ahli rows from league-wide history.")

    al_ahli.to_parquet(AL_AHLI_HISTORY_FILE, index=False)

    meta = {
        "league_id": LEAGUE_ID,
        "season_id": SEASON_ID,
        "finished_matches": len(matches),
        "matches_with_average_positions": covered,
        "average_position_coverage_pct": round(
            covered / len(matches) * 100.0,
            1,
        ),
        "players_with_profiles": len(players),
        "thresholds": {
            "defender_wide_distance": DEFENDER_WIDE_DISTANCE,
            "fullback_min_x": FULLBACK_MIN_X,
            "midfielder_wide_distance": MIDFIELDER_WIDE_DISTANCE,
            "forward_wide_distance": FORWARD_WIDE_DISTANCE,
            "attacking_mid_x": ATTACKING_MID_X,
            "side_left_max": SIDE_LEFT_MAX,
            "side_right_min": SIDE_RIGHT_MIN,
        },
    }

    META_FILE.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SAVED".center(100))
    print("=" * 100)

    for path in [
        MATCHES_FILE,
        AVERAGE_POSITIONS_FILE,
        PLAYERS_FILE,
        POSITION_HISTORY_FILE,
        AL_AHLI_HISTORY_FILE,
        META_FILE,
    ]:
        print(path)


if __name__ == "__main__":
    main()
