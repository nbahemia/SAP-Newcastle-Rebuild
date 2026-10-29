from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd"

BASE_URL = "https://sports.bzzoiro.com/api/v2"

LEAGUE_ID = 1
SEASON_ID = 1058

REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.20
MAX_RETRIES = 4

MATCHES_FILE = RAW_DIR / "matches.parquet"
PLAYER_STATS_FILE = RAW_DIR / "player_match_stats.parquet"
SHOTS_FILE = RAW_DIR / "shots.parquet"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("BSD_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "BSD_API_KEY not found. "
        "Add BSD_API_KEY=... to the project .env file."
    )

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Token {API_KEY}",
        "Accept": "application/json",
    }
)


def request_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> Any:
    url = (
        endpoint
        if endpoint.startswith("http")
        else f"{BASE_URL}{endpoint}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            time.sleep(REQUEST_SLEEP)
            return response.json()

        except (
            requests.RequestException,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc

            print(
                f"Request failed ({attempt}/{MAX_RETRIES}): {url}"
            )
            print(f"  {exc}")

            if attempt < MAX_RETRIES:
                sleep_seconds = attempt * 2
                print(f"  Retrying in {sleep_seconds}s...")
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"BSD request failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error


def get_paginated(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> list[dict]:
    params = params.copy() if params else {}
    params.setdefault("limit", 200)

    rows = []
    url = endpoint
    current_params = params

    while url:
        payload = request_json(
            url,
            params=current_params,
        )

        batch = payload.get("results", [])
        rows.extend(batch)

        print(
            f"  Retrieved {len(rows):,} / "
            f"{payload.get('count', len(rows)):,}"
        )

        url = payload.get("next")
        current_params = None

    return rows


def extract_matches() -> pd.DataFrame:
    print()
    print("=" * 88)
    print("EXTRACTING BSD PREMIER LEAGUE MATCHES".center(88))
    print("=" * 88)

    rows = get_paginated(
        "/events/",
        params={
            "league_id": LEAGUE_ID,
            "season_id": SEASON_ID,
            "status": "finished",
        },
    )

    if not rows:
        raise RuntimeError(
            "No finished Premier League matches were returned."
        )

    matches = pd.json_normalize(
        rows,
        sep="_",
    )

    for column in [
        "event_date",
        "start_time",
        "kickoff",
    ]:
        if column in matches.columns:
            matches[column] = pd.to_datetime(
                matches[column],
                errors="coerce",
                utc=True,
            )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches.to_parquet(
        MATCHES_FILE,
        index=False,
    )

    print()
    print(f"Saved {len(matches):,} matches")
    print(MATCHES_FILE)

    return matches


def get_availability(
    event_id: int,
) -> dict:
    return request_json(
        f"/events/{event_id}/availability/"
    )


def extract_player_stats(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 88)
    print("EXTRACTING BSD PLAYER-MATCH STATS".center(88))
    print("=" * 88)

    all_rows = []

    event_ids = (
        matches["id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    for number, event_id in enumerate(
        event_ids,
        start=1,
    ):
        print(
            f"[{number:>3}/{len(event_ids)}] event {event_id}"
        )

        try:
            availability = get_availability(event_id)
            available = availability.get("available", {})

            if not available.get(
                "player_stats",
                False,
            ):
                print("  player_stats unavailable")
                continue

            payload = request_json(
                f"/events/{event_id}/player-stats/"
            )

            rows = payload.get(
                "player_stats",
                []
            )

            if not rows:
                print("  no player rows")
                continue

            for row in rows:
                row = row.copy()
                row.setdefault(
                    "event_id",
                    event_id,
                )
                all_rows.append(row)

            print(f"  +{len(rows)} player rows")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

    if not all_rows:
        raise RuntimeError(
            "No BSD player-stat rows were extracted."
        )

    player_stats = pd.json_normalize(
        all_rows,
        sep="_",
    )

    player_stats.to_parquet(
        PLAYER_STATS_FILE,
        index=False,
    )

    print()
    print(
        f"Saved {len(player_stats):,} player-match rows"
    )
    print(PLAYER_STATS_FILE)

    return player_stats


def extract_shotmaps(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 88)
    print("EXTRACTING BSD SHOTMAPS".center(88))
    print("=" * 88)

    all_shots = []

    event_ids = (
        matches["id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    for number, event_id in enumerate(
        event_ids,
        start=1,
    ):
        print(
            f"[{number:>3}/{len(event_ids)}] event {event_id}"
        )

        try:
            availability = get_availability(event_id)
            available = availability.get("available", {})

            if not available.get(
                "shotmap",
                False,
            ):
                print("  shotmap unavailable")
                continue

            payload = request_json(
                f"/events/{event_id}/stats/"
            )

            shots = payload.get(
                "shotmap",
                []
            )

            if not shots:
                print("  no shots")
                continue

            for shot_index, shot in enumerate(
                shots,
                start=1,
            ):
                row = shot.copy()
                row["event_id"] = event_id
                row["shot_index"] = shot_index
                all_shots.append(row)

            print(f"  +{len(shots)} shots")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

    if not all_shots:
        raise RuntimeError(
            "No BSD shotmap rows were extracted."
        )

    shots = pd.json_normalize(
        all_shots,
        sep="_",
    )

    shots.to_parquet(
        SHOTS_FILE,
        index=False,
    )

    print()
    print(f"Saved {len(shots):,} shots")
    print(SHOTS_FILE)

    return shots


def print_summary(
    matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    shots: pd.DataFrame,
):
    print()
    print("=" * 88)
    print("BSD EXTRACTION SUMMARY".center(88))
    print("=" * 88)

    print(f"Matches:              {len(matches):,}")
    print(
        f"Player-match rows:    {len(player_stats):,}"
    )
    print(f"Shot rows:            {len(shots):,}")

    if "player_id" in player_stats.columns:
        print(
            f"Unique players:       "
            f"{player_stats['player_id'].nunique():,}"
        )

    if "team_id" in player_stats.columns:
        print(
            f"Unique teams:         "
            f"{player_stats['team_id'].nunique():,}"
        )

    if "minutes_played" in player_stats.columns:
        minutes = (
            player_stats["minutes_played"]
            .fillna(0)
        )

        print(
            f"Rows minutes > 0:     "
            f"{(minutes > 0).sum():,}"
        )

        print(
            f"Unused squad rows:    "
            f"{(minutes <= 0).sum():,}"
        )

    print()
    print("Outputs:")
    print(f"  {MATCHES_FILE}")
    print(f"  {PLAYER_STATS_FILE}")
    print(f"  {SHOTS_FILE}")


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = extract_matches()

    player_stats = extract_player_stats(
        matches
    )

    shots = extract_shotmaps(
        matches
    )

    print_summary(
        matches=matches,
        player_stats=player_stats,
        shots=shots,
    )


if __name__ == "__main__":
    main()
