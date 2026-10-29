from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bsd"

SEASON_METRICS_FILE = PROCESSED_DIR / "player_season_metrics.parquet"
OUTPUT_FILE = RAW_DIR / "players.parquet"
RAW_JSON_FILE = RAW_DIR / "players_raw.json"

BASE_URL = "https://sports.bzzoiro.com/api/v2"
REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.20
MAX_RETRIES = 4

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("BSD_API_KEY")

if not API_KEY:
    raise RuntimeError("BSD_API_KEY not found in project .env")

session = requests.Session()
session.headers.update({
    "Authorization": f"Token {API_KEY}",
    "Accept": "application/json",
})


def request_json(endpoint: str) -> dict:
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            time.sleep(REQUEST_SLEEP)
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"Request failed ({attempt}/{MAX_RETRIES}) for {url}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error


def first_value(payload: dict, candidates: list[str]):
    for key in candidates:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return None


def normalize_player_profile(profile: dict) -> dict:
    flat = pd.json_normalize(profile, sep="_").iloc[0].to_dict()

    return {
        "player_id": first_value(flat, ["id", "player_id"]),
        "player_name": first_value(flat, ["name", "player_name", "full_name"]),
        "position": first_value(flat, ["position", "primary_position"]),
        "date_of_birth": first_value(flat, ["date_of_birth", "dob", "birth_date"]),
        "height_cm": first_value(flat, ["height", "height_cm"]),
        "preferred_foot": first_value(flat, ["preferred_foot", "foot"]),
        "shirt_number": first_value(flat, ["shirt_number", "number"]),
        "nationality": first_value(flat, ["nationality", "country", "country_name"]),
        "nationality_code": first_value(
            flat, ["nationality_code", "country_code"]
        ),
        "market_value_eur": first_value(
            flat, ["market_value_eur", "market_value", "market_value_value"]
        ),
        "team_id": first_value(
            flat,
            ["team_id", "current_team_id", "team_id_id", "current_team_id_id"],
        ),
        "team_name": first_value(
            flat,
            [
                "team_name",
                "current_team_name",
                "team_name_name",
                "current_team_name_name",
            ],
        ),
        "contract_until": first_value(
            flat, ["contract_until", "contract_end", "contract_end_date"]
        ),
    }


def main():
    if not SEASON_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing season metrics file:\n{SEASON_METRICS_FILE}"
        )

    season = pd.read_parquet(SEASON_METRICS_FILE)

    if "player_id" not in season.columns:
        raise ValueError("player_season_metrics.parquet does not contain player_id.")

    player_ids = (
        season["player_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if OUTPUT_FILE.exists():
        existing = pd.read_parquet(OUTPUT_FILE)

    existing_ids = set()
    if not existing.empty and "player_id" in existing.columns:
        existing_ids = set(
            existing["player_id"].dropna().astype(int).tolist()
        )

    new_rows = []
    raw_profiles = []

    print(f"Season players: {len(player_ids):,}")
    print(f"Already cached: {len(existing_ids):,}")

    for number, player_id in enumerate(player_ids, start=1):
        if player_id in existing_ids:
            continue

        print(f"[{number:>3}/{len(player_ids)}] player {player_id}")

        try:
            profile = request_json(f"/players/{player_id}/")
            raw_profiles.append(profile)

            row = normalize_player_profile(profile)
            if row["player_id"] is None:
                row["player_id"] = player_id

            new_rows.append(row)

            print(
                f"  {row.get('player_name')} | "
                f"{row.get('position')} | "
                f"{row.get('team_name')}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")

    new_df = pd.DataFrame(new_rows)

    if existing.empty:
        players = new_df
    elif new_df.empty:
        players = existing
    else:
        players = pd.concat([existing, new_df], ignore_index=True)

    if players.empty:
        raise RuntimeError("No player metadata was extracted.")

    players = (
        players.drop_duplicates(subset=["player_id"], keep="last")
        .sort_values("player_id")
        .reset_index(drop=True)
    )

    if "date_of_birth" in players.columns:
        players["date_of_birth"] = pd.to_datetime(
            players["date_of_birth"], errors="coerce"
        )

    if "height_cm" in players.columns:
        players["height_cm"] = pd.to_numeric(
            players["height_cm"], errors="coerce"
        )

    if "market_value_eur" in players.columns:
        players["market_value_eur"] = pd.to_numeric(
            players["market_value_eur"], errors="coerce"
        )

    players.to_parquet(OUTPUT_FILE, index=False)

    if raw_profiles:
        with open(RAW_JSON_FILE, "w", encoding="utf-8") as file:
            json.dump(
                raw_profiles,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    print()
    print(f"Saved {len(players):,} player profiles")
    print(OUTPUT_FILE)

    print()
    print("Metadata coverage:")

    for column in [
        "player_name",
        "position",
        "date_of_birth",
        "height_cm",
        "preferred_foot",
        "market_value_eur",
        "team_name",
    ]:
        if column in players.columns:
            print(
                f"  {column:<20} "
                f"{players[column].notna().sum():>4}/{len(players)}"
            )


if __name__ == "__main__":
    main()
