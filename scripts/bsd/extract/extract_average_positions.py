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

MATCHES_FILE = RAW_DIR / "matches.parquet"
OUTPUT_FILE = RAW_DIR / "average_positions.parquet"
RAW_JSON_FILE = RAW_DIR / "average_positions_raw.json"

BASE_URL = "https://sports.bzzoiro.com/api/v2"

REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.20
MAX_RETRIES = 4

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("BSD_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "BSD_API_KEY not found. Add BSD_API_KEY=... to the project .env file."
    )

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Token {API_KEY}",
        "Accept": "application/json",
    }
)


def request_json(endpoint: str) -> dict[str, Any]:
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
                f"Request failed ({attempt}/{MAX_RETRIES}) "
                f"for {url}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error


def normalize_key(key: str) -> str:
    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def first_present(mapping: dict, candidates: list[str]):
    normalized = {
        normalize_key(key): value
        for key, value in mapping.items()
    }

    for candidate in candidates:
        candidate = normalize_key(candidate)

        if candidate in normalized:
            value = normalized[candidate]

            if value is not None and value != "":
                return value

    return None


def extract_xy(node: dict):
    """
    Find an x/y coordinate pair in a player average-position record.
    Supports several likely provider field names.
    """
    x = first_present(
        node,
        [
            "x",
            "avg_x",
            "average_x",
            "position_x",
            "coord_x",
        ],
    )

    y = first_present(
        node,
        [
            "y",
            "avg_y",
            "average_y",
            "position_y",
            "coord_y",
        ],
    )

    if x is not None and y is not None:
        return x, y

    for nested_key in [
        "position",
        "average_position",
        "coordinates",
        "coord",
        "point",
    ]:
        nested = node.get(nested_key)

        if isinstance(nested, dict):
            x = first_present(
                nested,
                [
                    "x",
                    "avg_x",
                    "average_x",
                    "position_x",
                ],
            )

            y = first_present(
                nested,
                [
                    "y",
                    "avg_y",
                    "average_y",
                    "position_y",
                ],
            )

            if x is not None and y is not None:
                return x, y

    return None, None


def extract_player_id(node: dict):
    player_id = first_present(
        node,
        [
            "player_id",
            "playerId",
        ],
    )

    if player_id is not None:
        return player_id

    player = node.get("player")

    if isinstance(player, dict):
        return first_present(
            player,
            [
                "id",
                "player_id",
            ],
        )

    return None


def extract_team_id(node: dict, inherited_team_id=None):
    team_id = first_present(
        node,
        [
            "team_id",
            "teamId",
        ],
    )

    team = node.get("team")

    if team_id is None and isinstance(team, dict):
        team_id = first_present(
            team,
            [
                "id",
                "team_id",
            ],
        )

    return (
        team_id
        if team_id is not None
        else inherited_team_id
    )


def walk_average_positions(
    node: Any,
    event_id: int,
    rows: list[dict],
    path: str = "average_positions",
    inherited_team_id=None,
):
    if isinstance(node, dict):
        team_id = extract_team_id(
            node,
            inherited_team_id,
        )

        player_id = extract_player_id(node)
        x, y = extract_xy(node)

        if (
            player_id is not None
            and x is not None
            and y is not None
        ):
            rows.append(
                {
                    "event_id": event_id,
                    "player_id": player_id,
                    "team_id": team_id,
                    "avg_x": x,
                    "avg_y": y,
                    "source_path": path,
                }
            )

        for key, value in node.items():
            walk_average_positions(
                node=value,
                event_id=event_id,
                rows=rows,
                path=f"{path}.{key}",
                inherited_team_id=team_id,
            )

    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk_average_positions(
                node=value,
                event_id=event_id,
                rows=rows,
                path=f"{path}[{index}]",
                inherited_team_id=inherited_team_id,
            )


def clean_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for column in [
        "event_id",
        "player_id",
        "team_id",
        "avg_x",
        "avg_y",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "event_id",
            "player_id",
            "avg_x",
            "avg_y",
        ]
    ).copy()

    df["event_id"] = df["event_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    # Keep sensible pitch percentages only.
    df = df[
        df["avg_x"].between(0, 100)
        & df["avg_y"].between(0, 100)
    ].copy()

    # Recursive parsing may find the same object more than once.
    df = (
        df.drop_duplicates(
            subset=[
                "event_id",
                "player_id",
            ],
            keep="first",
        )
        .sort_values(
            [
                "event_id",
                "team_id",
                "player_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df


def main():
    if not MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"Missing matches file:\n{MATCHES_FILE}"
        )

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    event_ids = (
        matches["id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    raw_payloads = {}

    print(
        f"Matches to inspect: {len(event_ids):,}"
    )

    for number, event_id in enumerate(
        event_ids,
        start=1,
    ):
        print(
            f"[{number:>3}/{len(event_ids)}] "
            f"event {event_id}"
        )

        try:
            payload = request_json(
                f"/events/{event_id}/stats/"
            )

            average_positions = payload.get(
                "average_positions"
            )

            raw_payloads[
                str(event_id)
            ] = average_positions

            rows = []

            if average_positions is not None:
                walk_average_positions(
                    node=average_positions,
                    event_id=event_id,
                    rows=rows,
                )

            all_rows.extend(rows)

            print(
                f"  parsed average-position rows: "
                f"{len(rows)}"
            )

        except Exception as exc:
            print(
                f"  ERROR: {exc}"
            )

    with open(
        RAW_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            raw_payloads,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    positions = clean_rows(
        all_rows
    )

    if positions.empty:
        raise RuntimeError(
            "No average-position rows were parsed. "
            "Inspect average_positions_raw.json for the real schema."
        )

    positions.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 88)
    print(
        "BSD AVERAGE POSITION EXTRACTION SUMMARY"
        .center(88)
    )
    print("=" * 88)

    print(
        f"Matches with positions: "
        f"{positions['event_id'].nunique():,}"
    )

    print(
        f"Average-position rows: "
        f"{len(positions):,}"
    )

    print(
        f"Unique players: "
        f"{positions['player_id'].nunique():,}"
    )

    print()
    print(
        f"x range: "
        f"{positions['avg_x'].min():.1f} "
        f"to "
        f"{positions['avg_x'].max():.1f}"
    )

    print(
        f"y range: "
        f"{positions['avg_y'].min():.1f} "
        f"to "
        f"{positions['avg_y'].max():.1f}"
    )

    print()
    print(
        "Sample:"
    )

    print(
        positions
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved:\n"
        f"  {OUTPUT_FILE}\n"
        f"  {RAW_JSON_FILE}"
    )


if __name__ == "__main__":
    main()
