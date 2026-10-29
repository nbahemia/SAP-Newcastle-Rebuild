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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "bsd"

MATCHES_FILE = RAW_DIR / "matches.parquet"
OUTPUT_FILE = RAW_DIR / "lineups.parquet"
RAW_JSON_FILE = RAW_DIR / "lineups_raw.json"

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


# ---------------------------------------------------------
# HTTP
# ---------------------------------------------------------

def request_json(endpoint: str) -> dict[str, Any]:
    url = (
        endpoint
        if endpoint.startswith("http")
        else f"{BASE_URL}{endpoint}"
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            time.sleep(
                REQUEST_SLEEP
            )

            return response.json()

        except (
            requests.RequestException,
            json.JSONDecodeError,
        ) as exc:

            last_error = exc

            print(
                f"Request failed "
                f"({attempt}/{MAX_RETRIES}) "
                f"for {url}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    attempt * 2
                )

    raise RuntimeError(
        f"Request failed after "
        f"{MAX_RETRIES} attempts: "
        f"{url}"
    ) from last_error


# ---------------------------------------------------------
# Generic recursive helpers
# ---------------------------------------------------------

PLAYER_ID_KEYS = {
    "player_id",
    "playerid",
}

ID_KEYS = {
    "id",
}

POSITION_KEYS = [
    "position",
    "position_name",
    "position_short",
    "pos",
    "slot",
    "role",
    "formation_position",
    "lineup_position",
]

PLAYER_NAME_KEYS = [
    "player_name",
    "name",
    "full_name",
]

TEAM_ID_KEYS = [
    "team_id",
    "teamid",
]

TEAM_NAME_KEYS = [
    "team_name",
]

STARTER_KEYS = [
    "starter",
    "is_starter",
    "starting",
    "starting_xi",
    "first_eleven",
]

SHIRT_KEYS = [
    "shirt_number",
    "jersey_number",
    "number",
]


def normalized_key(
    key: str,
) -> str:

    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def first_present(
    mapping: dict,
    candidates: list[str],
):
    normalized = {
        normalized_key(key): value
        for key, value in mapping.items()
    }

    for candidate in candidates:

        candidate = normalized_key(
            candidate
        )

        if candidate in normalized:

            value = normalized[
                candidate
            ]

            if (
                value is not None
                and value != ""
            ):
                return value

    return None


def looks_like_player_record(
    node: dict,
) -> bool:
    """
    Conservative test for a lineup player record.

    Prefer explicit player_id. If absent, allow an id only when the record
    also has other player-like fields such as position, shirt number, or
    starter status.
    """

    keys = {
        normalized_key(key)
        for key in node.keys()
    }

    if keys & PLAYER_ID_KEYS:
        return True

    has_id = bool(
        keys & ID_KEYS
    )

    has_player_signal = any(
        normalized_key(key) in keys
        for key in (
            POSITION_KEYS
            + SHIRT_KEYS
            + STARTER_KEYS
        )
    )

    return (
        has_id
        and has_player_signal
    )


def extract_team_context(
    node: dict,
    inherited_team_id=None,
    inherited_team_name=None,
):
    team_id = first_present(
        node,
        TEAM_ID_KEYS,
    )

    team_name = first_present(
        node,
        TEAM_NAME_KEYS,
    )

    # Handle nested team object.
    team_obj = node.get(
        "team"
    )

    if isinstance(
        team_obj,
        dict,
    ):
        if team_id is None:
            team_id = first_present(
                team_obj,
                ["id", "team_id"],
            )

        if team_name is None:
            team_name = first_present(
                team_obj,
                ["name", "team_name"],
            )

    return (
        team_id
        if team_id is not None
        else inherited_team_id,
        team_name
        if team_name is not None
        else inherited_team_name,
    )


def extract_player_row(
    node: dict,
    event_id: int,
    inherited_team_id=None,
    inherited_team_name=None,
    path: str = "",
) -> dict | None:

    if not looks_like_player_record(
        node
    ):
        return None

    player_id = first_present(
        node,
        [
            "player_id",
            "playerId",
        ],
    )

    # Sometimes player data lives inside a nested player object.
    player_obj = node.get(
        "player"
    )

    if isinstance(
        player_obj,
        dict,
    ):
        if player_id is None:
            player_id = first_present(
                player_obj,
                [
                    "id",
                    "player_id",
                ],
            )

    # Only use top-level id as a final fallback.
    if player_id is None:
        player_id = first_present(
            node,
            ["id"],
        )

    if player_id is None:
        return None

    team_id, team_name = (
        extract_team_context(
            node,
            inherited_team_id,
            inherited_team_name,
        )
    )

    player_name = first_present(
        node,
        PLAYER_NAME_KEYS,
    )

    if (
        player_name is None
        and isinstance(
            player_obj,
            dict,
        )
    ):
        player_name = first_present(
            player_obj,
            PLAYER_NAME_KEYS,
        )

    position = first_present(
        node,
        POSITION_KEYS,
    )

    if (
        position is None
        and isinstance(
            player_obj,
            dict,
        )
    ):
        position = first_present(
            player_obj,
            POSITION_KEYS,
        )

    starter = first_present(
        node,
        STARTER_KEYS,
    )

    shirt_number = first_present(
        node,
        SHIRT_KEYS,
    )

    return {
        "event_id": event_id,
        "player_id": player_id,
        "team_id": team_id,
        "team_name": team_name,
        "player_name": player_name,
        "observed_position": position,
        "starter": starter,
        "shirt_number": shirt_number,
        "source_path": path,
    }


def walk_lineup_payload(
    node: Any,
    event_id: int,
    rows: list[dict],
    path: str = "root",
    inherited_team_id=None,
    inherited_team_name=None,
):
    """
    Recursively walk arbitrary BSD lineup JSON and capture player-like dicts.
    """

    if isinstance(
        node,
        dict,
    ):

        team_id, team_name = (
            extract_team_context(
                node,
                inherited_team_id,
                inherited_team_name,
            )
        )

        row = extract_player_row(
            node=node,
            event_id=event_id,
            inherited_team_id=team_id,
            inherited_team_name=team_name,
            path=path,
        )

        if row is not None:
            rows.append(
                row
            )

        for key, value in node.items():

            walk_lineup_payload(
                node=value,
                event_id=event_id,
                rows=rows,
                path=f"{path}.{key}",
                inherited_team_id=team_id,
                inherited_team_name=team_name,
            )

    elif isinstance(
        node,
        list,
    ):

        for index, value in enumerate(
            node
        ):

            walk_lineup_payload(
                node=value,
                event_id=event_id,
                rows=rows,
                path=f"{path}[{index}]",
                inherited_team_id=inherited_team_id,
                inherited_team_name=inherited_team_name,
            )


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

def normalize_boolean(
    value,
):

    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(
        value
    ):
        return pd.NA

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
        "starter",
        "starting",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "bench",
        "substitute",
    }:
        return False

    return pd.NA


def clean_lineups(
    rows: list[dict],
) -> pd.DataFrame:

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    for column in [
        "event_id",
        "player_id",
        "team_id",
        "shirt_number",
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
        ]
    ).copy()

    df[
        "event_id"
    ] = df[
        "event_id"
    ].astype(int)

    df[
        "player_id"
    ] = df[
        "player_id"
    ].astype(int)

    if "starter" in df.columns:
        df[
            "starter"
        ] = df[
            "starter"
        ].apply(
            normalize_boolean
        )

    # Recursive traversal may see the same player record from multiple
    # nested structures. Prefer the row with an observed position and team.
    df[
        "_position_present"
    ] = df[
        "observed_position"
    ].notna()

    df[
        "_team_present"
    ] = df[
        "team_id"
    ].notna()

    df = (
        df.sort_values(
            [
                "_position_present",
                "_team_present",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "event_id",
                "player_id",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_position_present",
                "_team_present",
            ]
        )
        .sort_values(
            [
                "event_id",
                "team_id",
                "player_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not MATCHES_FILE.exists():

        raise FileNotFoundError(
            f"Missing matches file:\n"
            f"{MATCHES_FILE}"
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
        f"Matches to inspect: "
        f"{len(event_ids):,}"
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
                f"/events/{event_id}/lineups/"
            )

            raw_payloads[
                str(event_id)
            ] = payload

            rows = []

            walk_lineup_payload(
                node=payload,
                event_id=event_id,
                rows=rows,
            )

            all_rows.extend(
                rows
            )

            print(
                f"  candidate player records: "
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

    lineups = clean_lineups(
        all_rows
    )

    if lineups.empty:

        raise RuntimeError(
            "No player lineup records could be "
            "parsed. Inspect lineups_raw.json."
        )

    lineups.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 88)
    print(
        "BSD LINEUP EXTRACTION SUMMARY"
        .center(88)
    )
    print("=" * 88)

    print(
        f"Unique matches: "
        f"{lineups['event_id'].nunique():,}"
    )

    print(
        f"Lineup rows: "
        f"{len(lineups):,}"
    )

    print(
        f"Unique players: "
        f"{lineups['player_id'].nunique():,}"
    )

    print(
        f"Rows with observed position: "
        f"{lineups['observed_position'].notna().sum():,}"
    )

    print()
    print(
        "Observed positions:"
    )

    print(
        lineups[
            "observed_position"
        ]
        .value_counts(
            dropna=False
        )
        .head(30)
    )

    print()
    print(
        f"Saved:\n"
        f"  {OUTPUT_FILE}\n"
        f"  {RAW_JSON_FILE}"
    )


if __name__ == "__main__":
    main()
