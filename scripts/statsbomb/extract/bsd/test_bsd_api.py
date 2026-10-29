import os
import json

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("BSD_API_KEY")

if not API_KEY:
    raise ValueError("BSD_API_KEY not found in .env")


BASE_URL = "https://sports.bzzoiro.com/api/v2"

HEADERS = {
    "Authorization": f"Token {API_KEY}"
}

SEASON_ID = 1058
LEAGUE_ID = 1


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def get_json(endpoint, params=None):

    response = requests.get(
        f"{BASE_URL}{endpoint}",
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# 1. Get finished Premier League matches
# ---------------------------------------------------------

matches = get_json(
    "/events/",
    params={
        "season_id": SEASON_ID,
        "league_id": LEAGUE_ID,
        "status": "finished",
        "limit": 10,
    },
)


results = matches.get(
    "results",
    []
)


print(
    f"Finished matches returned: "
    f"{len(results)}"
)


for index, match in enumerate(
    results,
    start=1,
):

    print(
        f"{index:>2}. "
        f"ID {match.get('id')} | "
        f"{match.get('home_team')} vs "
        f"{match.get('away_team')} | "
        f"{match.get('event_date')}"
    )


if not results:
    raise RuntimeError(
        "No finished Premier League matches returned."
    )


# ---------------------------------------------------------
# 2. Pick first match
# ---------------------------------------------------------

match = results[0]

event_id = match["id"]


print()
print("=" * 80)
print(
    f"Inspecting event ID: {event_id}"
)
print("=" * 80)


# ---------------------------------------------------------
# 3. Availability
# ---------------------------------------------------------

availability = get_json(
    f"/events/{event_id}/availability/"
)


print(
    "\nAVAILABILITY"
)

print(
    json.dumps(
        availability,
        indent=2,
    )
)


# ---------------------------------------------------------
# 4. Player Stats
# ---------------------------------------------------------

player_stats = get_json(
    f"/events/{event_id}/player-stats/"
)


print(
    "\nPLAYER STATS TOP-LEVEL"
)

if isinstance(
    player_stats,
    dict,
):
    print(
        player_stats.keys()
    )

else:
    print(
        type(player_stats)
    )


print(
    "\nPLAYER STATS SAMPLE"
)

print(
    json.dumps(
        player_stats,
        indent=2,
    )[:12000]
)


# ---------------------------------------------------------
# 5. Match Stats / Shotmap
# ---------------------------------------------------------

stats = get_json(
    f"/events/{event_id}/stats/"
)


print(
    "\nMATCH STATS KEYS"
)

if isinstance(
    stats,
    dict,
):
    print(
        stats.keys()
    )


shotmap = (
    stats.get(
        "shotmap",
        []
    )
    if isinstance(
        stats,
        dict,
    )
    else []
)


print(
    f"\nShots in shotmap: "
    f"{len(shotmap)}"
)


if shotmap:

    print(
        "\nFIRST SHOT"
    )

    print(
        json.dumps(
            shotmap[0],
            indent=2,
        )
    )


# ---------------------------------------------------------
# 6. Save debug files
# ---------------------------------------------------------

os.makedirs(
    "data/raw/bsd/debug",
    exist_ok=True,
)


with open(
    f"data/raw/bsd/debug/"
    f"{event_id}_player_stats.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        player_stats,
        file,
        indent=2,
        ensure_ascii=False,
    )


with open(
    f"data/raw/bsd/debug/"
    f"{event_id}_stats.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        stats,
        file,
        indent=2,
        ensure_ascii=False,
    )


print(
    "\nSaved debug JSON files."
)