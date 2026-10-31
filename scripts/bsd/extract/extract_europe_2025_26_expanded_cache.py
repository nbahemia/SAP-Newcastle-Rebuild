from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from requests.utils import get_encoding_from_headers
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "bsd" / "europe_2025_26"
MANIFEST = RAW_ROOT / "cache_manifest.json"
BASE_URL = "https://sports.bzzoiro.com/api/v2"
REQUEST_TIMEOUT = 35
REQUEST_SLEEP = 0.08
MAX_RETRIES = 4
FAILURE_LOG = RAW_ROOT / "cache_failures.jsonl"

# Existing six leagues are already cached in the project. The expansion pass focuses
# on value-producing leagues that are likely to surface younger, cheaper players.
EXPANSION_TARGETS = [
    {"slug": "primeira_liga", "country": "Portugal", "name_candidates": ["Primeira Liga", "Liga Portugal", "Liga Portugal Betclic"]},
    {"slug": "belgian_pro_league", "country": "Belgium", "name_candidates": ["Pro League", "Jupiler Pro League", "Belgian Pro League"]},
    {"slug": "austrian_bundesliga", "country": "Austria", "name_candidates": ["Bundesliga", "Austrian Bundesliga"]},
    {"slug": "danish_superliga", "country": "Denmark", "name_candidates": ["Superliga", "Danish Superliga", "3F Superliga"]},
    {"slug": "swiss_super_league", "country": "Switzerland", "name_candidates": ["Super League", "Swiss Super League"]},
    {"slug": "scottish_premiership", "country": "Scotland", "name_candidates": ["Premiership", "Scottish Premiership"]},
    {"slug": "turkish_super_lig", "country": "Turkey", "name_candidates": ["Super Lig", "Süper Lig", "Turkish Super Lig"]},
]


class NoCookieHTTPAdapter(HTTPAdapter):
    """
    requests' default HTTPAdapter extracts Set-Cookie headers while building
    every response. The BSD API does not need cookies, and a malformed/large
    cookie header can stall inside http.cookiejar before requests' timeout
    machinery gets another chance to intervene.

    This adapter builds the requests.Response without cookie extraction.
    """

    def build_response(self, req, resp):
        response = Response()
        response.status_code = getattr(resp, "status", None)
        response.headers = CaseInsensitiveDict(getattr(resp, "headers", {}))
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp

        reason = getattr(resp, "reason", None)
        if isinstance(reason, bytes):
            try:
                reason = reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = reason.decode("iso-8859-1")

        response.reason = reason
        response.url = req.url
        response.request = req
        response.connection = self
        return response


def atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def log_failure(payload: dict):
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\\n")


class BudgetExhausted(RuntimeError):
    pass


class ApiClient:
    def __init__(self, max_requests: int):
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("BSD_API_KEY")
        if not api_key:
            raise RuntimeError("BSD_API_KEY not found in project .env")

        self.max_requests = max_requests
        self.requests_used = 0
        self.session = requests.Session()
        adapter = NoCookieHTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
            max_retries=0,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Accept": "application/json",
            "Connection": "close",
        })
        # BSD auth is token-based; cookies are unnecessary.
        self.session.cookies.clear()

    def _reserve(self):
        if self.requests_used >= self.max_requests:
            raise BudgetExhausted(
                f"Safety request ceiling reached: {self.requests_used}/{self.max_requests}"
            )
        self.requests_used += 1

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> Any:
        url = path if path.startswith("http") else f"{BASE_URL}{path}"
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._reserve()
            response = None
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=(10, REQUEST_TIMEOUT),
                    allow_redirects=True,
                )

                if allow_missing and response.status_code in {400, 404}:
                    time.sleep(REQUEST_SLEEP)
                    return {}

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = 2.0 * attempt
                    time.sleep(max(wait, 1.0))
                    continue

                if 500 <= response.status_code < 600:
                    raise requests.HTTPError(
                        f"BSD server returned {response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                payload = response.json()
                time.sleep(REQUEST_SLEEP)
                return payload

            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = min(8.0, 1.5 * attempt)
                    print(
                        f"    retry {attempt}/{MAX_RETRIES - 1}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    time.sleep(wait)
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

        raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_exc

    def paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 200,
    ) -> list[dict]:
        params = dict(params or {})
        rows: list[dict] = []
        offset = 0

        while True:
            payload = self.get_json(
                path,
                {**params, "limit": page_size, "offset": offset},
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


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def resolve_league(client: ApiClient, target: dict) -> dict:
    leagues = client.paginated(
        "/leagues/",
        params={"country": target["country"], "include_inactive": "true"},
    )

    desired = [norm(x) for x in target["name_candidates"]]
    exact = [x for x in leagues if norm(x.get("name")) in desired]
    if exact:
        return exact[0]

    partial = [
        x for x in leagues
        if any(d in norm(x.get("name")) or norm(x.get("name")) in d for d in desired)
    ]
    if partial:
        return partial[0]

    available = sorted({str(x.get("name")) for x in leagues})
    raise RuntimeError(
        f"Could not resolve {target['slug']} ({target['country']}). "
        f"Available: {available}"
    )


def season_is_2025_26(season: dict) -> bool:
    name = norm(season.get("name"))
    start = str(season.get("start_date") or "")
    end = str(season.get("end_date") or "")
    return (
        "2025 26" in name
        or "2025/26" in str(season.get("name") or "")
        or "2025-26" in str(season.get("name") or "")
        or (start.startswith("2025-") and end.startswith("2026-"))
    )


def resolve_season(client: ApiClient, league_id: int) -> dict:
    payload = client.get_json(f"/leagues/{league_id}/seasons/")
    seasons = payload if isinstance(payload, list) else (
        payload.get("results") or payload.get("seasons") or []
    )
    matches = [s for s in seasons if season_is_2025_26(s)]
    if not matches:
        raise RuntimeError(
            f"No 2025/26 season found for league {league_id}. "
            f"Available: {[s.get('name') for s in seasons]}"
        )
    return matches[0]


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"version": 1, "leagues": []}


def update_manifest(manifest: dict, league_meta: dict):
    leagues = manifest.setdefault("leagues", [])
    leagues[:] = [x for x in leagues if x.get("slug") != league_meta["slug"]]
    leagues.append(league_meta)
    leagues.sort(key=lambda x: x.get("slug", ""))
    manifest["updated_at_epoch"] = time.time()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MANIFEST, manifest)


def cache_league(client: ApiClient, target: dict, manifest: dict) -> dict:
    print("\n" + "=" * 110)
    print(f"CACHE EXPANSION: {target['country']} / {target['slug']}")
    print("=" * 110)

    league = resolve_league(client, target)
    league_id = int(league["id"])
    season = resolve_season(client, league_id)
    season_id = int(season["id"])

    events = client.paginated(
        "/events/",
        params={
            "league_id": league_id,
            "season_id": season_id,
            "status": "finished",
        },
    )

    event_dir = RAW_ROOT / target["slug"] / "events"
    event_dir.mkdir(parents=True, exist_ok=True)

    event_ids = []
    for event in events:
        raw_id = event.get("id") or event.get("event_id")
        if raw_id is not None:
            event_ids.append(int(raw_id))

    cached_before = sum((event_dir / f"{eid}.json").exists() for eid in event_ids)
    missing = len(event_ids) - cached_before

    print(f"League: {league.get('name')} (id={league_id})")
    print(f"Season: {season.get('name')} (id={season_id})")
    print(f"Finished matches: {len(event_ids)}")
    print(f"Already cached: {cached_before}")
    print(f"Need download: {missing}")
    print(f"Estimated detail requests: ~{missing * 2}")

    downloaded = 0
    failures = 0

    for i, event_id in enumerate(event_ids, start=1):
        cache_file = event_dir / f"{event_id}.json"
        if cache_file.exists():
            continue

        try:
            player_stats = client.get_json(
                f"/events/{event_id}/player-stats/",
                allow_missing=True,
            )
            stats = client.get_json(
                f"/events/{event_id}/stats/",
                allow_missing=True,
            )

            payload = {
                "player_stats": player_stats,
                "stats": stats,
                "cache_meta": {
                    "event_id": event_id,
                    "league_slug": target["slug"],
                    "league_id": league_id,
                    "season_id": season_id,
                },
            }
            atomic_write_json(cache_file, payload)
            downloaded += 1

        except BudgetExhausted:
            raise

        except KeyboardInterrupt:
            print(
                f"\n[INTERRUPTED] event {event_id}; completed cache files are safe."
            )
            raise

        except Exception as exc:
            failures += 1
            log_failure({
                "epoch": time.time(),
                "league_slug": target["slug"],
                "event_id": event_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "requests_used": client.requests_used,
            })
            print(f"  [FAIL] event {event_id}: {type(exc).__name__}: {exc}")

        processed = downloaded + failures
        if downloaded == 1 or processed % 10 == 0:
            cached_now = sum(
                1 for eid in event_ids
                if (event_dir / f"{eid}.json").exists()
            )
            print(
                f"  progress={cached_now}/{len(event_ids)} cached | "
                f"new={downloaded} fail={failures} | "
                f"requests={client.requests_used}/{client.max_requests}"
            )

    meta = {
        "slug": target["slug"],
        "country": target["country"],
        "league_id": league_id,
        "league_name": league.get("name"),
        "season_id": season_id,
        "season_name": season.get("name"),
        "finished_matches": len(event_ids),
        "cached_matches": sum(1 for eid in event_ids if (event_dir / f"{eid}.json").exists()),
        "downloaded_this_run": downloaded,
        "failures_this_run": failures,
    }
    update_manifest(manifest, meta)
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Resumable cache expansion for Europe 2025/26 BSD data."
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=7200,
        help="Hard safety ceiling for HTTP requests during this run.",
    )
    parser.add_argument(
        "--leagues",
        nargs="*",
        default=None,
        help="Optional slugs to run, e.g. primeira_liga belgian_pro_league",
    )
    args = parser.parse_args()

    selected = EXPANSION_TARGETS
    if args.leagues:
        wanted = set(args.leagues)
        selected = [x for x in EXPANSION_TARGETS if x["slug"] in wanted]
        unknown = wanted - {x["slug"] for x in selected}
        if unknown:
            raise SystemExit(f"Unknown league slugs: {sorted(unknown)}")

    client = ApiClient(max_requests=args.max_requests)
    manifest = load_manifest()
    summaries = []

    print("=" * 110)
    print("EUROPE 2025/26 EXPANDED BSD CACHE")
    print("=" * 110)
    print(f"Safety ceiling: {client.max_requests} requests")
    print("Cache is resumable: existing event JSON files are never re-downloaded.")

    try:
        for target in selected:
            try:
                summaries.append(cache_league(client, target, manifest))
            except BudgetExhausted:
                raise
            except Exception as exc:
                print(f"[SKIP] {target['slug']}: {exc}")
    except BudgetExhausted as exc:
        print(f"\n[STOPPED SAFELY] {exc}")
    except KeyboardInterrupt:
        print("\n[STOPPED BY USER] Existing cache files remain valid and resumable.")

    print("\n" + "=" * 110)
    print("CACHE RUN COMPLETE")
    print("=" * 110)
    for row in summaries:
        print(
            f"{row['slug']:<24} cached={row['cached_matches']:>3}/"
            f"{row['finished_matches']:<3} new={row['downloaded_this_run']:<3} "
            f"fail={row['failures_this_run']}"
        )
    print(f"Requests used: {client.requests_used}/{client.max_requests}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
