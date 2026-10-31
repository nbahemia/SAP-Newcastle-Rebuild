from __future__ import annotations

import math
import re
import unicodedata
import urllib.request
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed" / "bsd"
EXTERNAL = ROOT / "data" / "external" / "transfermarkt"

BSD_POOL = PROCESSED / "europe_2025_26_player_percentiles.parquet"
TM_DB = EXTERNAL / "transfermarkt-datasets.duckdb"
TM_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"

OUT_ENRICHED = PROCESSED / "europe_2025_26_player_percentiles_enriched.parquet"
OUT_AUDIT = PROCESSED / "europe_2025_26_transfermarkt_match_audit.parquet"
OUT_UNMATCHED = PROCESSED / "europe_2025_26_transfermarkt_unmatched.parquet"

TM_COMPETITIONS = {
    "premier_league": "GB1",
    "bundesliga": "L1",
    "la_liga": "ES1",
    "serie_a": "IT1",
    "ligue_1": "FR1",
    "eredivisie": "NL1",
    "primeira_liga": "PO1",
    "belgian_pro_league": "BE1",
    "austrian_bundesliga": "A1",
    "danish_superliga": "DK1",
    "swiss_super_league": "C1",
    "scottish_premiership": "SC1",
    "turkish_super_lig": "TR1",
}
TM_SEASON = 2025

AUTO_ACCEPT_SCORE = 95.0
AUTO_ACCEPT_MARGIN = 6.0
REVIEW_SCORE = 84.0


def banner(text: str) -> None:
    print()
    print("=" * 120)
    print(text.center(120))
    print("=" * 120)


def norm(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def split_bsd_name(name):
    s = norm(name)
    if not s:
        return "", "", ""
    parts = s.split()
    if len(parts[0]) == 1:
        return parts[0], " ".join(parts[1:]), s
    return "", s, s


def split_tm_name(name):
    s = norm(name)
    if not s:
        return "", "", ""
    parts = s.split()
    initial = parts[0][0]
    surname = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    return initial, surname, s


def overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def suffix_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a.endswith(b) or b.endswith(a):
        return 0.96
    aa, bb = a.split(), b.split()
    if aa[-1] == bb[-1]:
        return 0.88 + 0.10 * overlap(a, b)
    return 0.75 * overlap(a, b)


def score_name(bsd_name: str, tm_name: str) -> float:
    bi, bt, bf = split_bsd_name(bsd_name)
    ti, ts, tf = split_tm_name(tm_name)
    if not bf or not tf:
        return 0.0

    if not bi:
        if bf == tf:
            return 100.0
        if bf in tf.split():
            return 97.0
        return 100.0 * suffix_similarity(bf, ts)

    surname_score = suffix_similarity(bt, ts)
    if bi != ti:
        return 55.0 * surname_score

    return 30.0 + 70.0 * surname_score


def download_db() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    if TM_DB.exists():
        print(f"Using existing Transfermarkt DB:\n{TM_DB}")
        return

    banner("DOWNLOADING TRANSFERMARKT SNAPSHOT")
    print("One-time download of the July 2026 DuckDB snapshot.")

    temp = TM_DB.with_suffix(".duckdb.part")

    # R2 may return 403 to Python's default urllib user agent.
    # First try an explicit browser-like User-Agent.
    request = urllib.request.Request(
        TM_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/156 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )

    try:
        print("Trying direct Python download...")

        with urllib.request.urlopen(request, timeout=120) as response:
            with temp.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

    except Exception as exc:
        print(f"Python download failed: {exc}")
        print("Falling back to curl.exe...")

        if temp.exists():
            temp.unlink()

        result = subprocess.run(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-A",
                "Mozilla/5.0",
                "-o",
                str(temp),
                TM_URL,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if temp.exists():
                temp.unlink()

            raise RuntimeError(
                "Transfermarkt snapshot download failed with both urllib "
                "and curl.exe.\n\n"
                f"curl stderr:\n{result.stderr}\n\n"
                "Manual fallback:\n"
                f"  curl.exe -L -o \"{TM_DB}\" \"{TM_URL}\""
            )

    if not temp.exists() or temp.stat().st_size < 1_000_000:
        if temp.exists():
            temp.unlink()

        raise RuntimeError(
            "Downloaded file is unexpectedly small; refusing to use it."
        )

    temp.replace(TM_DB)

    print(f"Saved:\n{TM_DB}")


def columns(con, table: str) -> set[str]:
    return {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}


def build_tm_universe() -> pd.DataFrame:
    con = duckdb.connect(str(TM_DB), read_only=True)

    try:
        pcols = columns(con, "players")
        ccols = columns(con, "clubs")
        vcols = columns(con, "player_valuations")

        player_name = "p.name" if "name" in pcols else "CAST(p.player_id AS VARCHAR)"
        dob = "p.date_of_birth" if "date_of_birth" in pcols else "NULL::DATE"
        pos = "p.position" if "position" in pcols else "NULL::VARCHAR"
        subpos = "p.sub_position" if "sub_position" in pcols else "NULL::VARCHAR"
        foot = "p.foot" if "foot" in pcols else "NULL::VARCHAR"
        height = "p.height_in_cm" if "height_in_cm" in pcols else "NULL::DOUBLE"
        current_club_id = "p.current_club_id" if "current_club_id" in pcols else "NULL::BIGINT"
        current_value = "p.market_value_in_eur" if "market_value_in_eur" in pcols else "NULL::DOUBLE"

        if "name" in ccols:
            club_name = "c.name"
        elif "club_name" in ccols:
            club_name = "c.club_name"
        else:
            club_name = "NULL::VARCHAR"

        valuation_value = "market_value_in_eur" if "market_value_in_eur" in vcols else "NULL::DOUBLE"
        valuation_date = "date" if "date" in vcols else ("datetime" if "datetime" in vcols else "NULL::DATE")

        comp_list = ", ".join(f"'{x}'" for x in TM_COMPETITIONS.values())

        query = f"""
        WITH season_players AS (
            SELECT DISTINCT a.player_id, g.competition_id
            FROM appearances a
            JOIN games g USING (game_id)
            WHERE g.season = {TM_SEASON}
              AND g.competition_id IN ({comp_list})
        ),
        latest_valuation AS (
            SELECT
                player_id,
                {valuation_value} AS market_value_in_eur,
                {valuation_date} AS valuation_date,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY {valuation_date} DESC NULLS LAST
                ) AS rn
            FROM player_valuations
            WHERE {valuation_date} IS NULL
               OR CAST({valuation_date} AS DATE) <= DATE '2026-06-30'
        )
        SELECT
            sp.player_id AS tm_player_id,
            sp.competition_id AS tm_competition_id,
            {player_name} AS tm_player_name,
            {dob} AS date_of_birth,
            {pos} AS tm_position,
            {subpos} AS tm_sub_position,
            {foot} AS foot,
            {height} AS height_cm,
            {current_club_id} AS current_club_id,
            {club_name} AS current_club_name,
            COALESCE(lv.market_value_in_eur, {current_value}) AS market_value_eur,
            lv.valuation_date
        FROM season_players sp
        JOIN players p ON sp.player_id = p.player_id
        LEFT JOIN clubs c ON {current_club_id} = c.club_id
        LEFT JOIN latest_valuation lv
          ON sp.player_id = lv.player_id
         AND lv.rn = 1
        """

        return con.execute(query).df()

    finally:
        con.close()


def competition_slug(comp_id: str):
    reverse = {v: k for k, v in TM_COMPETITIONS.items()}
    return reverse.get(comp_id)


def build_matches(bsd: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    tm = tm.copy()
    tm["league_slug"] = tm["tm_competition_id"].map(competition_slug)

    identities = (
        bsd[["league_slug", "player_id", "player_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    records = []

    for i, row in identities.iterrows():
        league = row["league_slug"]
        bsd_name = row["player_name"]

        candidates = tm[tm["league_slug"].eq(league)].copy()

        if candidates.empty:
            records.append({
                "league_slug": league,
                "bsd_player_id": row["player_id"],
                "bsd_player_name": bsd_name,
                "match_status": "no_candidates",
                "match_score": 0.0,
                "score_margin": 0.0,
            })
            continue

        candidates["match_score"] = candidates["tm_player_name"].apply(
            lambda x: score_name(bsd_name, x)
        )
        candidates = candidates.sort_values(
            ["match_score", "market_value_eur"],
            ascending=[False, False],
            na_position="last",
        )

        best = candidates.iloc[0]
        second = float(candidates.iloc[1]["match_score"]) if len(candidates) > 1 else 0.0
        best_score = float(best["match_score"])
        margin = best_score - second

        if best_score >= AUTO_ACCEPT_SCORE and margin >= AUTO_ACCEPT_MARGIN:
            status = "accepted"
        elif best_score >= REVIEW_SCORE:
            status = "review"
        else:
            status = "unmatched"

        rec = {
            "league_slug": league,
            "bsd_player_id": row["player_id"],
            "bsd_player_name": bsd_name,
            "match_status": status,
            "match_score": best_score,
            "score_margin": margin,
        }

        for col in tm.columns:
            if col != "league_slug":
                rec[col] = best.get(col)

        records.append(rec)

        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(identities):
            print(f"Matched {i + 1:,}/{len(identities):,}")

    return pd.DataFrame(records)


def age_on(dob, ref="2026-07-01"):
    if pd.isna(dob):
        return math.nan
    dob = pd.Timestamp(dob)
    ref = pd.Timestamp(ref)
    return float(
        ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    )


def main() -> None:
    banner("TRANSFERMARKT ENRICHMENT — EUROPE 2025/26")

    if not BSD_POOL.exists():
        raise FileNotFoundError(f"Missing BSD pool:\n{BSD_POOL}")

    download_db()

    bsd = pd.read_parquet(BSD_POOL)
    print(f"BSD rows: {len(bsd):,}")

    banner("BUILDING TRANSFERMARKT 2025/26 PLAYER UNIVERSE")
    tm = build_tm_universe()
    print(f"Transfermarkt league-player rows: {len(tm):,}")

    banner("MATCHING BSD NAMES")
    matches = build_matches(bsd, tm)
    matches.to_parquet(OUT_AUDIT, index=False)

    print()
    print(matches["match_status"].value_counts(dropna=False).to_string())

    unmatched = matches[~matches["match_status"].eq("accepted")].copy()
    unmatched.to_parquet(OUT_UNMATCHED, index=False)

    accepted = matches[matches["match_status"].eq("accepted")].copy()

    keep = [
        "league_slug",
        "bsd_player_id",
        "tm_player_id",
        "tm_player_name",
        "date_of_birth",
        "tm_position",
        "tm_sub_position",
        "foot",
        "height_cm",
        "current_club_id",
        "current_club_name",
        "market_value_eur",
        "valuation_date",
        "match_score",
        "score_margin",
    ]
    keep = [c for c in keep if c in accepted.columns]

    enriched = bsd.merge(
        accepted[keep],
        how="left",
        left_on=["league_slug", "player_id"],
        right_on=["league_slug", "bsd_player_id"],
    )

    if "date_of_birth" in enriched.columns:
        enriched["age_at_2026_07_01"] = enriched["date_of_birth"].apply(age_on)

    enriched.to_parquet(OUT_ENRICHED, index=False)

    banner("SUMMARY")

    accepted_n = int(matches["match_status"].eq("accepted").sum())
    total_n = len(matches)
    print(
        f"Accepted matches: {accepted_n:,}/{total_n:,} "
        f"({accepted_n / total_n * 100:.1f}%)"
    )

    for col, label in [
        ("market_value_eur", "Market value"),
        ("age_at_2026_07_01", "Age"),
        ("height_cm", "Height"),
        ("current_club_name", "Club"),
    ]:
        if col in enriched.columns:
            cov = enriched[col].notna().mean() * 100
            print(f"{label} coverage: {cov:.1f}%")

    print()
    print("Saved:")
    print(OUT_ENRICHED)
    print(OUT_AUDIT)
    print(OUT_UNMATCHED)


if __name__ == "__main__":
    main()
