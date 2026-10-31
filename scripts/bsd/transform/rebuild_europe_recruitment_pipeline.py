from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = ROOT / "scripts" / "bsd"


def banner(text: str):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def find_script(*names: str) -> Path:
    # Prefer exact project locations first, then search recursively under scripts/bsd.
    for name in names:
        hits = list(SCRIPTS_ROOT.rglob(name))
        if hits:
            # Prefer shallower paths to avoid archived/backup copies.
            hits.sort(key=lambda p: (len(p.parts), str(p)))
            return hits[0]
    raise FileNotFoundError(
        "Could not find any of these scripts under scripts/bsd:\n  "
        + "\n  ".join(names)
    )


def run_stage(label: str, script: Path, extra_args: list[str] | None = None):
    banner(label)
    cmd = [sys.executable, str(script), *(extra_args or [])]
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed ({result.returncode}): {label}\n{script}")
    print(f"[OK] {label}")


def extraction_stage(max_requests: int):
    script = find_script("extract_europe_2025_26_expanded_cache.py")
    run_stage(
        "1. EXPAND / RESUME BSD EUROPE CACHE",
        script,
        ["--max-requests", str(max_requests)],
    )


def rebuild_stages():
    offline = find_script(
        "build_europe_2025_2026_pool_offline.py",
        "build_europe_2025_2026_pool_offline_expanded.py",
        "build_europe_2025_2026_pool_offline_remade.py",
    )
    transfermarkt = find_script(
        "enrich_europe_2025_26_transfermarkt.py",
        "enrich_europe_2025_26_transfermarkt_expanded.py",
        "enrich_europe_2025_26_transfermarkt_fixed.py",
    )
    shortlist = find_script(
        "build_europe_2025_26_full_role_shortlist.py",
        "build_europe_shortlist_2025_2026_split_exports.py",
    )
    master = find_script(
        "run_jaissle_recruitment_pipeline.py",
        "run_jaissle_recruitment_pipeline_v8.py",
    )

    stages = [
        ("2. REBUILD OFFLINE PLAYER-SEASON UNIVERSE", offline),
        ("3. RE-RUN TRANSFERMARKT ENRICHMENT", transfermarkt),
        ("4. REBUILD FULL ROLE SHORTLIST", shortlist),
        ("5. RE-RUN JAISSLE ROLE / SQUAD MODELS", master),
    ]

    for label, script in stages:
        run_stage(label, script)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "One-command Newcastle recruitment rebuild. Can expand the BSD cache, "
            "rebuild the offline Europe player universe, enrich Transfermarkt, "
            "rebuild the role shortlist, and rerun all Jaissle models."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--extract-only",
        action="store_true",
        help="Only run the resumable BSD cache expansion.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run cache extraction and then every rebuild stage.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=7200,
        help="Safety ceiling passed to the cache extractor (default: 7200).",
    )
    args = parser.parse_args()

    banner("NEWCASTLE EUROPE RECRUITMENT — MASTER REBUILD")
    print(f"Project root: {ROOT}")

    if args.extract_only:
        extraction_stage(args.max_requests)
        return

    if args.all:
        extraction_stage(args.max_requests)

    rebuild_stages()

    banner("REBUILD COMPLETE")
    print("Expected refreshed outputs include:")
    print("  data/processed/bsd/europe_2025_26_player_season_metrics.parquet")
    print("  data/processed/bsd/europe_2025_26_player_percentiles.parquet")
    print("  data/processed/bsd/europe_2025_26_player_percentiles_enriched.parquet")
    print("  data/processed/bsd/europe_2025_26_full_role_shortlist.parquet")
    print("  data/processed/bsd/shortlist/jaissle_recruitment_master_report.html")


if __name__ == "__main__":
    main()
