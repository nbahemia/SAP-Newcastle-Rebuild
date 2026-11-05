from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

EXTRACT_DIR = ROOT / "scripts" / "bsd" / "extract"
TRANSFORM_DIR = ROOT / "scripts" / "bsd" / "transform"
ANALYZE_DIR = ROOT / "scripts" / "bsd" / "analyze"
ROLE_DIR = ANALYZE_DIR / "role_scouting"


# Canonical Newcastle recruitment pipeline
CACHE_EXTRACTOR = EXTRACT_DIR / "extract_europe_2025_26_expanded_cache.py"

OFFLINE_POOL_BUILDER = (
    TRANSFORM_DIR / "build_europe_2025_2026_pool_offline.py"
)

TRANSFERMARKT_ENRICHER = (
    TRANSFORM_DIR / "enrich_europe_2025_2026_transfermarkt.py"
)

EUROPE_SHORTLIST_BUILDER = (
    ANALYZE_DIR / "build_europe_shortlist_2025_2026.py"
)

JAISSLE_PIPELINE = (
    ROLE_DIR / "run_jaissle_recruitment_pipeline.py"
)


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def validate_pipeline_scripts() -> None:
    """Fail immediately if a canonical pipeline script is missing."""

    scripts = {
        "BSD cache extractor": CACHE_EXTRACTOR,
        "Offline Europe pool builder": OFFLINE_POOL_BUILDER,
        "Transfermarkt enricher": TRANSFERMARKT_ENRICHER,
        "Europe shortlist builder": EUROPE_SHORTLIST_BUILDER,
        "Jaissle recruitment pipeline": JAISSLE_PIPELINE,
    }

    missing = [
        f"{label}: {path}"
        for label, path in scripts.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Canonical recruitment pipeline is incomplete:\n  "
            + "\n  ".join(missing)
        )


def run_stage(
    label: str,
    script: Path,
    extra_args: list[str] | None = None,
) -> None:
    banner(label)

    cmd = [
        sys.executable,
        str(script),
        *(extra_args or []),
    ]

    print("Command:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed ({result.returncode}): "
            f"{label}\n{script}"
        )

    print(f"[OK] {label}")


def extraction_stage(max_requests: int) -> None:
    run_stage(
        "1. EXPAND / RESUME BSD EUROPE CACHE",
        CACHE_EXTRACTOR,
        [
            "--max-requests",
            str(max_requests),
        ],
    )


def rebuild_stages() -> None:
    stages = [
        (
            "2. REBUILD OFFLINE PLAYER-SEASON UNIVERSE",
            OFFLINE_POOL_BUILDER,
        ),
        (
            "3. RE-RUN TRANSFERMARKT ENRICHMENT",
            TRANSFERMARKT_ENRICHER,
        ),
        (
            "4. REBUILD FULL ROLE SHORTLIST",
            EUROPE_SHORTLIST_BUILDER,
        ),
        (
            "5. RE-RUN JAISSLE ROLE / SQUAD MODELS",
            JAISSLE_PIPELINE,
        ),
    ]

    for label, script in stages:
        run_stage(label, script)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-command Newcastle recruitment rebuild. "
            "Can expand the BSD cache, rebuild the offline "
            "Europe player universe, enrich Transfermarkt, "
            "rebuild the role shortlist, and rerun all "
            "Jaissle recruitment models."
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
        help=(
            "Run cache extraction and then every "
            "rebuild stage."
        ),
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=7200,
        help=(
            "Safety ceiling passed to the cache extractor "
            "(default: 7200)."
        ),
    )

    args = parser.parse_args()

    banner("NEWCASTLE EUROPE RECRUITMENT — MASTER REBUILD")
    print(f"Project root: {ROOT}")

    # Check the entire canonical pipeline before doing expensive work.
    validate_pipeline_scripts()

    if args.extract_only:
        extraction_stage(args.max_requests)
        return

    if args.all:
        extraction_stage(args.max_requests)

    rebuild_stages()

    banner("REBUILD COMPLETE")

    print("Expected refreshed outputs include:")
    print(
        "  data/processed/bsd/"
        "europe_2025_26_player_season_metrics.parquet"
    )
    print(
        "  data/processed/bsd/"
        "europe_2025_26_player_percentiles.parquet"
    )
    print(
        "  data/processed/bsd/"
        "europe_2025_26_player_percentiles_enriched.parquet"
    )
    print(
        "  data/processed/bsd/"
        "europe_2025_26_full_role_shortlist.parquet"
    )
    print(
        "  data/processed/bsd/shortlist/"
        "jaissle_recruitment_master_report.html"
    )


if __name__ == "__main__":
    main()