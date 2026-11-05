from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import sys
import time
import unicodedata

import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]

ROLE_DIR = (
    ROOT
    / "scripts"
    / "bsd"
    / "analyze"
    / "role_scouting"
)

ANALYZE_DIR = (
    ROOT
    / "scripts"
    / "bsd"
    / "analyze"
)

BSD_DIR = (
    ROOT
    / "data"
    / "processed"
    / "bsd"
)

SHORTLIST_DIR = (
    BSD_DIR
    / "shortlist"
)


# ======================================================================================
# PIPELINE ORDER
#
# Defensive -> attacking
# ======================================================================================

STAGES = [
    # ------------------------------------------------------------------
    # GOALKEEPER
    # ------------------------------------------------------------------
    {
        "label": "Goalkeeper",
        "scripts": [
            ROLE_DIR / "analyze_goalkeeper.py",
        ],
    },

    # ------------------------------------------------------------------
    # CENTER-BACK
    # ------------------------------------------------------------------
    {
        "label": "Center-back",
        "scripts": [
            ROLE_DIR / "analyze_centerback.py"
        ],
    },

    # ------------------------------------------------------------------
    # FULLBACKS / WINGBACKS
    # ------------------------------------------------------------------
    {
        "label": "Left-back / wingback",
        "scripts": [
            ROLE_DIR / "analyze_leftback.py"
        ],
    },
    {
        "label": "Right-back / wingback",
        "scripts": [
            ROLE_DIR / "analyze_rightback.py"
        ],
    },

    # ------------------------------------------------------------------
    # CENTRAL MIDFIELD
    # ------------------------------------------------------------------
    {
        "label": "Deep transition controller",
        "scripts": [
            ROLE_DIR / "analyze_controller.py",
        ],
    },
    {
        "label": "Workhorse / advanced 8",
        "scripts": [
            ROLE_DIR / "analyze_workhorse_cm.py"
        ],
    },

    # ------------------------------------------------------------------
    # ATTACKING MIDFIELD
    # ------------------------------------------------------------------
    {
        "label": "Creative 10",
        "scripts": [
            ROLE_DIR / "analyze_cam.py"
        ],
    },

    # ------------------------------------------------------------------
    # WINGER
    # ------------------------------------------------------------------
    {
        "label": "Winger archetypes",
        "scripts": [
            ROLE_DIR / "analyze_winger.py"
        ],
    },

    # ------------------------------------------------------------------
    # STRIKER
    # ------------------------------------------------------------------
    {
        "label": "Focal striker",
        "scripts": [
            ROLE_DIR / "analyze_striker.py"
        ],
    },

    # ------------------------------------------------------------------
    # DOWNSTREAM ANALYSIS
    # ------------------------------------------------------------------
    {
        "label": "Metric repair",
        "scripts": [
            ROLE_DIR / "repair_metric_outputs.py",
        ],
    },
    {
        "label": "Squad depth",
        "scripts": [
            ROLE_DIR / "build_squad_depth_analysis.py",
        ],
    },
    {
        "label": "Recruitment value",
        "scripts": [
            ROLE_DIR / "build_recruitment_value_analysis.py",
        ],
    },
    {
        "label": "Visualization pack",
        "scripts": [
            ROOT / "scripts" / "bsd" / "visualize" / "generate_jaissle_visualizations.py",
        ],
    },
]


# ======================================================================================
# HELPERS
# ======================================================================================

def norm(value) -> str:
    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    ).strip()


def first_existing(
    df: pd.DataFrame,
    names: list[str],
) -> str | None:
    for name in names:
        if name in df.columns:
            return name

    return None


def resolve_script(
    candidates: list[Path],
) -> Path:
    for script in candidates:
        if script.exists():
            return script

    attempted = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise FileNotFoundError(
        "No matching script found.\n"
        f"Tried:\n{attempted}"
    )


# ======================================================================================
# QUIET STAGE EXECUTION
# ======================================================================================

def run_stage(
    label: str,
    candidates: list[Path],
):
    script = resolve_script(
        candidates
    )

    started = time.time()

    env = os.environ.copy()

    # Force subprocesses to use UTF-8.
    # This prevents Windows cp1252 crashes on names such as
    # Lukáš Horníček, Dedić, Kökçü, etc.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = (
        time.time()
        - started
    )

    if result.returncode != 0:
        print()
        print("=" * 120)
        print(
            f"FAILED: {label}"
        )
        print("=" * 120)

        if result.stdout:
            print(
                result.stdout
            )

        if result.stderr:
            print(
                result.stderr
            )

        raise RuntimeError(
            f"{label} failed "
            f"with exit code "
            f"{result.returncode}: "
            f"{script}"
        )

    print(
        f"[OK] "
        f"{label:<32} "
        f"{elapsed:>5.1f}s"
    )
    
# ======================================================================================
# CURRENT NEWCASTLE SQUAD NAMES
# ======================================================================================

def load_current_newcastle_names() -> set[str]:
    candidates = [
        BSD_DIR
        / "player_season_metrics_enriched.parquet",

        BSD_DIR
        / "player_team_season_metrics.parquet",

        BSD_DIR
        / "player_season_metrics.parquet",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_parquet(
            path
        )

        player_col = first_existing(
            df,
            [
                "player_name",
                "Player",
                "name",
            ],
        )

        if player_col is None:
            continue

        club_col = first_existing(
            df,
            [
                "club_name",
                "team_name",
                "display_club_name",
                "Club",
                "Team",
            ],
        )

        work = df.copy()

        if club_col is not None:
            mask = (
                work[
                    club_col
                ]
                .astype(str)
                .str.contains(
                    "Newcastle",
                    case=False,
                    na=False,
                )
            )

            if mask.any():
                work = work[
                    mask
                ].copy()

        names = set(
            work[
                player_col
            ]
            .astype(str)
            .map(norm)
        )

        names.discard("")

        if names:
            return names

    return set()


# ======================================================================================
# 2025/26 HISTORICAL ROLE VIEW
# ======================================================================================

def print_historical_newcastle_roles():
    historical_path = (
        SHORTLIST_DIR
        / "europe_2025_26_full_role_shortlist.parquet"
    )

    if not historical_path.exists():
        print(
            "[SKIP] 2025/26 historical role shortlist not found."
        )
        return

    current_names = (
        load_current_newcastle_names()
    )

    if not current_names:
        print(
            "[SKIP] Could not identify current Newcastle squad."
        )
        return

    hist = pd.read_parquet(
        historical_path
    ).copy()

    if (
        "player_name"
        not in hist.columns
    ):
        print(
            "[SKIP] Historical shortlist has no player_name."
        )
        return

    hist[
        "_name"
    ] = (
        hist[
            "player_name"
        ]
        .astype(str)
        .map(norm)
    )

    hist = hist[
        hist[
            "_name"
        ].isin(
            current_names
        )
    ].copy()

    if hist.empty:
        print(
            "[SKIP] No current Newcastle players matched "
            "the 2025/26 historical role universe."
        )
        return

    role_col = first_existing(
        hist,
        [
            "role",
            "role_name",
            "role_key",
        ],
    )

    fit_col = first_existing(
        hist,
        [
            "role_fit",
            "fit_score",
            "score",
        ],
    )

    if (
        role_col is None
        or fit_col is None
    ):
        print(
            "[SKIP] Historical shortlist lacks role / role-fit columns."
        )
        return

    hist[
        "_fit"
    ] = pd.to_numeric(
        hist[
            fit_col
        ],
        errors="coerce",
    )

    # Keep the two strongest 2025/26 roles for each current Newcastle player.
    hist = (
        hist.sort_values(
            [
                "player_name",
                "_fit",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "player_name",
            as_index=False,
        )
        .head(
            2
        )
        .copy()
    )

    columns = [
        "player_name",
        role_col,
        "_fit",
    ]

    if (
        "club_name"
        in hist.columns
    ):
        columns.insert(
            1,
            "club_name",
        )

    if (
        "minutes"
        in hist.columns
    ):
        columns.append(
            "minutes"
        )

    display = hist[
        columns
    ].copy()

    rename = {
        "player_name":
        "Player",

        "club_name":
        "2025/26 Club",

        role_col:
        "2025/26 Role",

        "_fit":
        "Role Fit",

        "minutes":
        "Minutes",
    }

    display = display.rename(
        columns=rename
    )

    if (
        "Role Fit"
        in display.columns
    ):
        display[
            "Role Fit"
        ] = pd.to_numeric(
            display[
                "Role Fit"
            ],
            errors="coerce",
        ).round(
            2
        )

    if (
        "Minutes"
        in display.columns
    ):
        display[
            "Minutes"
        ] = pd.to_numeric(
            display[
                "Minutes"
            ],
            errors="coerce",
        ).round(
            0
        )

    print()
    print(
        "=" * 120
    )
    print(
        "CURRENT NEWCASTLE SQUAD — "
        "2025/26 HISTORICAL ROLE REFERENCES"
    )
    print(
        "=" * 120
    )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Note: these are each current Newcastle player's "
        "strongest roles from the completed 2025/26 season. "
        "Use them alongside the smaller 2026/27 sample."
    )


# ======================================================================================
# SQUAD DEPTH SUMMARY
# ======================================================================================

def print_squad_depth():
    path = (
        SHORTLIST_DIR
        / "squad_depth_analysis.csv"
    )

    if not path.exists():
        return

    df = pd.read_csv(
        path
    )

    preferred = [
        "Role",
        "Choice 1",
        "Score 1",
        "Choice 2",
        "Score 2",
        "Depth Risk",
    ]

    cols = [
        c
        for c in preferred
        if c in df.columns
    ]

    if not cols:
        return

    view = df[
        cols
    ].copy()

    for col in [
        "Score 1",
        "Score 2",
    ]:
        if col in view.columns:
            view[
                col
            ] = pd.to_numeric(
                view[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )

    print()
    print(
        "=" * 120
    )
    print(
        "NEWCASTLE ROLE / DEPTH DIAGNOSIS"
    )
    print(
        "=" * 120
    )

    print(
        view.to_string(
            index=False
        )
    )


# ======================================================================================
# TOP RECRUITMENT TARGETS
# ======================================================================================

def print_recruitment_summary():
    path = (
        SHORTLIST_DIR
        / "recruitment_value_analysis.csv"
    )

    if not path.exists():
        return

    df = pd.read_csv(
        path
    )

    required = {
        "Role",
        "Player",
    }

    if not required.issubset(
        df.columns
    ):
        return

    sort_col = first_existing(
        df,
        [
            "Recruitment Value Score",
            "Upgrade",
            "Target Score",
        ],
    )

    if sort_col:
        df[
            "_sort"
        ] = pd.to_numeric(
            df[
                sort_col
            ],
            errors="coerce",
        )

        df = df.sort_values(
            [
                "Role",
                "_sort",
            ],
            ascending=[
                True,
                False,
            ],
        )

    # Keep only the strongest three players per role.
    df = (
        df.groupby(
            "Role",
            as_index=False,
            group_keys=False,
        )
        .head(
            3
        )
        .copy()
    )

    preferred = [
        "Role",
        "Player",
        "Age",
        "Target Score",
        "Current Benchmark",
        "Upgrade",
        "Value €M",
        "Class",
    ]

    cols = [
        c
        for c in preferred
        if c in df.columns
    ]

    view = df[
        cols
    ].copy()

    for col in [
        "Age",
        "Target Score",
        "Current Benchmark",
        "Upgrade",
        "Value €M",
    ]:
        if col in view.columns:
            view[
                col
            ] = pd.to_numeric(
                view[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )

    print()
    print(
        "=" * 120
    )
    print(
        "TOP RECRUITMENT OPTIONS BY ROLE"
    )
    print(
        "=" * 120
    )

    print(
        view.to_string(
            index=False
        )
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    print(
        "=" * 120
    )
    print(
        "JAISSLE POST-ANALYSIS PIPELINE"
    )
    print(
        "=" * 120
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Running role models "
        "from defensive -> attacking..."
    )
    print()

    for stage in STAGES:
        run_stage(
            stage[
                "label"
            ],
            stage[
                "scripts"
            ],
        )

    # ------------------------------------------------------------------
    # FINAL CURATED OUTPUT
    # ------------------------------------------------------------------

    print_squad_depth()

    print_historical_newcastle_roles()

    print_recruitment_summary()

    print()
    print(
        "=" * 120
    )
    print(
        "POST-ANALYSIS COMPLETE"
    )
    print(
        "=" * 120
    )

    print(
        "Use the 2026/27 Newcastle diagnosis together with "
        "the 2025/26 historical role references before assigning "
        "final recruitment priorities."
    )


if __name__ == "__main__":
    main()