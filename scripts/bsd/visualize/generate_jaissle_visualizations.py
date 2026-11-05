#!/usr/bin/env python3
"""
generate_jaissle_visualizations.py

Create a role-based visualization pack for the Jaissle recruitment pipeline.

Shortlist upgrade
-----------------
Shortlisted Newcastle targets are now:
1. Force-included in their relevant positional chart even if outside the normal top-N cut.
2. Highlighted with a dedicated marker and label.
3. Split between CORE BUY and OPTIONAL SHORTLIST.
4. Shown alongside Newcastle benchmarks and the external market.
5. Right-back now has its own chart so van Rooij / Lissah are compared against RBs.

Usage
-----
python scripts/bsd/analyze/generate_jaissle_visualizations.py
python scripts/bsd/analyze/generate_jaissle_visualizations.py --top-n 40 --price-cap 70
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Final working shortlist
# -----------------------------------------------------------------------------

FINAL_SHORTLIST = {
    "Nikola Krstović": {
        "status": "CORE",
        "fee": 33.0,
        "roles": {"striker"},
    },
    "Kenan Yıldız": {
        "status": "CORE",
        "fee": 90.0,
        "roles": {"wide_creator", "vertical_box_winger", "overall_winger"},
    },
    "Rayan Aït-Nouri": {
        "status": "CORE",
        "fee": 30.0,
        "roles": {"left_wb"},
    },
    "Lazar Samardžić": {
        "status": "CORE",
        "fee": 20.0,
        "roles": {"cam"},
    },
    "Bart van Rooij": {
        "status": "CORE",
        "fee": 13.0,
        "roles": {"right_wb"},
    },
    "Filip Lissah": {
        "status": "CORE",
        "fee": 1.0,
        "roles": {"right_wb", "center_back"},
    },
    "Chibuike Nwaiwu": {
        "status": "OPTIONAL",
        "fee": 10.0,
        "roles": {"center_back"},
    },
    "Francisco Conceição": {
        "status": "OPTIONAL",
        "fee": 30.0,
        "roles": {"wide_creator", "vertical_box_winger", "overall_winger", "cam"},
    },
}

SHORTLIST_ALIASES = {
    "Nikola Krstović": ["N. Krstović", "Nikola Krstovic"],
    "Kenan Yıldız": ["K. Yıldız", "Kenan Yildiz"],
    "Rayan Aït-Nouri": ["R. Aït-Nouri", "Rayan Ait-Nouri"],
    "Lazar Samardžić": ["L. Samardžić", "Lazar Samardzic"],
    "Bart van Rooij": ["B. van Rooij"],
    "Filip Lissah": ["F. Lissah"],
    "Chibuike Nwaiwu": ["C. Nwaiwu"],
    "Francisco Conceição": ["F. Conceição", "Francisco Conceicao"],
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    candidates = []

    if start is not None:
        candidates.append(start.resolve())

    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent)

    seen = set()

    for seed in candidates:
        cur = seed.parent if seed.is_file() else seed

        while True:
            if cur in seen:
                break

            seen.add(cur)

            if (cur / "data").exists() and (cur / "scripts").exists():
                return cur

            if cur.parent == cur:
                break

            cur = cur.parent

    raise FileNotFoundError(
        "Could not determine project root. Run this inside SAP-Newcastle-Rebuild "
        "or place the script inside the repo."
    )


def normalize_key(text: object) -> str:
    if text is None:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(text),
    )
    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )
    value = value.lower()
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return value.strip()


def shortlist_aliases(canonical: str) -> set[str]:
    values = [
        canonical,
        *SHORTLIST_ALIASES.get(canonical, []),
    ]

    return {
        normalize_key(value)
        for value in values
        if normalize_key(value)
    }


def canonical_shortlist_name(value: object) -> Optional[str]:
    key = normalize_key(value)

    if not key:
        return None

    for canonical in FINAL_SHORTLIST:
        if key in shortlist_aliases(canonical):
            return canonical

    return None


def shortlist_for_role(role_slug: str) -> list[str]:
    return [
        player
        for player, info in FINAL_SHORTLIST.items()
        if role_slug in info["roles"]
    ]


def choose_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {
        normalize_key(c): c
        for c in df.columns
    }

    for cand in candidates:
        hit = lookup.get(
            normalize_key(cand)
        )

        if hit is not None:
            return hit

    return None


def choose_existing_file(
    root: Path,
    patterns: Iterable[str],
) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(
            root.rglob(pattern)
        )

        if hits:
            return hits[0]

    return None


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(path)

    if suf == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file type: {path}"
    )


def numeric(s: object) -> pd.Series:
    return pd.to_numeric(
        s,
        errors="coerce",
    )


def safe_text(v: object) -> str:
    return (
        "NA"
        if pd.isna(v)
        else str(v)
    )


def shortlist_root(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "processed"
        / "bsd"
        / "shortlist"
    )


def visualizations_root(project_root: Path) -> Path:
    out = project_root / "visualizations"
    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    return out


def visualization_dir(
    project_root: Path,
    folder: str,
) -> Path:
    out = visualizations_root(project_root) / folder
    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    return out


def league_to_color(values: pd.Series) -> np.ndarray:
    mapping = {
        "premier_league": "tab:red",
        "serie_a": "tab:blue",
        "la_liga": "tab:orange",
        "bundesliga": "tab:green",
        "ligue_1": "tab:purple",
        "eredivisie": "tab:brown",
        "primeira_liga": "tab:pink",
        "liga_portugal": "tab:pink",
        "belgian_pro_league": "tab:olive",
        "jupiler_pro_league": "tab:olive",
        "turkish_super_lig": "tab:cyan",
        "swiss_super_league": "goldenrod",
        "scottish_premiership": "slategray",
    }

    return np.array([
        mapping.get(
            str(v),
            "tab:gray",
        )
        for v in values
    ])


def bubble_sizes(vals: pd.Series) -> np.ndarray:
    x = numeric(vals).fillna(0)

    if len(x) == 0:
        return np.array([])

    if x.max() <= 0:
        return np.full(
            len(x),
            210.0,
        )

    return (
        110
        + 520
        * np.sqrt(
            x.clip(lower=0)
            / x.max()
        )
    )


# -----------------------------------------------------------------------------
# Role specs
# -----------------------------------------------------------------------------

ROLE_SPECS = [
    {
        "slug": "cam",
        "folder": "cam",
        "label": "Creative 10 / CAM",
        "benchmark_labels": [
            "Creative 10 / CAM",
            "Creative 10",
            "CAM",
        ],
        "patterns": [
            "creative_10_jaissle_detailed.csv",
            "creative_10.csv",
            "*cam*.csv",
        ],
        "score_candidates": [
            "CAM Fit",
            "Final CAM Score",
            "jaissle_cam_score",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Value €M",
            "market_value_eur_m",
            "Age",
        ],
        "y_candidates": [
            "CAM Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "alt_y_candidates": ["Coverage %"],
        "x_label": "Market value (€M)",
        "y_label": "CAM fit",
        "caption": (
            "External CAM market. Shortlisted players are highlighted."
        ),
    },
    {
        "slug": "striker",
        "folder": "striker",
        "label": "Focal Striker",
        "benchmark_labels": [
            "Focal Striker",
            "Striker",
        ],
        "patterns": [
            "focal_striker_jaissle_detailed.csv",
            "focal_striker.csv",
            "*striker*.csv",
        ],
        "score_candidates": [
            "Jaissle ST",
            "jaissle_striker_score",
            "Combined Score",
            "Focal Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Focal Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "y_candidates": [
            "Pressing",
            "pressing_score",
            "Jaissle ST",
            "Combined Score",
        ],
        "alt_y_candidates": [
            "Box Threat",
            "Focal Presence",
            "Link Play",
        ],
        "x_label": "Focal-striker fit",
        "y_label": "Pressing / overall Jaissle score",
        "caption": (
            "Focal striker fit vs pressing / total role score."
        ),
    },
    {
        "slug": "center_back",
        "folder": "center_back",
        "label": "High-Line Center Back",
        "benchmark_labels": [
            "High-Line Center Back",
            "Center Back",
            "Centre Back",
        ],
        "patterns": [
            "center_back_jaissle_detailed.csv",
            "center_back.csv",
            "*center*back*.csv",
            "*centre*back*.csv",
            "*cb*.csv",
        ],
        "score_candidates": [
            "Jaissle CB",
            "jaissle_cb_score",
            "CB Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Build-Up",
            "Progression",
            "Passing",
            "CB Fit",
            "role_fit",
            "Role Fit",
        ],
        "y_candidates": [
            "Front-Foot",
            "Physical",
            "Defending",
            "Duels",
            "Aerial",
            "Jaissle CB",
        ],
        "alt_y_candidates": [
            "Recovery Pace",
            "role_fit",
            "Role Fit",
        ],
        "x_label": "Build-up / role fit",
        "y_label": "Front-foot / physical defending",
        "caption": (
            "High-line CB market. Nwaiwu and Lissah are force-included when present."
        ),
    },
    {
        "slug": "left_wb",
        "folder": "fullback/left",
        "label": "Left Wingback / Fullback",
        "benchmark_labels": [
            "Left Wingback / Fullback",
            "Left Wingback",
            "Left Back",
            "Fullback",
        ],
        "patterns": [
            "left_wingback_recruitment_progressive.csv",
            "left_wingback_recruitment_creation.csv",
            "wingback.csv",
            "*left*wingback*.csv",
            "*leftback*.csv",
        ],
        "score_candidates": [
            "Jaissle FB Score",
            "jaissle_fb_score",
            "WB Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Creation",
            "Prog/Create",
            "Attacking Creation",
            "WB Fit",
            "role_fit",
        ],
        "y_candidates": [
            "Physical",
            "Defending",
            "Jaissle FB Score",
            "jaissle_fb_score",
        ],
        "alt_y_candidates": [
            "WB Fit",
            "role_fit",
            "Role Fit",
            "xA/90",
            "Key Passes/90",
        ],
        "x_label": "Creation / progression",
        "y_label": "Physical / defensive activity",
        "caption": (
            "Left-back market. Aït-Nouri is highlighted against comparable options."
        ),
    },
    {
        "slug": "right_wb",
        "folder": "fullback/right",
        "label": "Right Wingback / Fullback",
        "benchmark_labels": [
            "Right WB / FB",
            "Right Wingback / Fullback",
            "Right Back",
        ],
        "patterns": [
            "right_back_jaissle_detailed.csv",
            "*right*back*.csv",
            "*right*wingback*.csv",
        ],
        "score_candidates": [
            "jaissle_fb_score",
            "Jaissle RB",
            "RB Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Creation",
            "attacking_creation_index",
            "RB Fit",
            "role_fit",
        ],
        "y_candidates": [
            "Physical",
            "physical_activity_index",
            "jaissle_fb_score",
            "Jaissle RB",
        ],
        "alt_y_candidates": [
            "Recoveries/90",
            "Duels Won/90",
            "Tackles Won/90",
        ],
        "x_label": "Creation / role fit",
        "y_label": "Physical / defensive activity",
        "caption": (
            "Right-back market. van Rooij and Lissah are highlighted against comparable RBs."
        ),
    },
    {
        "slug": "holding_controller",
        "folder": "central_midfield",
        "label": "Holding Controller / No. 6",
        "benchmark_labels": [
            "Holding Controller / No. 6",
            "Holding Pivot",
            "Controller",
        ],
        "patterns": [
            "holding_controller_jaissle_detailed.csv",
            "holding_pivot.csv",
            "*holding*pivot*.csv",
            "*controller*.csv",
        ],
        "score_candidates": [
            "Jaissle Controller",
            "jaissle_controller_score",
            "role_fit",
            "Role Fit",
            "Role Score",
            "Fit",
        ],
        "x_candidates": [
            "Value €M",
            "market_value_eur_m",
            "Age",
        ],
        "y_candidates": [
            "Jaissle Controller",
            "jaissle_controller_score",
            "role_fit",
            "Role Fit",
            "Role Score",
            "Fit",
        ],
        "alt_y_candidates": [
            "Coverage %",
        ],
        "x_label": "Market value (€M)",
        "y_label": "Holding / controller fit",
        "caption": (
            "Controller market for squad context."
        ),
    },
    {
        "slug": "workhorse_cm",
        "folder": "central_midfield",
        "label": "Workhorse Central Midfielder",
        "benchmark_labels": [
            "Workhorse Central Midfielder",
            "Central Midfielder",
            "CM",
            "Workhorse CM",
        ],
        "patterns": [
            "workhorse_cm_jaissle_detailed.csv",
            "advanced_pivot.csv",
            "*workhorse*cm*.csv",
            "*workhorse*.csv",
        ],
        "score_candidates": [
            "Jaissle CM",
            "jaissle_workhorse_score",
            "CM Fit",
            "role_fit",
            "Role Fit",
            "Role Score",
        ],
        "x_candidates": [
            "Workhorse",
            "Ball Winning",
            "Progression",
            "CM Fit",
            "role_fit",
            "Role Fit",
        ],
        "y_candidates": [
            "Security/Link",
            "Attack Support",
            "Pressing",
            "Physical",
            "Jaissle CM",
        ],
        "alt_y_candidates": [
            "Recoveries/90",
            "Duels Won/90",
            "role_fit",
        ],
        "x_label": "Workhorse / ball-winning score",
        "y_label": "Security / link support",
        "caption": (
            "Workhorse midfield market for squad context."
        ),
    },
    {
        "slug": "wide_creator",
        "folder": "winger/wide_creator",
        "label": "Wide Creator",
        "benchmark_labels": [
            "Winger",
            "Wide Creator",
            "Vertical Winger",
        ],
        "patterns": [
            "bona_fide_winger_recruitment.csv",
            "winger_jaissle_extended.csv",
            "winger_jaissle_archetypes.csv",
            "*winger*.csv",
        ],
        "score_candidates": [
            "Winger Recruitment Score",
            "wide_creator_score",
            "Creator",
            "Best Score",
        ],
        "x_candidates": [
            "Creation",
            "Creator",
            "xA/90",
        ],
        "y_candidates": [
            "Goal Threat",
            "Vertical/Box",
            "Best Score",
        ],
        "alt_y_candidates": [
            "Hybrid",
            "Vertical/Box",
        ],
        "x_label": "Chance creation",
        "y_label": "Wide-creator score",
        "caption": (
            "Wide-creator market. Yıldız and Conceição are highlighted."
        ),
    },
    {
        "slug": "vertical_box_winger",
        "folder": "winger/vertical_box",
        "label": "Vertical / Box Winger",
        "benchmark_labels": [
            "Winger",
            "Vertical / Box Winger",
            "Vertical Winger",
        ],
        "patterns": [
            "bona_fide_winger_recruitment.csv",
            "winger_jaissle_extended.csv",
            "winger_jaissle_archetypes.csv",
            "*winger*.csv",
        ],
        "score_candidates": [
            "Winger Recruitment Score",
            "vertical_box_score",
            "Vertical/Box",
            "Best Score",
        ],
        "x_candidates": [
            "Creation",
            "Creator",
            "xA/90",
        ],
        "y_candidates": [
            "Goal Threat",
            "Vertical/Box",
            "xG/90",
        ],
        "alt_y_candidates": [
            "Hybrid",
            "Creator",
        ],
        "x_label": "Direct / box threat",
        "y_label": "Vertical / box score",
        "caption": (
            "Vertical winger market. Yıldız and Conceição are highlighted."
        ),
    },
    {
        "slug": "overall_winger",
        "folder": "winger/overall",
        "label": "Overall Winger",
        "benchmark_labels": [
            "Winger",
            "Hybrid Winger",
        ],
        "patterns": [
            "bona_fide_winger_recruitment.csv",
            "winger_jaissle_extended.csv",
            "winger_jaissle_archetypes.csv",
            "*winger*.csv",
        ],
        "score_candidates": [
            "Winger Recruitment Score",
            "overall_winger_score",
            "Best Score",
            "hybrid_score",
            "Hybrid",
        ],
        "x_candidates": [
            "1v1 / Progression",
            "Age",
            "Creator",
        ],
        "y_candidates": [
            "Winger Recruitment Score",
            "overall_winger_score",
            "Best Score",
            "Hybrid",
        ],
        "alt_y_candidates": [
            "Best Score",
        ],
        "x_label": "1v1 / Progression",
        "y_label": "Overall winger score",
        "caption": (
            "Overall winger market. Yıldız and Conceição are highlighted."
        ),
    },
]

COMMON_FIELD_CANDIDATES = {
    "player": [
        "Player",
        "player_name",
        "name",
    ],
    "club": [
        "Club",
        "club_name",
        "current_club_name",
        "team_name",
    ],
    "league": [
        "League",
        "league_slug",
    ],
    "age": [
        "Age",
        "age",
    ],
    "value": [
        "Value €M",
        "market_value_eur_m",
        "market_value_eur",
        "Value",
        "Transfer Value",
    ],
}


# -----------------------------------------------------------------------------
# Loading role data
# -----------------------------------------------------------------------------

def load_role_dataframe(
    project_root: Path,
    spec: dict,
) -> tuple[pd.DataFrame, Path, str, str]:
    root = shortlist_root(project_root)

    path = choose_existing_file(
        root,
        spec["patterns"],
    )

    if path is None:
        path = choose_existing_file(
            project_root
            / "data"
            / "processed"
            / "bsd",
            spec["patterns"],
        )

    if path is None:
        raise FileNotFoundError(
            f"Could not find file for role: {spec['label']}"
        )

    df = read_table(path).copy()

    for out_name, aliases in COMMON_FIELD_CANDIDATES.items():
        col = choose_column(
            df,
            aliases,
        )

        if (
            col is not None
            and out_name not in df.columns
        ):
            df[out_name] = df[col]

    score_col = choose_column(
        df,
        spec["score_candidates"],
    )
    xcol = choose_column(
        df,
        spec["x_candidates"],
    )
    ycol = choose_column(
        df,
        spec["y_candidates"],
    )

    if score_col is None:
        raise KeyError(
            f"No score column found for {spec['label']} in {path.name}"
        )

    if xcol is None:
        xcol = score_col

    if ycol is None:
        ycol = score_col

    df["_score"] = numeric(
        df[score_col]
    )
    df["_x"] = numeric(
        df[xcol]
    )
    df["_y"] = numeric(
        df[ycol]
    )

    df["_player"] = df.get(
        "player",
        pd.Series(
            "Unknown",
            index=df.index,
        ),
    )

    df["_club"] = df.get(
        "club",
        pd.Series(
            np.nan,
            index=df.index,
        ),
    )

    df["_league"] = df.get(
        "league",
        pd.Series(
            np.nan,
            index=df.index,
        ),
    )

    df["_value"] = (
        numeric(df["value"])
        if "value" in df.columns
        else np.nan
    )

    df["_age"] = (
        numeric(df["age"])
        if "age" in df.columns
        else np.nan
    )

    df["_shortlist_name"] = df["_player"].map(
        canonical_shortlist_name
    )

    return (
        df,
        path,
        xcol,
        ycol,
    )


def zscore_series(series: pd.Series) -> pd.Series:
    x = numeric(series)

    if x.notna().sum() < 3:
        return pd.Series(
            np.nan,
            index=x.index,
        )

    sd = x.std(
        ddof=0
    )

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return pd.Series(
            0.0,
            index=x.index,
        )

    return (
        x - x.mean()
    ) / sd


def scaled_composite(
    df: pd.DataFrame,
    metric_candidates: list[
        tuple[list[str], float]
    ],
) -> pd.Series:
    parts = []
    weights = []

    for aliases, weight in metric_candidates:
        col = choose_column(
            df,
            aliases,
        )

        if col is None:
            continue

        z = zscore_series(
            df[col]
        )

        if z.notna().sum() < 3:
            continue

        parts.append(z)
        weights.append(
            float(weight)
        )

    if not parts:
        return pd.Series(
            np.nan,
            index=df.index,
        )

    total = sum(weights)

    out = (
        sum(
            part * weight
            for part, weight in zip(
                parts,
                weights,
            )
        )
        / total
    )

    return (
        50.0
        + 10.0 * out
    )


def add_independent_plot_metrics(
    df: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, str, str]:
    out = df.copy()
    slug = spec.get("slug")

    if slug in {"left_wb", "right_wb"}:
        creation_col = choose_column(
            out,
            [
                "Creation",
                "attacking_creation_index",
                "Attacking Creation",
                "WB Fit",
                "RB Fit",
            ],
        )

        if creation_col is not None:
            out["_x"] = numeric(
                out[creation_col]
            )

        defensive = scaled_composite(
            out,
            [
                (
                    [
                        "Duels Won/90",
                        "duels_won_per90",
                    ],
                    0.30,
                ),
                (
                    [
                        "Recoveries/90",
                        "recoveries_per90",
                    ],
                    0.25,
                ),
                (
                    [
                        "Interceptions/90",
                        "interceptions_per90",
                    ],
                    0.20,
                ),
                (
                    [
                        "Tackles Won/90",
                        "tackles_won_per90",
                    ],
                    0.25,
                ),
            ],
        )

        if defensive.notna().sum() >= 6:
            out["_y"] = defensive
            ylabel = "Defensive / physical composite"
        else:
            fallback = choose_column(
                out,
                [
                    "Physical",
                    "physical_activity_index",
                    "WB Fit",
                    "RB Fit",
                    "Jaissle FB Score",
                    "jaissle_fb_score",
                ],
            )

            if fallback is not None:
                out["_y"] = numeric(
                    out[fallback]
                )

            ylabel = (
                fallback
                or "Physical"
            )

        return (
            out,
            "Creation / progression",
            ylabel,
        )

    if slug in {
        "wide_creator",
        "vertical_box_winger",
    }:
        # Use the recruitment model's existing dimensions directly.
        # All underlying data is from the same 25/26 season, so rebuilding
        # fresh z-score composites here only distorts the established scores.
        creation_col = choose_column(
            out,
            [
                "Creation",
                "Creator",
                "xA/90",
            ],
        )

        threat_col = choose_column(
            out,
            [
                "Goal Threat",
                "Vertical/Box",
                "xG/90",
            ],
        )

        if creation_col is not None:
            out["_x"] = numeric(out[creation_col])

        if threat_col is not None:
            out["_y"] = numeric(out[threat_col])

        return (
            out,
            creation_col or "Creation",
            threat_col or "Goal Threat",
        )

    return (
        out,
        spec.get(
            "x_label",
            "X",
        ),
        spec.get(
            "y_label",
            "Y",
        ),
    )


# -----------------------------------------------------------------------------
# Newcastle benchmark helpers
# -----------------------------------------------------------------------------

def source_newcastle_mask(
    df: pd.DataFrame,
) -> pd.Series:
    mask = pd.Series(
        False,
        index=df.index,
    )

    club_col = choose_column(
        df,
        [
            "Club",
            "club",
            "club_name",
        ],
    )

    if club_col is not None:
        mask |= (
            df[club_col]
            .astype(str)
            .str.contains(
                "Newcastle",
                case=False,
                na=False,
            )
        )

    for marker in [
        "NUFC",
        "NUFC Benchmark",
        "Newcastle Benchmark",
    ]:
        col = choose_column(
            df,
            [marker],
        )

        if col is not None:
            vals = df[col]

            valid = (
                vals.notna()
                & vals.astype(str)
                .str.strip()
                .ne("")
                & ~vals.astype(str)
                .str.lower()
                .isin(
                    {
                        "nan",
                        "none",
                        "<na>",
                    }
                )
            )

            mask |= valid

    return mask


def benchmark_name_candidates(
    full_name: str,
) -> set[str]:
    if (
        not isinstance(
            full_name,
            str,
        )
        or not full_name.strip()
    ):
        return set()

    bits = full_name.strip().split()

    variants = {
        normalize_key(full_name)
    }

    if len(bits) >= 2:
        variants.add(
            normalize_key(
                f"{bits[0][0]}. {bits[-1]}"
            )
        )
        variants.add(
            normalize_key(
                f"{bits[0][0]} {bits[-1]}"
            )
        )
        variants.add(
            normalize_key(
                bits[-1]
            )
        )

    return variants


def load_newcastle_benchmarks(
    project_root: Path,
) -> pd.DataFrame:
    search_dirs = [
        shortlist_root(project_root),
        (
            project_root
            / "data"
            / "processed"
            / "bsd"
        ),
    ]

    patterns = [
        "newcastle_jaissle_role_benchmarks.csv",
        "newcastle_jaissle_role_benchmarks.parquet",
        "*newcastle*benchmark*.csv",
        "*newcastle*benchmark*.parquet",
        "*retain*review*benchmark*.csv",
        "*retain*review*.csv",
        "*newcastle*historical*.csv",
        "*newcastle*historical*.parquet",
    ]

    for search_dir in search_dirs:
        for pattern in patterns:
            for path in sorted(
                search_dir.rglob(pattern)
            ):
                try:
                    df = read_table(path)
                except Exception:
                    continue

                player_col = choose_column(
                    df,
                    [
                        "Player",
                        "player_name",
                        "Player Name",
                    ],
                )
                role_col = choose_column(
                    df,
                    [
                        "Role",
                        "role",
                        "role_name",
                        "Role Name",
                        "role_key",
                    ],
                )
                score_col = choose_column(
                    df,
                    [
                        "jaissle_role_score",
                        "Role Score",
                        "Score",
                        "jaissle_score",
                        "Role Fit",
                        "role_fit",
                    ],
                )

                if not (
                    player_col
                    and role_col
                    and score_col
                ):
                    continue

                out = df.copy()

                out["_player"] = out[
                    player_col
                ]
                out["_role"] = out[
                    role_col
                ]
                out["_score"] = numeric(
                    out[score_col]
                )

                fit_col = choose_column(
                    out,
                    [
                        "role_fit",
                        "Role Fit",
                        "Fit",
                    ],
                )

                out["_fit"] = (
                    numeric(
                        out[fit_col]
                    )
                    if fit_col
                    else np.nan
                )

                role_key_col = choose_column(
                    out,
                    [
                        "role_key",
                        "Role Key",
                    ],
                )

                out["_role_key"] = (
                    out[role_key_col]
                    .astype(str)
                    if role_key_col
                    else out["_role"]
                    .map(normalize_key)
                )

                archetype_col = choose_column(
                    out,
                    [
                        "Archetype",
                        "Best Archetype",
                        "best_archetype",
                    ],
                )

                out["_archetype"] = (
                    out[archetype_col]
                    if archetype_col
                    else np.nan
                )

                value_col = choose_column(
                    out,
                    [
                        "Value €M",
                        "market_value_eur_m",
                        "market_value_eur",
                    ],
                )

                decision_col = choose_column(
                    out,
                    [
                        "Decision Flag",
                        "decision_flag",
                    ],
                )

                age_col = choose_column(
                    out,
                    [
                        "Age",
                        "age",
                    ],
                )

                out["_value"] = (
                    numeric(
                        out[value_col]
                    )
                    if value_col
                    else np.nan
                )

                out["_decision"] = (
                    out[decision_col]
                    if decision_col
                    else np.nan
                )

                out["_age"] = (
                    numeric(
                        out[age_col]
                    )
                    if age_col
                    else np.nan
                )

                out["_norm_role"] = (
                    out["_role"]
                    .map(normalize_key)
                )

                return out

    raise FileNotFoundError(
        "Could not find a Newcastle benchmark table."
    )


def benchmark_rows_for_spec(
    bench: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    if bench.empty:
        return bench.copy()

    targets = {
        normalize_key(x)
        for x in spec.get(
            "benchmark_labels",
            [spec["label"]],
        )
    }

    rows = bench[
        bench["_norm_role"]
        .isin(targets)
    ].copy()

    # Do not use Nico as the benchmark reference on winger charts.
    if spec.get("slug") in {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }:
        rows = rows[
            ~rows["_player"]
            .astype(str)
            .str.contains("Nico Gonzalez", case=False, na=False)
        ].copy()

    if spec.get("slug") in {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }:
        rows = rows[
            ~rows["_player"]
            .astype(str)
            .str.contains(
                "Nico Gonzalez|Nico González",
                case=False,
                na=False,
            )
        ].copy()

    return rows.sort_values(
        "_score",
        ascending=False,
    )



WINGER_NEWCASTLE_PLAYERS = {
    "Bazoumana Touré",
    "Bazoumana Toure",
    "B. Touré",
    "B. Toure",
    "Matias Fernandez-Pardo",
    "Matías Fernández-Pardo",
    "M. Fernandez-Pardo",
    "Anthony Elanga",
    "A. Elanga",
    "Harvey Barnes",
    "H. Barnes",
    "Jacob Murphy",
    "J. Murphy",
}


def restrict_newcastle_winger_rows(
    df: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    """On winger charts, only keep genuine Newcastle winger references."""
    if spec.get("slug") not in {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }:
        return df

    player_col = choose_column(
        df,
        ["Player", "player_name", "name"],
    )

    if player_col is None:
        return df.iloc[0:0].copy()

    allowed = {
        normalize_key(name)
        for name in WINGER_NEWCASTLE_PLAYERS
    }

    return df[
        df[player_col]
        .astype(str)
        .map(normalize_key)
        .isin(allowed)
    ].copy()


def force_include_newcastle_rows(
    source_df: pd.DataFrame,
    plotted_df: pd.DataFrame,
    bench_rows: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    src = source_df.copy()
    mask = source_newcastle_mask(src)

    player_col = choose_column(
        src,
        [
            "Player",
            "player_name",
            "name",
        ],
    )

    if (
        player_col is not None
        and bench_rows is not None
        and not bench_rows.empty
    ):
        bench_variants = set()

        for name in (
            bench_rows["_player"]
            .dropna()
            .astype(str)
        ):
            bench_variants.update(
                benchmark_name_candidates(name)
            )

        src_names = (
            src[player_col]
            .astype(str)
            .map(normalize_key)
        )

        mask |= src_names.isin(
            bench_variants
        )

    forced = src[
        mask
    ].copy()

    forced = restrict_newcastle_winger_rows(
        forced,
        spec,
    )

    if forced.empty:
        out = plotted_df.copy()
    else:
        out = pd.concat(
            [
                plotted_df,
                forced,
            ],
            axis=0,
        )

        out = out[
            ~out.index.duplicated(
                keep="first"
            )
        ].copy()

    out["_is_newcastle"] = (
        source_newcastle_mask(out)
        .fillna(False)
        .astype(bool)
    )

    if spec.get("slug") in {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }:
        allowed_rows = restrict_newcastle_winger_rows(
            out,
            spec,
        )

        allowed_idx = set(
            allowed_rows.index
        )

        out["_is_newcastle"] = (
            out.index.to_series()
            .map(lambda idx: idx in allowed_idx)
            .astype(bool)
        )

    return out


# -----------------------------------------------------------------------------
# Shortlist forcing / highlighting
# -----------------------------------------------------------------------------

def force_include_shortlist_rows(
    source_df: pd.DataFrame,
    plotted_df: pd.DataFrame,
    role_slug: str,
) -> pd.DataFrame:
    relevant = shortlist_for_role(
        role_slug
    )

    if not relevant:
        out = plotted_df.copy()
        out["_is_shortlist"] = False
        out["_shortlist_canonical"] = None
        return out

    player_col = choose_column(
        source_df,
        [
            "Player",
            "player_name",
            "name",
        ],
    )

    if player_col is None:
        out = plotted_df.copy()
        out["_is_shortlist"] = False
        out["_shortlist_canonical"] = None
        return out

    source = source_df.copy()

    source["_shortlist_canonical"] = (
        source[player_col]
        .map(
            canonical_shortlist_name
        )
    )

    relevant_set = set(
        relevant
    )

    forced = source[
        source["_shortlist_canonical"]
        .isin(relevant_set)
    ].copy()

    out = pd.concat(
        [
            plotted_df,
            forced,
        ],
        axis=0,
    )

    out = out[
        ~out.index.duplicated(
            keep="first"
        )
    ].copy()

    # IMPORTANT:
    # A shortlisted player may already be in the normal top-N market rows.
    # In that case the duplicate forced row is removed above, so we MUST
    # recompute shortlist identity on every surviving row. Otherwise only
    # players added by the force-include step get highlighted.
    out["_shortlist_canonical"] = (
        out[player_col]
        .map(
            canonical_shortlist_name
        )
    )

    out["_is_shortlist"] = (
        out["_shortlist_canonical"]
        .isin(relevant_set)
        .fillna(False)
        .astype(bool)
    )

    return out


def shortlist_status(
    canonical: object,
) -> str:
    if canonical is None or pd.isna(canonical):
        return ""

    return str(
        FINAL_SHORTLIST
        .get(
            str(canonical),
            {},
        )
        .get(
            "status",
            "",
        )
    )


def shortlist_fee(
    canonical: object,
) -> float:
    if canonical is None or pd.isna(canonical):
        return np.nan

    return pd.to_numeric(
        pd.Series([
            FINAL_SHORTLIST
            .get(
                str(canonical),
                {},
            )
            .get(
                "fee",
                np.nan,
            )
        ]),
        errors="coerce",
    ).iloc[0]


def plot_shortlisted_players(
    ax,
    df: pd.DataFrame,
) -> None:
    shortlisted = df[
        df["_is_shortlist"]
        .fillna(False)
    ].copy()

    if shortlisted.empty:
        return

    for status, marker, size, face in [
        (
            "CORE",
            "D",
            280,
            "crimson",
        ),
        (
            "OPTIONAL",
            "P",
            310,
            "deepskyblue",
        ),
    ]:
        group = shortlisted[
            shortlisted["_shortlist_canonical"]
            .map(shortlist_status)
            .eq(status)
        ].copy()

        if group.empty:
            continue

        ax.scatter(
            group["_x"],
            group["_y"],
            s=size,
            marker=marker,
            c=face,
            edgecolors="black",
            linewidths=1.6,
            zorder=9,
            label=(
                "Core shortlist"
                if status == "CORE"
                else "Optional shortlist"
            ),
        )

    winger_slugs = {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }

    winger_label_offsets = {
        "Kenan Yıldız": (14, 24),
        "Francisco Conceição": (18, 38),
    }

    for i, (_, row) in enumerate(
        shortlisted.iterrows()
    ):
        canonical = row[
            "_shortlist_canonical"
        ]

        fee = shortlist_fee(
            canonical
        )

        fee_text = (
            f"€{float(fee):.0f}m"
            if pd.notna(fee)
            else ""
        )

        role_slug = row.get("_role_slug", "")
        is_winger_shortlist = role_slug in winger_slugs

        # Give Yıldız a stronger visual hierarchy on winger charts.
        if is_winger_shortlist and canonical == "Kenan Yıldız":
            ax.scatter(
                [row["_x"]],
                [row["_y"]],
                s=620,
                marker="o",
                facecolors="none",
                edgecolors="gold",
                linewidths=2.6,
                zorder=8.8,
            )

        if is_winger_shortlist and canonical in winger_label_offsets:
            extra_x, extra_y = winger_label_offsets[canonical]
        elif is_winger_shortlist:
            extra_x = 8 if i % 2 == 0 else -6
            extra_y = 8 + (i % 4) * 6
        else:
            extra_x = 0
            extra_y = 0

        ax.annotate(
            f"{canonical}\n{fee_text}",
            (
                row["_x"],
                row["_y"],
            ),
            xytext=(
                10 + extra_x,
                12 + (i % 3) * 8 + extra_y,
            ),
            textcoords="offset points",
            fontsize=10.0 if canonical == "Kenan Yıldız" else 9.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.30",
                fc="white",
                ec=("goldenrod" if canonical == "Kenan Yıldız" else "black"),
                lw=(1.8 if canonical == "Kenan Yıldız" else 1.1),
                alpha=0.97,
            ),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.9,
                alpha=0.82,
            ),
            zorder=10,
        )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def build_annotation_mask(
    df: pd.DataFrame,
    annotate_n: int,
) -> pd.Series:
    mask = pd.Series(
        False,
        index=df.index,
    )

    non_shortlist = df[
        ~df["_is_shortlist"]
        .fillna(False)
    ].copy()

    mask.loc[
        non_shortlist.head(
            min(
                annotate_n,
                len(non_shortlist),
            )
        ).index
    ] = True

    if "_value" in non_shortlist.columns:
        value_eff = (
            non_shortlist["_score"]
            / non_shortlist["_value"]
            .replace(
                0,
                np.nan,
            )
        )

        extra = (
            value_eff
            .sort_values(
                ascending=False
            )
            .head(5)
            .index
        )

        mask.loc[
            extra
        ] = True

    if len(non_shortlist) >= 8:
        mask.loc[
            non_shortlist["_x"]
            .nlargest(3)
            .index
        ] = True
        mask.loc[
            non_shortlist["_y"]
            .nlargest(3)
            .index
        ] = True
        mask.loc[
            non_shortlist["_score"]
            .nlargest(6)
            .index
        ] = True

    return mask.fillna(
        False
    )


def annotate_points(
    ax,
    df: pd.DataFrame,
    label_mask: pd.Series,
) -> None:
    subset = df[
        label_mask
    ].copy()

    if subset.empty:
        return

    for i, (_, row) in enumerate(
        subset.iterrows()
    ):
        x = row["_x"]
        y = row["_y"]
        name = safe_text(
            row["_player"]
        )

        dx = (
            0.18
            if i % 2 == 0
            else -0.18
        )
        dy = (
            0.24
            if i % 3 == 0
            else (
                -0.20
                if i % 3 == 1
                else 0.05
            )
        )

        ha = (
            "left"
            if dx > 0
            else "right"
        )

        ax.annotate(
            name,
            (
                x,
                y,
            ),
            xytext=(
                x + dx,
                y + dy,
            ),
            textcoords="data",
            fontsize=8,
            ha=ha,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.4,
                alpha=0.45,
            ),
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="none",
                alpha=0.72,
            ),
        )


def draw_league_legend(
    ax,
    df: pd.DataFrame,
):
    if "_league" not in df.columns:
        return

    leagues = [
        v
        for v in sorted(
            df["_league"]
            .dropna()
            .astype(str)
            .unique()
        )
    ]

    if not leagues:
        return

    handles = []

    for lg in leagues:
        color = league_to_color(
            pd.Series([lg])
        )[0]

        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=lg,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
            )
        )

    ax.legend(
        handles=handles,
        title="League",
        fontsize=8,
        loc="best",
    )


def draw_newcastle_panel(
    ax,
    bench_rows: pd.DataFrame,
    threshold: float,
    spec: dict,
):
    ax.clear()

    ax.set_title(
        "Newcastle benchmarks",
        fontsize=12,
        pad=10,
    )

    if bench_rows.empty:
        ax.text(
            0.5,
            0.5,
            "No benchmark rows found",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    plot = (
        bench_rows
        .dropna(
            subset=[
                "_player",
                "_score",
            ]
        )
        .copy()
    )

    if plot.empty:
        ax.text(
            0.5,
            0.5,
            "No usable benchmark rows",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    plot = plot.sort_values(
        "_score",
        ascending=True,
    )

    y = np.arange(
        len(plot)
    )

    colors = [
        (
            "tab:green"
            if s >= threshold
            else "tab:orange"
        )
        for s in plot["_score"]
    ]

    ax.barh(
        y,
        plot["_score"],
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.78,
    )

    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot["_player"],
        fontsize=8,
    )
    ax.set_xlabel(
        "Role score"
    )
    ax.grid(
        True,
        axis="x",
        alpha=0.22,
    )

    max_x = (
        max(
            float(
                plot["_score"].max()
            ),
            threshold,
        )
        + 6
    )

    ax.set_xlim(
        0,
        max(
            70,
            max_x,
        ),
    )

    for yi, (_, row) in enumerate(
        plot.iterrows()
    ):
        score = float(
            row["_score"]
        )

        ax.text(
            score + 0.4,
            yi,
            f"{score:.1f}",
            va="center",
            fontsize=8,
        )


# -----------------------------------------------------------------------------
# Main role chart
# -----------------------------------------------------------------------------

def create_role_chart(
    project_root: Path,
    spec: dict,
    top_n: int = 40,
    annotate_n: int = 18,
    price_cap: Optional[float] = None,
    threshold: float = 52.5,
    benchmarks: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    try:
        (
            original_df,
            src,
            xcol_used,
            ycol_used,
        ) = load_role_dataframe(
            project_root,
            spec,
        )
    except Exception as exc:
        print(
            f"[skip] {spec['label']}: {exc}"
        )
        return None

    (
        original_df,
        x_label_override,
        y_label_override,
    ) = add_independent_plot_metrics(
        original_df,
        spec,
    )

    bench_rows = (
        benchmark_rows_for_spec(
            benchmarks,
            spec,
        )
        if benchmarks is not None
        else pd.DataFrame()
    )

    eligible = (
        original_df
        .dropna(
            subset=[
                "_score",
                "_x",
                "_y",
            ]
        )
        .copy()
    )

    if price_cap is not None:
        market_mask = (
            eligible["_value"].isna()
            | (
                eligible["_value"]
                <= price_cap
            )
        )
    else:
        market_mask = pd.Series(
            True,
            index=eligible.index,
        )

    nufc_mask = source_newcastle_mask(
        eligible
    )

    external = eligible[
        market_mask
        & ~nufc_mask
    ].copy()

    external = (
        external
        .sort_values(
            [
                "_score",
                "_x",
                "_y",
            ],
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    plotted = force_include_newcastle_rows(
        eligible,
        external,
        bench_rows,
        spec,
    )

    plotted = force_include_shortlist_rows(
        eligible,
        plotted,
        spec["slug"],
    )

    if plotted.empty:
        print(
            f"[skip] {spec['label']}: no rows after filtering"
        )
        return None

    # Rebuild boolean flags after concatenating Newcastle and shortlist rows.
    # Concat can upcast boolean columns to float/object when new rows contain NaN,
    # which breaks expressions such as ~plotted["_is_newcastle"].
    plotted["_is_newcastle"] = (
        source_newcastle_mask(plotted)
        .fillna(False)
        .astype(bool)
    )

    if "_is_shortlist" not in plotted.columns:
        plotted["_is_shortlist"] = False
    else:
        plotted["_is_shortlist"] = (
            plotted["_is_shortlist"]
            .fillna(False)
            .astype(bool)
        )

    plotted["_role_slug"] = spec.get("slug", "")

    plotted["_bubble"] = (
        220.0
        if spec.get("slug") == "cam"
        else bubble_sizes(
            plotted["_value"]
        )
    )

    plotted["_color"] = (
        league_to_color(
            plotted["_league"]
        )
    )

    is_winger_chart = spec.get("slug") in {
        "wide_creator",
        "vertical_box_winger",
        "overall_winger",
    }

    if not bench_rows.empty:
        fig, (
            ax,
            bx,
        ) = plt.subplots(
            1,
            2,
            figsize=(
                23.6 if is_winger_chart else 20.5,
                10.2 if is_winger_chart else 8.8,
            ),
            gridspec_kw={
                "width_ratios": [
                    5.45 if is_winger_chart else 5.0,
                    1.45 if is_winger_chart else 1.55,
                ],
                "wspace": 0.18,
            },
        )
    else:
        fig, ax = plt.subplots(
            figsize=(
                19.4 if is_winger_chart else 16.5,
                9.8 if is_winger_chart else 8.6,
            )
        )
        bx = None

    market = plotted[
        (~plotted["_is_newcastle"])
        & (~plotted["_is_shortlist"])
    ].copy()

    nufc = plotted[
        plotted["_is_newcastle"]
    ].copy()

    shortlisted = plotted[
        plotted["_is_shortlist"]
    ].copy()

    if not market.empty:
        ax.scatter(
            market["_x"],
            market["_y"],
            s=market["_bubble"],
            c=market["_color"],
            alpha=0.64,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )

    if not nufc.empty:
        ax.scatter(
            nufc["_x"],
            nufc["_y"],
            s=np.maximum(
                nufc["_bubble"],
                420,
            ),
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=1.3,
            label="Newcastle benchmark",
            zorder=6,
        )

    plot_shortlisted_players(
        ax,
        shortlisted,
    )

    if (
        spec.get("slug")
        not in {
            "left_wb",
            "right_wb",
            "wide_creator",
            "vertical_box_winger",
        }
    ):
        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )

    label_mask = (
        build_annotation_mask(
            market,
            annotate_n=annotate_n,
        )
        if not market.empty
        else pd.Series(
            False,
            index=market.index,
        )
    )

    annotate_points(
        ax,
        market,
        label_mask,
    )

    if not nufc.empty:
        for i, (_, row) in enumerate(
            nufc.iterrows()
        ):
            ax.annotate(
                safe_text(
                    row["_player"]
                ),
                (
                    row["_x"],
                    row["_y"],
                ),
                xytext=(
                    7,
                    7 + (i % 3) * 5,
                ),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="#fff5bf",
                    ec="black",
                    alpha=0.9,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.55,
                    alpha=0.7,
                ),
                zorder=7,
            )

    # Give winger charts more breathing room so shortlist labels do not feel cramped.
    if is_winger_chart:
        combined = plotted[
            plotted[["_x", "_y"]].notna().all(axis=1)
        ].copy()

        if not combined.empty:
            xmin = float(combined["_x"].min())
            xmax = float(combined["_x"].max())
            ymin = float(combined["_y"].min())
            ymax = float(combined["_y"].max())

            xpad = max(2.2, 0.09 * (xmax - xmin if xmax > xmin else 10.0))
            ypad = max(1.8, 0.12 * (ymax - ymin if ymax > ymin else 10.0))

            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

    if not market.empty:
        x_med = float(
            market["_x"].median()
        )
        y_med = float(
            market["_y"].median()
        )

        if np.isfinite(x_med):
            ax.axvline(
                x_med,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )

        if np.isfinite(y_med):
            ax.axhline(
                y_med,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )

    role_shortlist = shortlist_for_role(
        spec["slug"]
    )

    shortlist_text = (
        " | Shortlist: "
        + ", ".join(role_shortlist)
        if role_shortlist
        else ""
    )

    ax.set_title(
        (
            f"{spec['label']} — External Market vs Newcastle"
            f"{shortlist_text}"
        ),
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel(
        x_label_override
    )
    ax.set_ylabel(
        y_label_override
    )
    ax.grid(
        True,
        alpha=0.23,
    )

    # Build combined legend manually.
    handles, labels = ax.get_legend_handles_labels()

    if not nufc.empty:
        if "Newcastle benchmark" not in labels:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    label="Newcastle benchmark",
                    markerfacecolor="gold",
                    markeredgecolor="black",
                    markersize=14,
                )
            )
            labels.append(
                "Newcastle benchmark"
            )

    if (
        not shortlisted[
            shortlisted[
                "_shortlist_canonical"
            ]
            .map(shortlist_status)
            .eq("CORE")
        ].empty
        and "Core shortlist" not in labels
    ):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                label="Core shortlist",
                markerfacecolor="crimson",
                markeredgecolor="black",
                markersize=10,
            )
        )
        labels.append(
            "Core shortlist"
        )

    if (
        not shortlisted[
            shortlisted[
                "_shortlist_canonical"
            ]
            .map(shortlist_status)
            .eq("OPTIONAL")
        ].empty
        and "Optional shortlist" not in labels
    ):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="P",
                color="w",
                label="Optional shortlist",
                markerfacecolor="deepskyblue",
                markeredgecolor="black",
                markersize=11,
            )
        )
        labels.append(
            "Optional shortlist"
        )

    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            title="Key",
            fontsize=8,
            loc="best",
        )

    shortlist_note = ""

    if not shortlisted.empty:
        details = []

        for canonical in (
            shortlisted[
                "_shortlist_canonical"
            ]
            .dropna()
            .unique()
        ):
            info = FINAL_SHORTLIST.get(
                canonical,
                {},
            )

            details.append(
                f"{canonical} ({info.get('status', '')}, €{info.get('fee', np.nan):.0f}m)"
            )

        shortlist_note = (
            " Highlighted targets: "
            + "; ".join(details)
            + "."
        )

    caption = (
        f"Source: {src.name}\n"
        f"{spec['caption']}"
        f"{shortlist_note} "
        f"Shortlisted players are force-included even if outside the top-{top_n} cut or price cap."
    )

    ax.text(
        0.01,
        -0.17,
        caption,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="#f5f5f5",
            ec="#cccccc",
        ),
    )

    if bx is not None:
        draw_newcastle_panel(
            bx,
            bench_rows,
            threshold=threshold,
            spec=spec,
        )

    outpath = (
        visualization_dir(
            project_root,
            spec.get(
                "folder",
                spec["slug"],
            ),
        )
        / f"{spec['slug']}_top_targets_chart.png"
    )

    fig.tight_layout(
        rect=[
            0,
            0.045,
            1,
            1,
        ]
    )

    fig.savefig(
        outpath,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    shortlist_found = (
        shortlisted[
            "_shortlist_canonical"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if role_shortlist:
        missing = [
            player
            for player in role_shortlist
            if player not in shortlist_found
        ]

        if missing:
            print(
                f"[warn] {spec['label']} shortlist player(s) not found in source: "
                + ", ".join(missing)
            )

    print(
        f"[saved] {outpath}"
    )

    return outpath


# -----------------------------------------------------------------------------
# Squad summary visuals
# -----------------------------------------------------------------------------

SQUAD_ROLE_COLUMNS = [
    "High-Line CB",
    "Left WB / FB",
    "Right WB / FB",
    "Holding Controller",
    "Advanced / Workhorse 8",
    "Creative 10",
    "Wide Creator",
    "Vertical / Box Winger",
    "Hybrid Winger",
    "Focal Striker",
]



def create_squad_depth_visualization(
    project_root: Path,
) -> Optional[Path]:
    path = (
        shortlist_root(project_root)
        / "squad_depth_analysis.csv"
    )

    if not path.exists():
        print(
            "[skip] Squad depth: run build_squad_depth_analysis.py first"
        )
        return None

    df = pd.read_csv(path)

    # Older squad_depth_analysis.csv files may still contain the goalkeeper row.
    # Filter it out here so the visualization stays aligned with the outfield-only
    # squad review even before the depth-analysis builder is rerun.
    if "Role" in df.columns:
        df = df[
            ~df["Role"]
            .astype(str)
            .str.contains("GK|Goalkeeper", case=False, na=False)
        ].copy()

    roles = df["Role"].tolist()
    y = np.arange(
        len(df)
    )
    h = 0.22

    fig, ax = plt.subplots(
        figsize=(
            14,
            max(
                8,
                0.75 * len(df) + 2,
            ),
        )
    )

    for i, label in enumerate(
        [
            "Starter",
            "2nd choice",
            "3rd choice",
        ]
    ):
        vals = pd.to_numeric(
            df[
                f"Score {i + 1}"
            ],
            errors="coerce",
        )

        ax.barh(
            y + (i - 1) * h,
            vals.fillna(0),
            height=h,
            label=label,
            alpha=0.78,
        )

        for yi, v in enumerate(vals):
            if pd.notna(v):
                ax.text(
                    v + 0.25,
                    yi + (i - 1) * h,
                    f"{v:.1f}",
                    va="center",
                    fontsize=8,
                )

    ax.axvline(
        52.5,
        color="red",
        linestyle="--",
        lw=1.1,
        label="52.5 threshold",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        roles
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Role score"
    )
    ax.set_title(
        "Newcastle Tactical Depth by Role"
    )
    ax.grid(
        True,
        axis="x",
        alpha=0.22,
    )
    ax.legend(
        loc="lower right"
    )

    out = (
        visualization_dir(
            project_root,
            "squad",
        )
        / "newcastle_tactical_role_depth.png"
    )

    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        top=0.92,
        bottom=0.10,
    )

    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[saved] {out}"
    )

    return out


def create_value_efficiency_visualization(
    project_root: Path,
) -> Optional[Path]:
    path = (
        shortlist_root(project_root)
        / "recruitment_value_analysis.csv"
    )

    if not path.exists():
        print(
            "[skip] Value analysis: run build_recruitment_value_analysis.py first"
        )
        return None

    df = pd.read_csv(path)

    if (
        "Upgrade" not in df.columns
        or "Value Score" not in df.columns
    ):
        return None

    df["Upgrade"] = pd.to_numeric(
        df["Upgrade"],
        errors="coerce",
    )
    df["Value Score"] = pd.to_numeric(
        df["Value Score"],
        errors="coerce",
    )

    work = (
        df[
            df["Upgrade"].gt(0)
            & df["Value Score"].notna()
        ]
        .sort_values(
            "Value Score",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    if work.empty:
        return None

    labels = (
        work["Player"]
        .astype(str)
        + " — "
        + work["Role"]
        .astype(str)
    ).tolist()

    y = np.arange(
        len(work)
    )

    fig, ax = plt.subplots(
        figsize=(
            13,
            max(
                8,
                0.38 * len(work) + 2,
            ),
        )
    )

    ax.barh(
        y,
        work["Value Score"],
        alpha=0.8,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Recruitment Value Score"
    )
    ax.set_title(
        "Recruitment Value — Positive Upgrades"
    )
    ax.grid(
        True,
        axis="x",
        alpha=0.22,
    )

    out = (
        visualization_dir(
            project_root,
            "value",
        )
        / "recruitment_value_score.png"
    )

    fig.subplots_adjust(
        left=0.35,
        right=0.97,
        top=0.92,
        bottom=0.10,
    )

    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[saved] {out}"
    )

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Jaissle recruitment visualization pack."
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help=(
            "How many external market players to plot per role."
        ),
    )

    parser.add_argument(
        "--annotate-n",
        type=int,
        default=18,
        help=(
            "How many non-shortlist market names to annotate."
        ),
    )

    parser.add_argument(
        "--price-cap",
        type=float,
        default=100.0,
        help=(
            "External target price cap in €M. "
            "Shortlisted players are force-included even above this."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=52.5,
        help=(
            "Role-fit threshold for benchmark context."
        ),
    )

    args = parser.parse_args()

    project_root = find_project_root()
    out_root = visualizations_root(
        project_root
    )

    price_cap = (
        None
        if (
            args.price_cap is not None
            and args.price_cap < 0
        )
        else args.price_cap
    )

    print(
        "=" * 110
    )
    print(
        "JAISSLE IMAGE CREATION — FINAL SHORTLIST"
    )
    print(
        "=" * 110
    )
    print(
        f"Project root: {project_root}"
    )
    print(
        f"Output dir   : {out_root}"
    )

    print()
    print(
        "Shortlist highlighted:"
    )

    for player, info in FINAL_SHORTLIST.items():
        print(
            f"  {info['status']:<8} "
            f"{player:<24} "
            f"€{info['fee']:.0f}m"
        )

    benchmarks = None

    try:
        benchmarks = load_newcastle_benchmarks(
            project_root
        )

        print()
        print(
            f"Loaded Newcastle benchmark rows: {len(benchmarks)}"
        )
    except Exception as exc:
        print(
            f"[warn] Could not load Newcastle benchmarks: {exc}"
        )

    saved = []

    for spec in ROLE_SPECS:
        path = create_role_chart(
            project_root=project_root,
            spec=spec,
            top_n=args.top_n,
            annotate_n=args.annotate_n,
            price_cap=price_cap,
            threshold=args.threshold,
            benchmarks=benchmarks,
        )

        if path is not None:
            saved.append(path)

    depth_chart = create_squad_depth_visualization(
        project_root
    )

    if depth_chart is not None:
        saved.append(
            depth_chart
        )

    value_chart = create_value_efficiency_visualization(
        project_root
    )

    if value_chart is not None:
        saved.append(
            value_chart
        )

    print()
    print(
        "=" * 110
    )
    print(
        "SAVED FILES"
    )
    print(
        "=" * 110
    )

    if not saved:
        print(
            "No images were generated."
        )
    else:
        for path in saved:
            print(
                path
            )


if __name__ == "__main__":
    main()
