#!/usr/bin/env python3
"""
plot_final_shortlist_vs_position.py

Guaranteed final-shortlist positional comparison charts.

IMPORTANT:
The highlighted Newcastle targets do NOT depend on surviving a role analyzer.
Every target has an explicit project fee + role score in this file, so every
shortlisted player is always plotted.

Gray circles  = positional market peers
Red diamond   = core buy
Orange diamond= core value buy
Blue X        = optional shortlist
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SHORTLIST_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"
OUT_DIR = ROOT / "visualizations" / "final_shortlist_vs_position"


# ======================================================================================
# EXPLICIT FINAL SHORTLIST
#
# role_scores are intentionally hard-coded from our final recruitment work.
# This guarantees that the visualization never drops a target just because a
# role analyzer filtered him out of its candidate CSV.
# ======================================================================================

SHORTLIST = {
    "Nikola Krstović": {
        "project_fee": 33.0,
        "status": "BUY",
        "priority": "CORE",
        "role_scores": {
            "Focal Striker": 62.58,
        },
    },
    "Kenan Yıldız": {
        "project_fee": 90.0,
        "status": "BUY",
        "priority": "CORE",
        "role_scores": {
            "Winger": 86.14,
        },
    },
    "Rayan Aït-Nouri": {
        "project_fee": 30.0,
        "status": "BUY",
        "priority": "CORE",
        "role_scores": {
            "Left Back": 60.94,
        },
    },
    "Lazar Samardžić": {
        "project_fee": 20.0,
        "status": "BUY",
        "priority": "CORE",
        "role_scores": {
            "Creative 10": 65.01,
        },
    },
    "Bart van Rooij": {
        "project_fee": 13.0,
        "status": "BUY",
        "priority": "CORE",
        "role_scores": {
            "Right Back": 59.28,
        },
    },
    "Filip Lissah": {
        "project_fee": 1.0,
        "status": "BUY",
        "priority": "CORE VALUE",
        "role_scores": {
            "Right Back": 57.88,
        },
    },
    "Chibuike Nwaiwu": {
        "project_fee": 10.0,
        "status": "SHORTLIST",
        "priority": "OPTIONAL",
        "role_scores": {
            "Center Back": 59.73,
        },
    },
    "Francisco Conceição": {
        "project_fee": 30.0,
        "status": "SHORTLIST",
        "priority": "OPTIONAL",
        "role_scores": {
            "Winger": 84.70,
            "Creative 10": 62.25,
        },
    },
}


ROLE_CONFIG = {
    "Focal Striker": {
        "folder": "striker",
        "patterns": [
            "focal_striker_jaissle_detailed.csv",
            "focal_striker.csv",
        ],
        "score_candidates": [
            "Jaissle ST",
            "jaissle_striker_score",
            "Combined Score",
            "Focal Fit",
            "Role Score",
        ],
    },
    "Creative 10": {
        "folder": "cam",
        "patterns": [
            "creative_10_jaissle_detailed.csv",
            "creative_10.csv",
        ],
        "score_candidates": [
            "CAM Fit",
            "Final CAM Score",
            "jaissle_cam_score",
            "Role Score",
            "role_fit",
        ],
    },
    "Left Back": {
        "folder": "fullback_left",
        "patterns": [
            "left_wingback_recruitment_progressive.csv",
            "left_wingback_recruitment_creation.csv",
            "wingback.csv",
        ],
        "score_candidates": [
            "Jaissle FB Score",
            "jaissle_fb_score",
            "WB Fit",
            "Role Score",
            "role_fit",
        ],
    },
    "Right Back": {
        "folder": "fullback_right",
        "patterns": [
            "right_wingback_recruitment_progressive.csv",
            "right_back_jaissle_detailed.csv",
            "wingback.csv",
        ],
        "score_candidates": [
            "Jaissle RB",
            "jaissle_fb_score",
            "RB Fit",
            "Role Score",
            "role_fit",
        ],
    },
    "Center Back": {
        "folder": "center_back",
        "patterns": [
            "center_back_jaissle_detailed.csv",
            "young_cb_recruitment.csv",
            "center_back.csv",
        ],
        "score_candidates": [
            "Jaissle CB",
            "jaissle_cb_score",
            "CB Fit",
            "Role Score",
        ],
    },
    "Winger": {
        "folder": "winger",
        "patterns": [
            "bona_fide_winger_recruitment.csv",
            "winger_jaissle_archetypes.csv",
            "winger_jaissle_extended.csv",
        ],
        "score_candidates": [
            "Winger Recruitment Score",
            "Best Score",
            "overall_winger_score",
            "Hybrid",
            "Creator",
        ],
    },
}


# ======================================================================================
# HELPERS
# ======================================================================================

def normalize(text: object) -> str:
    if text is None or pd.isna(text):
        return ""

    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return value.strip()


def choose_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {
        normalize(col): col
        for col in df.columns
    }

    for candidate in candidates:
        hit = lookup.get(normalize(candidate))

        if hit is not None:
            return hit

    return None


def choose_file(patterns: list[str]) -> Optional[Path]:
    for pattern in patterns:
        direct = SHORTLIST_DIR / pattern

        if "*" not in pattern and direct.exists():
            return direct

        hits = sorted(
            SHORTLIST_DIR.glob(pattern)
        )

        if hits:
            return hits[0]

    return None


def targets_for_role(role_name: str) -> list[str]:
    out = []

    for player, meta in SHORTLIST.items():
        if role_name in meta["role_scores"]:
            out.append(player)

    return out


def marker_style(player: str) -> dict:
    meta = SHORTLIST[player]

    if (
        meta["status"] == "BUY"
        and meta["priority"] == "CORE VALUE"
    ):
        return {
            "marker": "D",
            "color": "darkorange",
            "label": "Core value buy",
        }

    if meta["status"] == "BUY":
        return {
            "marker": "D",
            "color": "crimson",
            "label": "Core buy",
        }

    return {
        "marker": "X",
        "color": "deepskyblue",
        "label": "Optional shortlist",
    }


def load_peer_market(
    role_name: str,
) -> tuple[pd.DataFrame, Path]:
    config = ROLE_CONFIG[role_name]
    path = choose_file(
        config["patterns"]
    )

    if path is None:
        raise FileNotFoundError(
            f"No positional file found for {role_name}"
        )

    df = pd.read_csv(path)

    player_col = choose_column(
        df,
        [
            "Player",
            "player_name",
            "name",
        ],
    )

    score_col = choose_column(
        df,
        config["score_candidates"],
    )

    value_col = choose_column(
        df,
        [
            "Value €M",
            "Current Value €M",
            "market_value_eur_m",
            "Value",
        ],
    )

    if player_col is None:
        raise KeyError(
            f"{path.name}: no player column"
        )

    if score_col is None:
        raise KeyError(
            f"{path.name}: no compatible role-score column"
        )

    peers = pd.DataFrame({
        "Player": df[player_col].astype(str),
        "Score": pd.to_numeric(
            df[score_col],
            errors="coerce",
        ),
    })

    if value_col is not None:
        peers["Value €M"] = pd.to_numeric(
            df[value_col],
            errors="coerce",
        )
    else:
        peers["Value €M"] = np.nan

    peers = (
        peers
        .dropna(subset=["Score"])
        .sort_values(
            "Score",
            ascending=False,
        )
        .copy()
    )

    return peers, path


# ======================================================================================
# CHART
# ======================================================================================

def create_chart(
    role_name: str,
) -> Optional[Path]:
    targets = targets_for_role(
        role_name
    )

    if not targets:
        return None

    try:
        peers, source = load_peer_market(
            role_name
        )
    except Exception as exc:
        print(
            f"[skip] {role_name}: {exc}"
        )
        return None

    # Remove our targets from the ordinary peer cloud.
    target_keys = {
        normalize(player)
        for player in targets
    }

    peers = peers[
        ~peers["Player"]
        .astype(str)
        .map(normalize)
        .isin(target_keys)
    ].copy()

    peers = peers.head(
        60
    ).copy()

    priced = peers[
        peers["Value €M"].notna()
    ].copy()

    fig, ax = plt.subplots(
        figsize=(13.2, 8.0)
    )

    if not priced.empty:
        ax.scatter(
            priced["Value €M"],
            priced["Score"],
            s=85,
            c="lightgray",
            alpha=0.45,
            edgecolors="black",
            linewidths=0.35,
            label="Positional market",
            zorder=1,
        )

    # Label top market peers + value outliers.
    label_peers = priced.head(
        12
    ).copy()

    if not priced.empty:
        efficiency = (
            priced["Score"]
            / priced["Value €M"]
            .replace(0, np.nan)
        )

        extra = priced.loc[
            efficiency
            .nlargest(5)
            .index
        ]

        label_peers = (
            pd.concat([
                label_peers,
                extra,
            ])
            .drop_duplicates(
                "Player"
            )
        )

    for i, (_, row) in enumerate(
        label_peers.iterrows()
    ):
        ax.annotate(
            str(row["Player"]),
            (
                row["Value €M"],
                row["Score"],
            ),
            xytext=(
                4,
                4 + (i % 3) * 5,
            ),
            textcoords="offset points",
            fontsize=8,
            alpha=0.72,
        )

    # ------------------------------------------------------------------
    # ALWAYS plot every shortlisted target for this role.
    # No lookup / matching / filtering is involved here.
    # ------------------------------------------------------------------

    used_labels = set()

    for i, player in enumerate(targets):
        meta = SHORTLIST[player]
        score = float(
            meta["role_scores"][role_name]
        )
        fee = float(
            meta["project_fee"]
        )
        style = marker_style(
            player
        )

        legend_label = (
            style["label"]
            if style["label"] not in used_labels
            else None
        )

        used_labels.add(
            style["label"]
        )

        ax.scatter(
            [fee],
            [score],
            s=380,
            marker=style["marker"],
            c=style["color"],
            edgecolors="black",
            linewidths=2.0,
            zorder=10,
            label=legend_label,
        )

        ax.annotate(
            f"{player}\n€{fee:.0f}m | {score:.1f}",
            (
                fee,
                score,
            ),
            xytext=(
                11,
                13 + (i % 3) * 8,
            ),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.30",
                fc="white",
                ec=style["color"],
                lw=1.7,
                alpha=0.98,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=style["color"],
                lw=1.0,
            ),
            zorder=11,
        )

    if not priced.empty:
        ax.axhline(
            priced["Score"].median(),
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
        )

        ax.axvline(
            priced["Value €M"].median(),
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_title(
        f"{role_name} — Final Shortlist vs Positional Market",
        fontsize=15,
        pad=14,
    )

    ax.set_xlabel(
        "Market value / Newcastle project fee (€M)"
    )
    ax.set_ylabel(
        "Role score"
    )

    ax.grid(
        True,
        alpha=0.22,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique = {}

    for handle, label in zip(
        handles,
        labels,
    ):
        if label and label not in unique:
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="best",
    )

    ax.text(
        0.01,
        -0.13,
        (
            f"Peer source: {source.name}. "
            f"Shortlist targets on this chart: {', '.join(targets)}. "
            "Highlighted targets use Newcastle's agreed project fee and final role score."
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    out_folder = (
        OUT_DIR
        / ROLE_CONFIG[role_name]["folder"]
    )

    out_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = (
        out_folder
        / (
            normalize(role_name)
            .replace(" ", "_")
            + "_shortlist_vs_market.png"
        )
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[OK] {role_name}: "
        + ", ".join(targets)
    )
    print(
        f"[saved] {out_file}"
    )

    return out_file


# ======================================================================================
# MASTER SHORTLIST SUMMARY
# ======================================================================================

def create_master_chart() -> Path:
    rows = []

    for player, meta in SHORTLIST.items():
        for role, score in meta["role_scores"].items():
            rows.append({
                "Player": player,
                "Role": role,
                "Fee": meta["project_fee"],
                "Score": score,
                "Status": meta["status"],
                "Priority": meta["priority"],
            })

    df = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(13.5, 8.0)
    )

    used_labels = set()

    for i, (_, row) in enumerate(
        df.iterrows()
    ):
        player = row["Player"]
        style = marker_style(
            player
        )

        legend_label = (
            style["label"]
            if style["label"] not in used_labels
            else None
        )
        used_labels.add(
            style["label"]
        )

        ax.scatter(
            [row["Fee"]],
            [row["Score"]],
            s=360,
            marker=style["marker"],
            c=style["color"],
            edgecolors="black",
            linewidths=1.9,
            label=legend_label,
            zorder=5,
        )

        ax.annotate(
            f"{player}\n{row['Role']}",
            (
                row["Fee"],
                row["Score"],
            ),
            xytext=(
                8,
                8 + (i % 3) * 7,
            ),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Newcastle Final Shortlist — Fee vs Role Score",
        fontsize=15,
        pad=14,
    )
    ax.set_xlabel(
        "Project fee (€M)"
    )
    ax.set_ylabel(
        "Role score"
    )
    ax.grid(
        True,
        alpha=0.22,
    )
    ax.legend(
        loc="best"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        OUT_DIR
        / "all_shortlisted_players.png"
    )

    fig.tight_layout()
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


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:
    print(
        "=" * 110
    )
    print(
        "FINAL SHORTLIST VS POSITIONAL MARKET — GUARANTEED TARGET PLOTTING"
    )
    print(
        "=" * 110
    )

    print(
        "\nTargets:"
    )

    for player, meta in SHORTLIST.items():
        roles = ", ".join(
            meta["role_scores"].keys()
        )

        print(
            f"  {meta['priority']:<10} "
            f"{player:<24} "
            f"€{meta['project_fee']:>5.1f}m  "
            f"{roles}"
        )

    saved = []

    for role_name in ROLE_CONFIG:
        path = create_chart(
            role_name
        )

        if path is not None:
            saved.append(
                path
            )

    saved.append(
        create_master_chart()
    )

    print()
    print(
        "=" * 110
    )
    print(
        "COMPLETE"
    )
    print(
        "=" * 110
    )
    print(
        f"Saved {len(saved)} chart(s)."
    )


if __name__ == "__main__":
    main()
