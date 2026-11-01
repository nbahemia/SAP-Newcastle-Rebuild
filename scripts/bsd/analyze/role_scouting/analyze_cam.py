from __future__ import annotations

from pathlib import Path
import math
import unicodedata
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SHORTLIST = ROOT / "data" / "processed" / "bsd" / "shortlist"
FULL = SHORTLIST / "europe_2025_26_full_role_shortlist.parquet"
ROLE_CSV = SHORTLIST / "creative_10.csv"
OUT_CSV = SHORTLIST / "creative_10_jaissle_detailed.csv"
OUT_HTML = SHORTLIST / "creative_10_jaissle_detailed.html"

MAX_VALUE_EUR_M = 70.0


def norm(value) -> str:
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return value


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_pool() -> pd.DataFrame:
    if FULL.exists():
        df = pd.read_parquet(FULL)
        if "role_key" in df.columns:
            cam = df[df["role_key"].astype(str).eq("creative_cam")].copy()
        elif "role" in df.columns:
            cam = df[
                df["role"].astype(str).str.contains("creative", case=False, na=False)
            ].copy()
        else:
            raise RuntimeError(
                f"{FULL.name} has neither role_key nor role column."
            )
    elif ROLE_CSV.exists():
        cam = pd.read_csv(ROLE_CSV)
    else:
        raise FileNotFoundError(
            "No Creative 10 shortlist found. Run "
            "build_europe_shortlist_2025_2026.py first."
        )

    if cam.empty:
        raise RuntimeError(
            "Creative 10 shortlist is empty. Check the creative_cam coverage "
            "gate in build_europe_shortlist_2025_2026.py."
        )

    # Canonicalize player name from any legacy/exported naming.
    if "player_name" not in cam.columns:
        name_col = first_existing(
            cam,
            ["Player", "player", "name", "tm_player_name"],
        )
        if name_col is None:
            raise RuntimeError(
                "CAM source has no recognizable player-name column. "
                f"Columns: {list(cam.columns)}"
            )
        cam["player_name"] = cam[name_col]

    if "league_slug" not in cam.columns:
        league_col = first_existing(cam, ["League", "league"])
        if league_col is None:
            cam["league_slug"] = pd.NA
        else:
            cam["league_slug"] = cam[league_col]

    club_col = first_existing(
        cam,
        [
            "display_club_name",
            "current_club_name",
            "club_name",
            "Club",
            "historical_club_name",
        ],
    )
    cam["club_display"] = cam[club_col] if club_col else pd.NA

    age_col = first_existing(cam, ["age", "Age"])
    value_col = first_existing(
        cam,
        ["market_value_eur_m", "Value €M", "value_eur_m"],
    )
    fit_col = first_existing(cam, ["role_fit", "Fit", "CAM Fit"])
    coverage_col = first_existing(
        cam,
        ["metric_coverage_pct", "Coverage", "coverage_pct"],
    )

    cam["age_display"] = (
        pd.to_numeric(cam[age_col], errors="coerce") if age_col else np.nan
    )
    cam["value_display"] = (
        pd.to_numeric(cam[value_col], errors="coerce") if value_col else np.nan
    )
    cam["cam_fit"] = (
        pd.to_numeric(cam[fit_col], errors="coerce") if fit_col else np.nan
    )
    cam["coverage_display"] = (
        pd.to_numeric(cam[coverage_col], errors="coerce")
        if coverage_col else np.nan
    )

    # Keep external targets only; preserve unknown values for inspection.
    external = ~cam["club_display"].astype(str).str.contains(
        "newcastle", case=False, na=False
    )
    cam = cam[external].copy()

    cam["_player_key"] = (
        cam["player_name"].astype(str).map(norm)
        + "|"
        + cam["league_slug"].astype(str)
    )

    return cam


def main() -> None:
    cam = load_pool()

    cam = cam.sort_values(
        ["cam_fit", "value_display"],
        ascending=[False, True],
        na_position="last",
    )

    cols = [
        "player_name",
        "club_display",
        "league_slug",
        "age_display",
        "cam_fit",
        "coverage_display",
        "value_display",
    ]
    available = [c for c in cols if c in cam.columns]
    out = cam[available].copy()

    out = out.rename(
        columns={
            "player_name": "Player",
            "club_display": "Club",
            "league_slug": "League",
            "age_display": "Age",
            "cam_fit": "CAM Fit",
            "coverage_display": "Coverage %",
            "value_display": "Value €M",
        }
    )

    print("=" * 140)
    print("TOP JAISSLE CREATIVE 10 TARGETS")
    print("=" * 140)
    print(out.head(30).to_string(index=False))

    out.to_csv(OUT_CSV, index=False)
    out.to_html(OUT_HTML, index=False)

    print()
    print("Saved:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_HTML}")


if __name__ == "__main__":
    main()
