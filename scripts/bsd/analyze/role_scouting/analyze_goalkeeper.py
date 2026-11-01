#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").exists() and (parent / "scripts").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


ROOT = project_root()
OUT = ROOT / "data" / "processed" / "bsd" / "shortlist"
OUT.mkdir(parents=True, exist_ok=True)

# Manual/current data supplied for this project. Missing keepers intentionally stay unscored.
GK_ROWS = [
    {
        "Player": "Lukáš Horníček", "Club": "Newcastle United", "Season": "2026/27",
        "Role": "Starter", "Trust": "High", "Future": "Keep",
        "Saves/90": 3.38, "Save %": 65.9, "Goals Conceded/90": 1.75,
        "Goals Prevented": 0.34, "Clean Sheets/90": 0.25, "Sweeper Actions/90": 1.13,
        "High Claims/90": 0.13, "Pass %": 63.8, "Acc Long Balls/90": 4.50,
        "Long Ball %": 29.3, "Errors to Goal/90": 0.0,
    },
    {
        "Player": "Édouard Mendy", "Club": "Al-Ahli", "Season": "2025/26 Jaissle reference",
        "Role": "Reference", "Trust": "Reference", "Future": "Reference",
        "Saves/90": 1.77, "Save %": 71.9, "Goals Conceded/90": 0.69,
        "Goals Prevented": 0.08, "Clean Sheets/90": 0.54, "Sweeper Actions/90": 0.31,
        "High Claims/90": 0.50, "Pass %": 77.1, "Acc Long Balls/90": 5.85,
        "Long Ball %": 50.3, "Errors to Goal/90": 0.0,
    },
    {"Player": "Nick Pope", "Club": "Newcastle United", "Season": "2026/27", "Role": "Backup", "Trust": "Medium", "Future": "Keep / monitor"},
    {"Player": "Jaouen", "Club": "Newcastle United", "Season": "2026/27", "Role": "Developmental", "Trust": "Low / unknown", "Future": "Develop"},
    {"Player": "Mark Gillespie", "Club": "Newcastle United", "Season": "2026/27", "Role": "Emergency", "Trust": "Medium-low", "Future": "Emergency depth"},
]


def minmax_100(series: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2 or np.isclose(s.max(), s.min()):
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    out = 40 + 20 * (s - s.min()) / (s.max() - s.min())
    if not higher_better:
        out = 80 - out
    return out


def main() -> None:
    df = pd.DataFrame(GK_ROWS)
    numeric_cols = [
        "Saves/90", "Save %", "Goals Conceded/90", "Goals Prevented", "Clean Sheets/90",
        "Sweeper Actions/90", "High Claims/90", "Pass %", "Acc Long Balls/90",
        "Long Ball %", "Errors to Goal/90",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Scores are deliberately relative to the small reference set for now.
    save_pct = minmax_100(df["Save %"])
    prevented = minmax_100(df["Goals Prevented"])
    errors = minmax_100(df["Errors to Goal/90"], higher_better=False)
    sweep = minmax_100(df["Sweeper Actions/90"])
    claims = minmax_100(df["High Claims/90"])
    pass_pct = minmax_100(df["Pass %"])
    long_pct = minmax_100(df["Long Ball %"])

    df["Shot Stopping"] = 0.50 * save_pct + 0.40 * prevented + 0.10 * errors
    df["Sweeper"] = sweep
    df["Distribution"] = 0.55 * pass_pct + 0.45 * long_pct
    df["Box Command"] = claims
    df["Jaissle GK Score"] = (
        0.35 * df["Shot Stopping"]
        + 0.25 * df["Sweeper"]
        + 0.25 * df["Distribution"]
        + 0.15 * df["Box Command"]
    )

    # Do not fabricate scores for manual-only rows.
    has_stats = df["Save %"].notna() & df["Pass %"].notna()
    score_cols = ["Shot Stopping", "Sweeper", "Distribution", "Box Command", "Jaissle GK Score"]
    df.loc[~has_stats, score_cols] = np.nan
    df["Model Status"] = np.where(has_stats, "MODELED / SMALL REFERENCE SET", "MANUAL CONTEXT ONLY")

    csv = OUT / "goalkeeper_jaissle_detailed.csv"
    html = OUT / "goalkeeper_jaissle_detailed.html"
    df.to_csv(csv, index=False)
    df.to_html(html, index=False)

    print("=" * 110)
    print("JAISSLE GOALKEEPER PROFILE")
    print("=" * 110)
    print(df[["Player", "Role", "Trust", "Shot Stopping", "Sweeper", "Distribution", "Box Command", "Jaissle GK Score", "Model Status"]].to_string(index=False))
    print("\nSaved:")
    print(" ", csv)
    print(" ", html)


if __name__ == "__main__":
    main()
