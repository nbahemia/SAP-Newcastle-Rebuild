#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


def root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "data").exists() and (parent / "scripts").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


ROOT = root()
SHORT = ROOT / "data" / "processed" / "bsd" / "shortlist"
SOURCE = SHORT / "winger_jaissle_archetypes.csv"
OUT = SHORT / "winger_jaissle_extended.csv"


def norm(s: object) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(s)).split())


def col(df: pd.DataFrame, names: Iterable[str]):
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lookup:
            return lookup[norm(n)]
    return None


def zscore_50_10(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 3 or np.isclose(x.std(ddof=0), 0):
        return pd.Series(np.where(x.notna(), 50.0, np.nan), index=x.index)
    z = (x - x.mean()) / x.std(ddof=0)
    return (50 + 10 * z).clip(20, 80)


def composite(df: pd.DataFrame, spec: list[tuple[list[str], float]]) -> pd.Series:
    parts = []
    weights = []
    for aliases, w in spec:
        c = col(df, aliases)
        if c is None:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() < 3 or np.isclose(vals.std(ddof=0), 0):
            continue
        parts.append(zscore_50_10(vals) * w)
        weights.append(w)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    num = sum(parts)
    den = sum(weights)
    return num / den


def find_hist() -> Path | None:
    candidates = [
        ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_season_metrics.parquet",
        ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_percentiles_enriched.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def repair_raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    hist_path = find_hist()
    if hist_path is None:
        return df
    hist = pd.read_parquet(hist_path)

    player_src = col(df, ["Player", "player_name"])
    league_src = col(df, ["League", "league_slug"])
    hp = col(hist, ["player_name", "Player"])
    hl = col(hist, ["league_slug", "League"])
    if not player_src or not hp:
        return df

    left = df.copy()
    right = hist.copy()
    left["__player_key"] = left[player_src].map(norm)
    right["__player_key"] = right[hp].map(norm)
    merge_keys = ["__player_key"]
    if league_src and hl:
        left["__league_key"] = left[league_src].map(norm)
        right["__league_key"] = right[hl].map(norm)
        merge_keys.append("__league_key")

    wanted = {
        "xA/90": ["xa_per90", "xA/90"],
        "Key Passes/90": ["key_passes_per90", "Key Passes/90"],
        "xG/90": ["xg_per90", "xG/90"],
        "Shots/90": ["shots_per90", "Shots/90"],
        "Box Touches/90": ["box_touches_per90", "touches_in_box_per90", "Box Touches/90"],
        "Progressive Carries/90": ["progressive_carries_per90"],
        "Successful Dribbles/90": ["successful_dribbles_per90"],
        "Recoveries/90": ["recoveries_per90", "Recoveries/90"],
        "Final Third Regains/90": ["possession_won_final_third_per90"],
        "Tackles Won/90": ["tackles_won_per90"],
        "Interceptions/90": ["interceptions_per90"],
    }
    rhs_cols = merge_keys.copy()
    rename = {}
    for out_name, aliases in wanted.items():
        c = col(right, aliases)
        if c:
            rhs_cols.append(c)
            rename[c] = f"__raw_{out_name}"
    right = right[rhs_cols].drop_duplicates(merge_keys).rename(columns=rename)
    merged = left.merge(right, on=merge_keys, how="left")

    for out_name in wanted:
        rc = f"__raw_{out_name}"
        if rc not in merged.columns:
            continue
        existing = col(merged, [out_name])
        if existing is None:
            merged[out_name] = merged[rc]
        else:
            cur = pd.to_numeric(merged[existing], errors="coerce")
            raw = pd.to_numeric(merged[rc], errors="coerce")
            # Repair clearly broken all-zero export columns.
            if cur.notna().sum() and np.isclose(cur.fillna(0).abs().sum(), 0) and raw.notna().sum():
                merged[existing] = raw
            else:
                merged[existing] = cur.where(cur.notna(), raw)
        merged = merged.drop(columns=[rc])
    merged = merged.drop(columns=[c for c in ["__player_key", "__league_key"] if c in merged.columns])
    return merged


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Run the existing winger analyzer first; missing {SOURCE}")
    df = pd.read_csv(SOURCE)
    df = repair_raw_metrics(df)

    creator_col = col(df, ["Creator", "wide_creator_score", "Wide Creator", "creator_score"])
    vertical_col = col(df, ["Vertical/Box", "vertical_box_score", "Vertical Box", "vertical_score"])
    hybrid_col = col(df, ["Hybrid", "hybrid_score", "Hybrid Winger"])
    best_col = col(df, ["Best Score", "best_archetype_score", "overall_winger_score"])

    # Reuse existing archetype scores when present. If an older/newer export
    # is missing them, reconstruct the archetypes from the repaired raw metrics
    # instead of failing immediately.
    if creator_col is None:
        df["wide_creator_score"] = composite(df, [
            (["xA/90", "xa_per90"], .35),
            (["Key Passes/90", "key_passes_per90"], .20),
            (["Progressive Carries/90", "progressive_carries_per90"], .25),
            (["Successful Dribbles/90", "successful_dribbles_per90"], .20),
        ])
    else:
        df["wide_creator_score"] = pd.to_numeric(df[creator_col], errors="coerce")

    if vertical_col is None:
        df["vertical_box_score"] = composite(df, [
            (["xG/90", "xg_per90"], .35),
            (["Shots/90", "shots_per90"], .25),
            (["Box Touches/90", "box_touches_per90", "touches_in_box_per90"], .20),
            (["Successful Dribbles/90", "successful_dribbles_per90"], .20),
        ])
    else:
        df["vertical_box_score"] = pd.to_numeric(df[vertical_col], errors="coerce")

    if hybrid_col is None:
        df["hybrid_score"] = (
            0.50 * df["wide_creator_score"]
            + 0.50 * df["vertical_box_score"]
        )
    else:
        df["hybrid_score"] = pd.to_numeric(df[hybrid_col], errors="coerce")

    if (
        df["wide_creator_score"].notna().sum() == 0
        and df["vertical_box_score"].notna().sum() == 0
        and df["hybrid_score"].notna().sum() == 0
    ):
        raise RuntimeError(
            "Could not recover winger archetype scores. "
            "Run the BASE analyze_winger.py first so winger_jaissle_archetypes.csv "
            "contains player rows and winger metrics."
        )
    if best_col:
        df["best_archetype_score"] = pd.to_numeric(df[best_col], errors="coerce")
    else:
        df["best_archetype_score"] = df[["wide_creator_score", "vertical_box_score", "hybrid_score"]].max(axis=1)

    df["creation_composite"] = composite(df, [
        (["xA/90"], .35), (["Key Passes/90"], .20), (["Progressive Carries/90"], .25), (["Successful Dribbles/90"], .20),
    ])
    df["direct_box_composite"] = composite(df, [
        (["xG/90"], .35), (["Shots/90"], .25), (["Box Touches/90"], .20), (["Successful Dribbles/90"], .20),
    ])
    df["off_ball_score"] = composite(df, [
        (["Recoveries/90"], .45), (["Final Third Regains/90"], .25), (["Tackles Won/90"], .15), (["Interceptions/90"], .15),
    ])
    df["off_ball_score"] = df["off_ball_score"].fillna(50.0)

    arch = df[["wide_creator_score", "vertical_box_score", "hybrid_score"]]
    primary = arch.max(axis=1)
    secondary = arch.apply(lambda r: r.nlargest(2).iloc[-1] if r.notna().sum() >= 2 else np.nan, axis=1)
    df["overall_winger_score"] = (
        .40 * primary + .25 * df["hybrid_score"] + .20 * df["off_ball_score"] + .15 * secondary
    )
    df["best_archetype"] = arch.idxmax(axis=1).map({
        "wide_creator_score": "Wide Creator",
        "vertical_box_score": "Vertical / Box Winger",
        "hybrid_score": "Hybrid Winger",
    })

    df.to_csv(OUT, index=False)
    df.to_html(OUT.with_suffix(".html"), index=False)
    print("=" * 120)
    print("EXTENDED WINGER ANALYSIS")
    print("=" * 120)
    show = [c for c in ["Player", "Club", "Age", "best_archetype", "wide_creator_score", "vertical_box_score", "hybrid_score", "off_ball_score", "overall_winger_score", "Value €M"] if c in df.columns]
    print(df.sort_values("overall_winger_score", ascending=False)[show].head(40).to_string(index=False))
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
