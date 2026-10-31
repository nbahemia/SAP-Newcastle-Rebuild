from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

FULL_SHORTLIST = (
    ROOT / "data" / "processed" / "bsd" / "shortlist"
    / "europe_2025_26_full_role_shortlist.parquet"
)

HISTORICAL_METRICS = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_season_metrics.parquet"
)

OUT_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"
IMAGE_DIR = OUT_DIR / "images"

OUT_CSV = OUT_DIR / "creative_10_jaissle_detailed.csv"
OUT_HTML = OUT_DIR / "creative_10_jaissle_detailed.html"
OUT_CHART = IMAGE_DIR / "creative_10_jaissle_detailed_chart.png"


ALIASES = {
    "xa": ["xa_per90", "xa"],
    "key_passes": ["key_passes_per90", "key_passes"],
    "successful_passes": ["successful_passes_per90", "successful_passes"],
    "total_passes": ["total_passes_per90", "total_passes"],
    "touches": ["touches_per90", "touches"],
    "xg": ["xg_per90", "xg"],
    "shots": ["shots_per90", "shots", "total_shots_per90", "total_shots"],
    "shots_on_target": ["shots_on_target_per90", "shots_on_target"],
    "recoveries": ["recoveries_per90", "recoveries"],
    "interceptions": ["interceptions_per90", "interceptions"],
    "tackles_won": ["tackles_won_per90", "tackles_won"],
    "duels_won": ["duels_won_per90", "duels_won"],
}


def norm(value) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def find_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c
        for c in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
        if key in normalized:
            return normalized[key]
    return None


def p90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    minutes = pd.to_numeric(minutes, errors="coerce")
    return pd.Series(
        np.where(minutes > 0, values * 90 / minutes, np.nan),
        index=values.index,
        dtype="float64",
    )


def winsor_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index, dtype=float)
    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)
    clipped = s.clip(lo, hi)
    sd = clipped.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (clipped - clipped.mean()) / sd


def index_from_weights(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weighted = pd.Series(0.0, index=df.index)
    observed = pd.Series(0.0, index=df.index)

    for col, weight in weights.items():
        if col not in df.columns:
            continue
        z = winsor_z(df[col])
        valid = z.notna()
        weighted.loc[valid] += z.loc[valid] * weight
        observed.loc[valid] += weight

    score = np.where(observed > 0, weighted / observed, np.nan)
    return pd.Series(50 + 10 * score, index=df.index).clip(20, 80)


def player_key(df: pd.DataFrame) -> pd.Series:
    return df["player_name"].astype(str).map(norm) + "|" + df["league_slug"].astype(str)


def natural_cam_cm(row) -> bool:
    text = " ".join(
        norm(row.get(c, ""))
        for c in ["natural_position", "tm_sub_position", "tm_position"]
    )
    return any(term in text for term in [
        "attacking midfield",
        "central midfield",
        "centre midfield",
    ])


NEWCASTLE_BENCHMARK_ALIASES = {'Willock': ['j willock', 'joe willock'], 'Ramsey': ['j ramsey', 'jacob ramsey'], 'Bamba': ['b bamba', 'bamba']}

def benchmark_label(name: str) -> str | None:
    n = norm(name)
    for label, aliases in NEWCASTLE_BENCHMARK_ALIASES.items():
        for alias in aliases:
            if n == norm(alias):
                return label
    return None

def mark_newcastle_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["benchmark_label"] = out["player_name"].astype(str).map(benchmark_label)
    out["is_newcastle_benchmark"] = out["benchmark_label"].notna()
    return out

def benchmark_first(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return df.sort_values(
        ["is_newcastle_benchmark", score_col],
        ascending=[False, False],
    )

def load_pool() -> pd.DataFrame:
    shortlist = pd.read_parquet(FULL_SHORTLIST).copy()
    hist = pd.read_parquet(HISTORICAL_METRICS).copy()

    mask = pd.Series(False, index=shortlist.index)
    if "role_key" in shortlist.columns:
        mask |= shortlist["role_key"].astype(str).eq("creative_cam")
    if "role" in shortlist.columns:
        mask |= shortlist["role"].astype(str).str.contains("Creative 10", case=False, na=False)

    cam = shortlist[mask].copy()
    cam = cam[cam.apply(natural_cam_cm, axis=1)].copy()

    cam["_player_key"] = player_key(cam)
    hist["_player_key"] = player_key(hist)

    source = hist[["_player_key", "minutes"]].copy()
    source = source.rename(columns={"minutes": "_src_minutes"})

    detected = {}
    for metric, aliases in ALIASES.items():
        col = find_alias(hist, aliases)
        detected[metric] = col
        if col:
            source[f"_src_{metric}"] = pd.to_numeric(hist[col], errors="coerce")

    source = source.drop_duplicates("_player_key", keep="first")
    cam = cam.merge(source, on="_player_key", how="left").copy()

    print("=" * 110)
    print("CAM METRIC SOURCES")
    print("=" * 110)

    for metric, col in detected.items():
        print(f"{metric:<22} {col if col else 'MISSING'}")

    for metric, source_col in detected.items():
        out = f"{metric}_p90"
        if source_col is None:
            cam[out] = np.nan
            continue
        raw = pd.to_numeric(cam[f"_src_{metric}"], errors="coerce")
        if "per90" in source_col.lower():
            cam[out] = raw
        else:
            cam[out] = p90(raw, cam["_src_minutes"])

    # Creation/link play: the core Jaissle 10 requirement.
    cam["creation_link_index"] = index_from_weights(
        cam,
        {
            "xa_p90": 0.30,
            "key_passes_p90": 0.30,
            "successful_passes_p90": 0.18,
            "total_passes_p90": 0.12,
            "touches_p90": 0.10,
        },
    )

    # Goal / box threat so the 10 can arrive beyond the striker.
    cam["attacking_threat_index"] = index_from_weights(
        cam,
        {
            "xg_p90": 0.45,
            "shots_p90": 0.30,
            "shots_on_target_p90": 0.25,
        },
    )

    # Jaissle pressing / counterpressing activity.
    cam["pressing_activity_index"] = index_from_weights(
        cam,
        {
            "recoveries_p90": 0.35,
            "tackles_won_p90": 0.25,
            "interceptions_p90": 0.20,
            "duels_won_p90": 0.20,
        },
    )

    cam["role_fit"] = pd.to_numeric(cam["role_fit"], errors="coerce")

    # Overall Newcastle / Jaissle CAM score.
    # Creation is deliberately dominant because Newcastle's diagnosed gap is central creation.
    cam["jaissle_cam_score"] = (
        0.40 * cam["role_fit"]
        + 0.35 * cam["creation_link_index"]
        + 0.10 * cam["attacking_threat_index"]
        + 0.15 * cam["pressing_activity_index"]
    )

    cam = mark_newcastle_benchmarks(cam)



    cam = cam.sort_values(
        ["jaissle_cam_score", "creation_link_index", "role_fit"],
        ascending=False,
    )

    return cam


def top_with_benchmarks(df: pd.DataFrame, score_col: str, n: int = 30) -> pd.DataFrame:
    top = df.sort_values(score_col, ascending=False).head(n)
    benches = df[df.get("is_newcastle_benchmark", False)].copy()
    combined = pd.concat([top, benches], ignore_index=False)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_values(score_col, ascending=False)


def build_chart(df: pd.DataFrame, top_n: int = 30):
    chart = top_with_benchmarks(
        df.dropna(subset=["role_fit", "creation_link_index", "jaissle_cam_score"]),
        "jaissle_cam_score",
        top_n,
    ).copy()

    if chart.empty:
        return

    physical = chart["pressing_activity_index"].fillna(50)
    size = np.clip(physical - 25, 5, None) * 13

    fig, ax = plt.subplots(figsize=(16, 10))

    sc = ax.scatter(
        chart["role_fit"],
        chart["creation_link_index"],
        s=size,
        c=chart["attacking_threat_index"].fillna(50),
        cmap="viridis",
        alpha=0.78,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Attacking threat index")

    ax.axvline(50, linestyle="--", linewidth=1)
    ax.axhline(50, linestyle="--", linewidth=1)

    for _, row in chart.iterrows():
        ax.annotate(
            f"{row['player_name']} ({row['jaissle_cam_score']:.1f})",
            (row["role_fit"], row["creation_link_index"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Creative 10 Role Fit")
    ax.set_ylabel("Creation / Link Index")
    ax.set_title(
        "Jaissle Creative 10 Recruitment\n"
        "Bubble size = pressing activity | Color = attacking threat"
    )

    fig.text(
        0.01, 0.035,
        "Jaissle CAM Score = 40% Creative-10 Fit + 35% Creation/Link + "
        "10% Attacking Threat + 15% Pressing Activity.",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CHART, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_html(df: pd.DataFrame):
    cols = [
        "player_name", "club_name", "league_slug", "age", "benchmark_label", "is_newcastle_benchmark",
        "jaissle_cam_score", "role_fit", "creation_link_index",
        "attacking_threat_index", "pressing_activity_index",
        "xa_p90", "key_passes_p90", "xg_p90", "shots_p90",
        "recoveries_p90", "duels_won_p90", "market_value_eur_m",
    ]
    cols = [c for c in cols if c in df.columns]

    html = top_with_benchmarks(df, "jaissle_cam_score", 80)[cols].to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}",
        classes="scout-table",
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jaissle Creative 10 Recruitment</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; background:#f4f6f7; color:#17202a; }}
.card {{ background:white; padding:16px; margin-bottom:18px; border-left:4px solid #34495e; }}
table {{ border-collapse:collapse; width:100%; background:white; font-size:12px; }}
th {{ background:#17202a; color:white; padding:8px; }}
td {{ padding:7px; border-bottom:1px solid #ddd; }}
</style>
</head>
<body>
<h1>Jaissle Creative 10 Recruitment</h1>
<div class="card">
<strong>Role context:</strong> operate between the lines, connect midfield to the front line,
create chances centrally, arrive as a secondary scoring threat, and participate in a high counterpress.<br><br>
<strong>Jaissle CAM Score:</strong> 40% Creative-10 Fit, 35% Creation/Link,
10% Attacking Threat, 15% Pressing Activity.
</div>
{html}
</body>
</html>"""
    OUT_HTML.write_text(doc, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    result = load_pool()

    export_cols = [
        "player_name", "club_name", "league_slug", "age", "minutes",
        "jaissle_cam_score", "role_fit", "creation_link_index",
        "attacking_threat_index", "pressing_activity_index",
        "xa_p90", "key_passes_p90", "successful_passes_p90",
        "total_passes_p90", "touches_p90", "xg_p90", "shots_p90",
        "shots_on_target_p90", "recoveries_p90", "interceptions_p90",
        "tackles_won_p90", "duels_won_p90", "market_value_eur_m",
        "natural_position", "tm_sub_position",
    ]
    export_cols = [c for c in export_cols if c in result.columns]
    result[export_cols].to_csv(OUT_CSV, index=False)

    save_html(result)
    build_chart(result)

    print()
    print("=" * 130)
    print("TOP JAISSLE CREATIVE 10 TARGETS")
    print("=" * 130)

    display_cols = [
        "player_name", "club_name", "age", "benchmark_label", "jaissle_cam_score",
        "role_fit", "creation_link_index", "attacking_threat_index",
        "pressing_activity_index", "xa_p90", "key_passes_p90",
        "xg_p90", "recoveries_p90", "duels_won_p90",
        "market_value_eur_m",
    ]
    display_cols = [c for c in display_cols if c in result.columns]

    display = top_with_benchmarks(result, "jaissle_cam_score", 30)[display_cols].copy()
    display = display.rename(columns={
        "player_name": "Player",
        "club_name": "Club",
        "age": "Age",
        "benchmark_label": "NUFC Benchmark",
        "jaissle_cam_score": "Jaissle CAM",
        "role_fit": "CAM Fit",
        "creation_link_index": "Creation/Link",
        "attacking_threat_index": "Attack Threat",
        "pressing_activity_index": "Pressing",
        "xa_p90": "xA/90",
        "key_passes_p90": "Key Passes/90",
        "xg_p90": "xG/90",
        "recoveries_p90": "Recoveries/90",
        "duels_won_p90": "Duels Won/90",
        "market_value_eur_m": "Value €M",
    })

    for col in display.columns:
        if col not in ["Player", "Club", "NUFC Benchmark"]:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(2)

    print(display.to_string(index=False))

    print()
    print("Saved:")
    print(" ", OUT_CSV)
    print(" ", OUT_HTML)
    print(" ", OUT_CHART)


if __name__ == "__main__":
    main()
