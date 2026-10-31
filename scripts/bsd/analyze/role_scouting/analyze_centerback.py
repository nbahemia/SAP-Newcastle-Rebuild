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

OUT_CSV = OUT_DIR / "center_back_jaissle_detailed.csv"
OUT_HTML = OUT_DIR / "center_back_jaissle_detailed.html"
OUT_CHART = IMAGE_DIR / "center_back_jaissle_detailed_chart.png"


ALIASES = {
    "successful_passes": ["successful_passes_per90", "successful_passes"],
    "total_passes": ["total_passes_per90", "total_passes"],
    "pass_accuracy": ["pass_accuracy_pct"],
    "accurate_long_balls": ["accurate_long_balls_per90", "accurate_long_balls"],
    "total_long_balls": ["total_long_balls_per90", "total_long_balls"],
    "recoveries": ["recoveries_per90", "recoveries"],
    "interceptions": ["interceptions_per90", "interceptions"],
    "tackles_won": ["tackles_won_per90", "tackles_won"],
    "duels_won": ["duels_won_per90", "duels_won"],
    "total_duels": ["total_duels_per90", "total_duels"],
    "aerials_won": ["aerials_won_per90", "aerials_won"],
    "total_aerials": ["total_aerials_per90", "total_aerials"],
    "touches": ["touches_per90", "touches"],
}


def norm(value) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def find_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {
        re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c
        for c in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
        if key in lookup:
            return lookup[key]
    return None


def p90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    minutes = pd.to_numeric(minutes, errors="coerce")
    return pd.Series(
        np.where(minutes > 0, values * 90.0 / minutes, np.nan),
        index=values.index,
        dtype=float,
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


def score_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
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
    return (
        df["player_name"].astype(str).map(norm)
        + "|"
        + df["league_slug"].astype(str)
    )


def is_cb(row) -> bool:
    text = " ".join(
        norm(row.get(c, ""))
        for c in ["natural_position", "tm_sub_position", "tm_position"]
    )
    return any(term in text for term in [
        "centre back",
        "center back",
        "central defender",
    ])


def experience_band(age):
    age = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(age):
        return "Unknown"
    if age <= 22:
        return "Development"
    if age <= 26:
        return "Prime / growth"
    if age <= 30:
        return "Prime / experienced"
    return "Veteran"


NEWCASTLE_BENCHMARK_ALIASES = {'Botman': ['s botman', 'sven botman'], 'Thiaw': ['m thiaw', 'malick thiaw'], 'Schar': ['f schar', 'fabian schar', 'f schär', 'fabian schär'], 'Burn': ['d burn', 'dan burn']}

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
        mask |= shortlist["role_key"].astype(str).eq("high_line_cb")

    if "role" in shortlist.columns:
        mask |= shortlist["role"].astype(str).str.contains(
            "High-Line|Centre-Back|Center-Back|CB",
            case=False,
            na=False,
            regex=True,
        )

    cb = shortlist[mask].copy()
    cb = cb[cb.apply(is_cb, axis=1)].copy()

    cb["_player_key"] = player_key(cb)
    hist["_player_key"] = player_key(hist)

    source = hist[["_player_key", "minutes"]].copy()
    source = source.rename(columns={"minutes": "_src_minutes"})

    detected = {}

    for metric, aliases in ALIASES.items():
        col = find_alias(hist, aliases)
        detected[metric] = col

        if col is not None:
            source[f"_src_{metric}"] = pd.to_numeric(
                hist[col],
                errors="coerce",
            )

    source = source.drop_duplicates("_player_key", keep="first")
    cb = cb.merge(source, on="_player_key", how="left").copy()

    print("=" * 110)
    print("CENTER-BACK METRIC SOURCES")
    print("=" * 110)

    for metric, col in detected.items():
        print(f"{metric:<24} {col if col else 'MISSING'}")

    for metric, source_col in detected.items():
        out = f"{metric}_p90"

        if source_col is None:
            cb[out] = np.nan
            continue

        raw = pd.to_numeric(cb[f"_src_{metric}"], errors="coerce")

        if "per90" in source_col.lower() or source_col == "pass_accuracy_pct":
            cb[out] = raw
        else:
            cb[out] = p90(raw, cb["_src_minutes"])

    # Build-up / possession security.
    cb["build_up_index"] = score_index(
        cb,
        {
            "pass_accuracy_p90": 0.30,
            "successful_passes_p90": 0.25,
            "total_passes_p90": 0.20,
            "accurate_long_balls_p90": 0.15,
            "touches_p90": 0.10,
        },
    )

    # Front-foot high-line defending. This is NOT a pace score.
    cb["front_foot_defending_index"] = score_index(
        cb,
        {
            "recoveries_p90": 0.30,
            "interceptions_p90": 0.30,
            "tackles_won_p90": 0.20,
            "duels_won_p90": 0.20,
        },
    )

    # Physical security against direct play / isolated duels.
    cb["physical_security_index"] = score_index(
        cb,
        {
            "aerials_won_p90": 0.40,
            "duels_won_p90": 0.35,
            "total_aerials_p90": 0.15,
            "total_duels_p90": 0.10,
        },
    )

    cb["role_fit"] = pd.to_numeric(cb["role_fit"], errors="coerce")

    cb["jaissle_cb_score"] = (
        0.40 * cb["role_fit"]
        + 0.25 * cb["build_up_index"]
        + 0.20 * cb["front_foot_defending_index"]
        + 0.15 * cb["physical_security_index"]
    )

    cb["experience_band"] = cb["age"].apply(experience_band)

    cb = mark_newcastle_benchmarks(cb)



    cb = cb.sort_values(
        [
            "jaissle_cb_score",
            "role_fit",
            "front_foot_defending_index",
        ],
        ascending=False,
    )

    return cb


def top_with_benchmarks(df: pd.DataFrame, score_col: str, n: int = 30) -> pd.DataFrame:
    top = df.sort_values(score_col, ascending=False).head(n)
    benches = df[df.get("is_newcastle_benchmark", False)].copy()
    combined = pd.concat([top, benches], ignore_index=False)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_values(score_col, ascending=False)


def build_chart(df: pd.DataFrame, top_n: int = 30):
    chart = top_with_benchmarks(
        df.dropna(subset=["role_fit", "build_up_index", "jaissle_cb_score"]),
        "jaissle_cb_score",
        top_n,
    ).copy()

    if chart.empty:
        return

    bubble = (
        np.clip(chart["physical_security_index"].fillna(50) - 25, 5, None)
        * 13
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    sc = ax.scatter(
        chart["role_fit"],
        chart["build_up_index"],
        s=bubble,
        c=chart["front_foot_defending_index"].fillna(50),
        cmap="viridis",
        alpha=0.78,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Front-foot defending index")

    ax.axvline(50, linestyle="--", linewidth=1)
    ax.axhline(50, linestyle="--", linewidth=1)

    for _, row in chart.iterrows():
        ax.annotate(
            f"{row['player_name']} ({row['jaissle_cb_score']:.1f})",
            (row["role_fit"], row["build_up_index"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("High-Line CB Role Fit")
    ax.set_ylabel("Build-Up Index")
    ax.set_title(
        "Jaissle High-Line Centre-Back Recruitment\n"
        "Bubble size = physical security | Color = front-foot defending"
    )

    fig.text(
        0.01,
        0.035,
        "Jaissle CB Score = 40% role fit + 25% build-up + "
        "20% front-foot defending + 15% physical security. "
        "No pace metric is available, so high-line recovery speed must be validated separately.",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CHART, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_html(df: pd.DataFrame):
    cols = [
        "player_name",
        "club_name",
        "league_slug",
        "age",
        "benchmark_label",
        "is_newcastle_benchmark",
        "experience_band",
        "jaissle_cb_score",
        "role_fit",
        "build_up_index",
        "front_foot_defending_index",
        "physical_security_index",
        "pass_accuracy_p90",
        "successful_passes_p90",
        "accurate_long_balls_p90",
        "recoveries_p90",
        "interceptions_p90",
        "tackles_won_p90",
        "duels_won_p90",
        "aerials_won_p90",
        "market_value_eur_m",
    ]
    cols = [c for c in cols if c in df.columns]

    table = top_with_benchmarks(df, "jaissle_cb_score", 80)[cols].to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jaissle High-Line CB Recruitment</title>
<style>
body {{ font-family: Arial, sans-serif; margin:28px; background:#f4f6f7; color:#17202a; }}
.card {{ background:white; padding:16px; margin-bottom:18px; border-left:4px solid #34495e; }}
table {{ border-collapse:collapse; width:100%; background:white; font-size:12px; }}
th {{ background:#17202a; color:white; padding:8px; }}
td {{ padding:7px; border-bottom:1px solid #ddd; }}
</style>
</head>
<body>
<h1>Jaissle High-Line Centre-Back Recruitment</h1>
<div class="card">
<strong>Role context:</strong> defend aggressively in a high line, recover loose balls,
win isolated duels, protect against direct play, and contribute cleanly to buildup.<br><br>
<strong>Jaissle CB Score:</strong> 40% high-line CB role fit, 25% buildup,
20% front-foot defending, 15% physical security.<br><br>
<strong>Important:</strong> age/experience is shown as squad-building context, not scored.
Leadership, communication, mentoring and recovery pace are not captured by this dataset.
</div>
{table}
</body>
</html>"""

    OUT_HTML.write_text(doc, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    result = load_pool()

    export_cols = [
        "player_name",
        "club_name",
        "league_slug",
        "age",
        "experience_band",
        "minutes",
        "jaissle_cb_score",
        "role_fit",
        "build_up_index",
        "front_foot_defending_index",
        "physical_security_index",
        "pass_accuracy_p90",
        "successful_passes_p90",
        "total_passes_p90",
        "accurate_long_balls_p90",
        "recoveries_p90",
        "interceptions_p90",
        "tackles_won_p90",
        "duels_won_p90",
        "aerials_won_p90",
        "market_value_eur_m",
        "natural_position",
        "tm_sub_position",
    ]

    export_cols = [c for c in export_cols if c in result.columns]
    result[export_cols].to_csv(OUT_CSV, index=False)

    save_html(result)
    build_chart(result)

    print()
    print("=" * 145)
    print("TOP JAISSLE HIGH-LINE CENTER-BACK TARGETS")
    print("=" * 145)

    display_cols = [
        "player_name",
        "club_name",
        "age",
        "benchmark_label",
        "experience_band",
        "jaissle_cb_score",
        "role_fit",
        "build_up_index",
        "front_foot_defending_index",
        "physical_security_index",
        "pass_accuracy_p90",
        "recoveries_p90",
        "interceptions_p90",
        "duels_won_p90",
        "aerials_won_p90",
        "market_value_eur_m",
    ]
    display_cols = [c for c in display_cols if c in result.columns]

    display = top_with_benchmarks(result, "jaissle_cb_score", 30)[display_cols].copy()

    display = display.rename(
        columns={
            "player_name": "Player",
            "club_name": "Club",
            "age": "Age",
            "benchmark_label": "NUFC Benchmark",
            "experience_band": "Experience",
            "jaissle_cb_score": "Jaissle CB",
            "role_fit": "CB Fit",
            "build_up_index": "Build-Up",
            "front_foot_defending_index": "Front-Foot",
            "physical_security_index": "Physical",
            "pass_accuracy_p90": "Pass %",
            "recoveries_p90": "Recoveries/90",
            "interceptions_p90": "Interceptions/90",
            "duels_won_p90": "Duels Won/90",
            "aerials_won_p90": "Aerials Won/90",
            "market_value_eur_m": "Value €M",
        }
    )

    for col in display.columns:
        if col not in ["Player", "Club", "Experience", "NUFC Benchmark"]:
            display[col] = pd.to_numeric(
                display[col],
                errors="coerce",
            ).round(2)

    print(display.to_string(index=False))

    print()
    print("Saved:")
    print(" ", OUT_CSV)
    print(" ", OUT_HTML)
    print(" ", OUT_CHART)


if __name__ == "__main__":
    main()
