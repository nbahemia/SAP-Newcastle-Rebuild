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

OUT_CSV = OUT_DIR / "workhorse_cm_jaissle_detailed.csv"
OUT_HTML = OUT_DIR / "workhorse_cm_jaissle_detailed.html"
OUT_CHART = IMAGE_DIR / "workhorse_cm_jaissle_detailed_chart.png"


ALIASES = {
    "recoveries": ["recoveries_per90", "recoveries"],
    "interceptions": ["interceptions_per90", "interceptions"],
    "tackles_won": ["tackles_won_per90", "tackles_won"],
    "duels_won": ["duels_won_per90", "duels_won"],
    "aerials_won": ["aerials_won_per90", "aerials_won"],
    "successful_passes": ["successful_passes_per90", "successful_passes"],
    "total_passes": ["total_passes_per90", "total_passes"],
    "pass_accuracy": ["pass_accuracy_pct"],
    "touches": ["touches_per90", "touches"],
    "xa": ["xa_per90", "xa"],
    "key_passes": ["key_passes_per90", "key_passes"],
    "xg": ["xg_per90", "xg"],
    "shots": ["shots_per90", "shots", "total_shots_per90", "total_shots"],
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

    return pd.Series(
        50 + 10 * score,
        index=df.index,
    ).clip(20, 80)


def player_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["player_name"].astype(str).map(norm)
        + "|"
        + df["league_slug"].astype(str)
    )


def is_cm(row) -> bool:
    text = " ".join(
        norm(row.get(c, ""))
        for c in ["natural_position", "tm_sub_position", "tm_position"]
    )

    return any(term in text for term in [
        "central midfield",
        "centre midfield",
        "defensive midfield",
        "attacking midfield",
    ])


def experience_band(age):
    age = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]

    if pd.isna(age):
        return "Unknown"
    if age <= 21:
        return "Development"
    if age <= 25:
        return "Prime / growth"
    if age <= 29:
        return "Prime / experienced"
    return "Veteran"


NEWCASTLE_BENCHMARK_ALIASES = {'Ramsey': ['j ramsey', 'jacob ramsey'], 'Joelinton': ['joelinton'], 'Bamba': ['b bamba', 'bamba'], 'Willock': ['j willock', 'joe willock'], 'Miley': ['l miley', 'lewis miley'], 'Nico': ['n gonzalez', 'nico gonzalez', 'nicolás gonzález', 'nicolas gonzalez']}

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
        mask |= shortlist["role_key"].astype(str).eq("advanced_pivot")

    if "role" in shortlist.columns:
        mask |= shortlist["role"].astype(str).str.contains(
            "Advanced|Workhorse",
            case=False,
            na=False,
            regex=True,
        )

    cm = shortlist[mask].copy()
    cm = cm[cm.apply(is_cm, axis=1)].copy()

    cm["_player_key"] = player_key(cm)
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
    cm = cm.merge(source, on="_player_key", how="left").copy()

    print("=" * 110)
    print("WORKHORSE CM METRIC SOURCES")
    print("=" * 110)

    for metric, col in detected.items():
        print(f"{metric:<22} {col if col else 'MISSING'}")

    for metric, source_col in detected.items():
        out = f"{metric}_p90"

        if source_col is None:
            cm[out] = np.nan
            continue

        raw = pd.to_numeric(cm[f"_src_{metric}"], errors="coerce")

        if "per90" in source_col.lower() or source_col == "pass_accuracy_pct":
            cm[out] = raw
        else:
            cm[out] = p90(raw, cm["_src_minutes"])

    # Core workhorse requirement: repeat defensive actions and compete physically.
    cm["workhorse_activity_index"] = score_index(
        cm,
        {
            "recoveries_p90": 0.30,
            "tackles_won_p90": 0.25,
            "interceptions_p90": 0.20,
            "duels_won_p90": 0.20,
            "aerials_won_p90": 0.05,
        },
    )

    # Keep the ball moving after regains and support buildup.
    cm["ball_security_link_index"] = score_index(
        cm,
        {
            "pass_accuracy_p90": 0.30,
            "successful_passes_p90": 0.25,
            "total_passes_p90": 0.20,
            "touches_p90": 0.15,
            "xa_p90": 0.10,
        },
    )

    # Advanced pivot should still support attacks / late arrivals.
    cm["attacking_support_index"] = score_index(
        cm,
        {
            "xa_p90": 0.30,
            "key_passes_p90": 0.25,
            "xg_p90": 0.25,
            "shots_p90": 0.20,
        },
    )

    cm["role_fit"] = pd.to_numeric(cm["role_fit"], errors="coerce")

    # Newcastle already has control through Nico; this emphasizes activity and role execution.
    cm["jaissle_workhorse_score"] = (
        0.40 * cm["role_fit"]
        + 0.30 * cm["workhorse_activity_index"]
        + 0.20 * cm["ball_security_link_index"]
        + 0.10 * cm["attacking_support_index"]
    )

    cm["experience_band"] = cm["age"].apply(experience_band)

    cm = mark_newcastle_benchmarks(cm)



    cm = cm.sort_values(
        [
            "jaissle_workhorse_score",
            "workhorse_activity_index",
            "role_fit",
        ],
        ascending=False,
    )

    return cm


def top_with_benchmarks(df: pd.DataFrame, score_col: str, n: int = 30) -> pd.DataFrame:
    top = df.sort_values(score_col, ascending=False).head(n)
    benches = df[df.get("is_newcastle_benchmark", False)].copy()
    combined = pd.concat([top, benches], ignore_index=False)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_values(score_col, ascending=False)


def build_chart(df: pd.DataFrame, top_n: int = 30):
    chart = top_with_benchmarks(
        df.dropna(subset=["role_fit", "workhorse_activity_index", "jaissle_workhorse_score"]),
        "jaissle_workhorse_score",
        top_n,
    ).copy()

    if chart.empty:
        return

    bubble = (
        np.clip(chart["ball_security_link_index"].fillna(50) - 25, 5, None)
        * 13
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    sc = ax.scatter(
        chart["role_fit"],
        chart["workhorse_activity_index"],
        s=bubble,
        c=chart["attacking_support_index"].fillna(50),
        cmap="viridis",
        alpha=0.78,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Attacking support index")

    ax.axvline(50, linestyle="--", linewidth=1)
    ax.axhline(50, linestyle="--", linewidth=1)

    for _, row in chart.iterrows():
        ax.annotate(
            f"{row['player_name']} ({row['jaissle_workhorse_score']:.1f})",
            (
                row["role_fit"],
                row["workhorse_activity_index"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Advanced / Workhorse Pivot Role Fit")
    ax.set_ylabel("Workhorse Activity Index")
    ax.set_title(
        "Jaissle Workhorse Central Midfielder Recruitment\n"
        "Bubble size = ball security/link play | Color = attacking support"
    )

    fig.text(
        0.01,
        0.035,
        "Jaissle Workhorse Score = 40% role fit + 30% defensive/workhorse activity + "
        "20% ball security/link play + 10% attacking support. "
        "Leadership and mentoring are shown as contextual considerations, not statistical scores.",
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
        "jaissle_workhorse_score",
        "role_fit",
        "workhorse_activity_index",
        "ball_security_link_index",
        "attacking_support_index",
        "recoveries_p90",
        "tackles_won_p90",
        "interceptions_p90",
        "duels_won_p90",
        "pass_accuracy_p90",
        "successful_passes_p90",
        "xa_p90",
        "key_passes_p90",
        "xg_p90",
        "market_value_eur_m",
    ]
    cols = [c for c in cols if c in df.columns]

    table = top_with_benchmarks(df, "jaissle_workhorse_score", 80)[cols].to_html(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jaissle Workhorse CM Recruitment</title>
<style>
body {{ font-family: Arial, sans-serif; margin:28px; background:#f4f6f7; color:#17202a; }}
.card {{ background:white; padding:16px; margin-bottom:18px; border-left:4px solid #34495e; }}
table {{ border-collapse:collapse; width:100%; background:white; font-size:12px; }}
th {{ background:#17202a; color:white; padding:8px; }}
td {{ padding:7px; border-bottom:1px solid #ddd; }}
</style>
</head>
<body>
<h1>Jaissle Workhorse Central Midfielder Recruitment</h1>
<div class="card">
<strong>Role context:</strong> press aggressively, recover second balls, cover transitions,
carry the team's running load, keep possession moving after regains, and support attacks with
late runs / chance creation.<br><br>
<strong>Jaissle Workhorse Score:</strong> 40% role fit, 30% workhorse activity,
20% ball security/link play, 10% attacking support.<br><br>
<strong>Important squad-building context:</strong> age and experience are displayed, but not scored.
Leadership, communication, mentoring, adaptability and dressing-room influence must be assessed
qualitatively before replacing an experienced player.
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
        "jaissle_workhorse_score",
        "role_fit",
        "workhorse_activity_index",
        "ball_security_link_index",
        "attacking_support_index",
        "recoveries_p90",
        "tackles_won_p90",
        "interceptions_p90",
        "duels_won_p90",
        "aerials_won_p90",
        "pass_accuracy_p90",
        "successful_passes_p90",
        "total_passes_p90",
        "touches_p90",
        "xa_p90",
        "key_passes_p90",
        "xg_p90",
        "shots_p90",
        "market_value_eur_m",
        "natural_position",
        "tm_sub_position",
    ]

    export_cols = [c for c in export_cols if c in result.columns]
    result[export_cols].to_csv(OUT_CSV, index=False)

    save_html(result)
    build_chart(result)

    print()
    print("=" * 150)
    print("TOP JAISSLE WORKHORSE CENTRAL-MIDFIELD TARGETS")
    print("=" * 150)

    display_cols = [
        "player_name",
        "club_name",
        "age",
        "benchmark_label",
        "experience_band",
        "jaissle_workhorse_score",
        "role_fit",
        "workhorse_activity_index",
        "ball_security_link_index",
        "attacking_support_index",
        "recoveries_p90",
        "tackles_won_p90",
        "interceptions_p90",
        "duels_won_p90",
        "pass_accuracy_p90",
        "xa_p90",
        "market_value_eur_m",
    ]

    display_cols = [c for c in display_cols if c in result.columns]
    display = top_with_benchmarks(result, "jaissle_workhorse_score", 30)[display_cols].copy()

    display = display.rename(
        columns={
            "player_name": "Player",
            "club_name": "Club",
            "age": "Age",
            "benchmark_label": "NUFC Benchmark",
            "experience_band": "Experience",
            "jaissle_workhorse_score": "Jaissle CM",
            "role_fit": "CM Fit",
            "workhorse_activity_index": "Workhorse",
            "ball_security_link_index": "Security/Link",
            "attacking_support_index": "Attack Support",
            "recoveries_p90": "Recoveries/90",
            "tackles_won_p90": "Tackles Won/90",
            "interceptions_p90": "Interceptions/90",
            "duels_won_p90": "Duels Won/90",
            "pass_accuracy_p90": "Pass %",
            "xa_p90": "xA/90",
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
