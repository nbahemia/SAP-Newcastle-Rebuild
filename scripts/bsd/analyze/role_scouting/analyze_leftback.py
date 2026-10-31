from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================


# ======================================================================================
# OFFLINE NOTE
# This script uses only local parquet/CSV files already produced by the historical
# scouting pipeline. It does not call the BSD API and does not require BSD_API_KEY.
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]

FULL_SHORTLIST = (
    ROOT / "data" / "processed" / "bsd" / "shortlist"
    / "europe_2025_26_full_role_shortlist.parquet"
)

ENRICHED = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_percentiles_enriched.parquet"
)

HISTORICAL_METRICS = (
    ROOT / "data" / "processed" / "bsd"
    / "europe_2025_26_player_season_metrics.parquet"
)

OUT_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"
IMAGE_DIR = OUT_DIR / "images"

OUT_CSV = OUT_DIR / "left_wingback_recruitment_progressive.csv"
OUT_HTML = OUT_DIR / "left_wingback_recruitment_progressive.html"
OUT_CHART = IMAGE_DIR / "left_wingback_progressive_creativity_chart.png"

HALL_WB_FIT = 62.2


# ======================================================================================
# METRIC ALIASES
# ======================================================================================

ALIASES = {
    "xa": [
        "xa_per90",
        "xa",
    ],
    "key_passes": [
        "key_passes_per90",
        "key_passes",
    ],
    "accurate_crosses": [
        "accurate_crosses_per90",
        "accurate_crosses",
    ],
    "touches": [
        "touches_per90",
        "touches",
    ],
    "successful_passes": [
        "successful_passes_per90",
        "successful_passes",
    ],
    "duels_won": [
        "duels_won_per90",
        "duels_won",
    ],
    "aerials_won": [
        "aerials_won_per90",
        "aerials_won",
    ],
    "recoveries": [
        "recoveries_per90",
        "recoveries",
    ],
    "interceptions": [
        "interceptions_per90",
        "interceptions",
    ],
    "tackles_won": [
        "tackles_won_per90",
        "tackles_won",
    ],
}



# ======================================================================================
# HELPERS
# ======================================================================================

def norm(value) -> str:
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    ).strip()


def is_hall(value) -> bool:
    x = norm(value)
    return x in {"lewis hall", "l hall"} or x.endswith(" lewis hall")


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
        np.where(
            minutes > 0,
            values * 90.0 / minutes,
            np.nan,
        ),
        index=values.index,
        dtype="float64",
    )


def source_is_already_rate(source_name: str, values: pd.Series) -> bool:
    """
    The enriched historical file contains a mix of season totals and already-derived
    rate features. Detect likely rates so we do not divide by minutes twice.

    Explicit naming wins first. For derived aliases such as progressive_carries,
    key_passes and duels_won, use their scale as a secondary check.
    """
    if source_name is None:
        return False

    name = source_name.lower()

    if any(token in name for token in ["_p90", "per90", "per_90", "_90"]):
        return True

    known_rate_names = {
        "progressive_carries",
        "progressive_carry_distance",
        "key_passes",
        "duels_won",
        "aerials_won",
        "recoveries",
        "tackles_won",
        "accurate_crosses",
        "accurate_opposition_half_passes",
    }

    if name in known_rate_names:
        return True

    s = pd.to_numeric(values, errors="coerce").dropna()

    if s.empty:
        return False

    # Conservative fallback: metrics with modest distributions are probably already rates.
    # This does NOT apply to xA, which is intentionally treated as a season total unless named per90.
    if name != "xa":
        q95 = float(s.quantile(0.95))
        if q95 < 100:
            return True

    return False


def winsor_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if len(valid) < 5:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)

    clipped = s.clip(lo, hi)

    mean = clipped.mean(skipna=True)
    std = clipped.std(skipna=True, ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index, dtype="float64")

    return (clipped - mean) / std


def position_text(row) -> str:
    values = []

    for col in ["tm_sub_position", "tm_position", "natural_position"]:
        if col in row.index and pd.notna(row.get(col, np.nan)):
            values.append(norm(row[col]))

    return " | ".join(values)


def is_left_back(row) -> bool:
    pos = position_text(row)

    right_terms = [
        "right back",
        "right wing back",
        "right wingback",
        "right full back",
        "right fullback",
    ]

    if any(term in pos for term in right_terms):
        return False

    left_terms = [
        "left back",
        "left wing back",
        "left wingback",
        "left full back",
        "left fullback",
    ]

    return any(term in pos for term in left_terms)


def detect_wb_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)

    if "role_key" in df.columns:
        key = df["role_key"].astype(str).str.lower()
        mask |= key.str.contains(
            r"vertical_wingback|wingback|wing_back|fullback|full_back|fb_wb",
            regex=True,
            na=False,
        )

    if "role" in df.columns:
        role = df["role"].astype(str).str.lower()
        mask |= role.str.contains(
            r"wingback|wing back|fullback|full back|vertical full",
            regex=True,
            na=False,
        )

    return df[mask].copy()


def player_key(df: pd.DataFrame) -> pd.Series:
    """
    Use normalized full player name + league for every dataset.

    Do NOT switch to player_id merely because the column exists: the manually
    appended Hall row can introduce player_id into a shortlist where all other
    rows have NaN player_id values, which would break the merge for everyone else.
    """
    return (
        df["player_name"].astype(str).map(norm)
        + "|"
        + df["league_slug"].astype(str)
    )


# ======================================================================================
# LOAD FULLBACK RECRUITMENT POOL
# ======================================================================================

def build_pool() -> pd.DataFrame:
    shortlist = pd.read_parquet(FULL_SHORTLIST).copy()
    enriched = pd.read_parquet(ENRICHED).copy()
    historical = pd.read_parquet(HISTORICAL_METRICS).copy()

    wb = detect_wb_rows(shortlist)
    wb = wb[wb.apply(is_left_back, axis=1)].copy()

    # Hall is missing from generated role shortlist. Pull metadata from enriched
    # historical dataset and append him with the known historical WB fit.
    hall = enriched[
        enriched["player_name"].astype(str).map(is_hall)
    ].copy()

    if not hall.empty:
        hall = hall.sort_values(
            "minutes",
            ascending=False,
            na_position="last",
        ).head(1).copy()

        hall["role_key"] = "vertical_wingback"
        hall["role"] = "Vertical Full-Back / Wingback"
        hall["role_fit"] = HALL_WB_FIT
        hall["natural_position"] = hall["tm_sub_position"]
        hall["club_name"] = hall.get(
            "club_name",
            pd.Series(index=hall.index, dtype=object),
        ).fillna("Newcastle United")

        # Reindex in one operation to avoid DataFrame fragmentation warnings.
        combined_cols = list(dict.fromkeys(list(wb.columns) + list(hall.columns)))
        wb = wb.reindex(columns=combined_cols)
        hall = hall.reindex(columns=combined_cols)

        if not wb["player_name"].astype(str).map(is_hall).any():
            wb = pd.concat(
                [wb, hall],
                ignore_index=True,
                sort=False,
            )

    wb = wb.copy()

    # ------------------------------------------------------------------
    # Merge ACTUAL football metrics from historical season metrics.
    # The enriched file contains metadata but its non-xA football fields
    # were zero-filled, so it should not be used as the metric source.
    # ------------------------------------------------------------------

    wb["_player_key"] = player_key(wb)
    historical["_player_key"] = player_key(historical)

    metric_source = historical[["_player_key", "minutes"]].copy()
    metric_source = metric_source.rename(
        columns={"minutes": "_src_minutes"}
    )

    detected = {}
    source_map = {}

    for metric, aliases in ALIASES.items():
        source_col = find_alias(
            historical,
            aliases,
        )
        detected[metric] = source_col

        imported_col = f"_src_{metric}"

        if source_col is not None:
            metric_source[imported_col] = pd.to_numeric(
                historical[source_col],
                errors="coerce",
            )
            source_map[metric] = imported_col
        else:
            source_map[metric] = None

    metric_source = metric_source.drop_duplicates(
        "_player_key",
        keep="first",
    )

    wb = wb.merge(
        metric_source,
        on="_player_key",
        how="left",
    ).copy()

    print()
    print("=" * 110)
    print("HISTORICAL METRIC MERGE COVERAGE")
    print("=" * 110)

    usable_cols = [
        c for c in source_map.values()
        if c is not None
    ]

    matched = int(
        wb[usable_cols]
        .notna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Rows with matched historical metrics: {matched} / {len(wb)}"
    )

    print()
    print("=" * 110)
    print("FULLBACK METRIC SOURCES")
    print("=" * 110)

    for metric, source in detected.items():
        print(
            f"{metric:<24} {source if source else 'MISSING'}"
        )

    # All chosen aliases prefer explicit *_per90 fields.
    for metric, imported in source_map.items():
        out_col = f"{metric}_p90"

        if imported is None:
            wb[out_col] = np.nan
            continue

        source_name = detected[metric]
        raw = pd.to_numeric(
            wb[imported],
            errors="coerce",
        )

        if source_name and "per90" in source_name.lower():
            wb[out_col] = raw
        else:
            wb[out_col] = p90(
                raw,
                pd.to_numeric(
                    wb["_src_minutes"],
                    errors="coerce",
                ),
            )

    # ------------------------------------------------------------------
    # ATTACKING CREATION INDEX
    #
    # We deliberately do NOT call this "progression":
    # progressive carries and opposition-half pass fields are zero upstream.
    #
    # 30% xA
    # 25% key passes
    # 20% accurate crosses
    # 15% successful passes
    # 10% touches
    #
    # This measures attacking involvement/creation from fullback areas.
    # ------------------------------------------------------------------

    creation_inputs = {
        "xa_p90": 0.30,
        "key_passes_p90": 0.25,
        "accurate_crosses_p90": 0.20,
        "successful_passes_p90": 0.15,
        "touches_p90": 0.10,
    }

    weighted = pd.Series(
        0.0,
        index=wb.index,
    )
    observed = pd.Series(
        0.0,
        index=wb.index,
    )

    for col, weight in creation_inputs.items():
        z = winsor_z(
            wb[col]
        )
        valid = z.notna()

        weighted.loc[valid] += (
            z.loc[valid]
            * weight
        )
        observed.loc[valid] += weight

    wb["attacking_creation_index"] = (
        50
        + 10
        * np.where(
            observed > 0,
            weighted / observed,
            np.nan,
        )
    ).clip(20, 80)

    # ------------------------------------------------------------------
    # PHYSICAL / PRESSING ACTIVITY INDEX
    #
    # 30% duels won
    # 15% aerials won
    # 25% recoveries
    # 15% interceptions
    # 15% tackles won
    # ------------------------------------------------------------------

    physical_inputs = {
        "duels_won_p90": 0.30,
        "aerials_won_p90": 0.15,
        "recoveries_p90": 0.25,
        "interceptions_p90": 0.15,
        "tackles_won_p90": 0.15,
    }

    weighted = pd.Series(
        0.0,
        index=wb.index,
    )
    observed = pd.Series(
        0.0,
        index=wb.index,
    )

    for col, weight in physical_inputs.items():
        z = winsor_z(
            wb[col]
        )
        valid = z.notna()

        weighted.loc[valid] += (
            z.loc[valid]
            * weight
        )
        observed.loc[valid] += weight

    wb["physical_activity_index"] = (
        50
        + 10
        * np.where(
            observed > 0,
            weighted / observed,
            np.nan,
        )
    ).clip(20, 80)

    wb["role_fit"] = pd.to_numeric(
        wb["role_fit"],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # JAISSLE FB SCORE
    #
    # 40% overall vertical wingback fit
    # 30% attacking creation
    # 30% physical / pressing activity
    #
    # This is the main Newcastle-specific ranking metric.
    # ------------------------------------------------------------------
    wb["jaissle_fb_score"] = (
        0.40 * wb["role_fit"]
        + 0.30 * wb["attacking_creation_index"]
        + 0.30 * wb["physical_activity_index"]
    )

    print()
    print("=" * 110)
    print("DERIVED METRIC COVERAGE")
    print("=" * 110)

    for col in [
        "xa_p90",
        "key_passes_p90",
        "accurate_crosses_p90",
        "successful_passes_p90",
        "touches_p90",
        "duels_won_p90",
        "aerials_won_p90",
        "recoveries_p90",
        "interceptions_p90",
        "tackles_won_p90",
        "attacking_creation_index",
        "physical_activity_index",
        "jaissle_fb_score",
    ]:
        print(
            f"{col:<32} "
            f"{int(wb[col].notna().sum())} / {len(wb)}"
        )

    wb = wb.sort_values(
        [
            "jaissle_fb_score",
            "role_fit",
            "attacking_creation_index",
        ],
        ascending=False,
    )

    return wb


# ======================================================================================
# CHART
# ======================================================================================

def build_chart(df: pd.DataFrame, top_n: int = 30):
    chart = df.dropna(
        subset=[
            "role_fit",
            "attacking_creation_index",
        ]
    ).copy()

    hall = chart[
        chart["player_name"].astype(str).map(is_hall)
    ].copy()

    candidates = chart[
        ~chart["player_name"].astype(str).map(is_hall)
    ].copy()

    # Take strongest overall WB candidates, then force Hall benchmark in.
    top = candidates.head(top_n).copy()

    if not hall.empty:
        top = pd.concat(
            [top, hall.head(1)],
            ignore_index=True,
            sort=False,
        )

    if top.empty:
        return

    physical = pd.to_numeric(
        top["physical_activity_index"],
        errors="coerce",
    )

    median_physical = (
        float(physical.median())
        if physical.notna().any()
        else 50.0
    )

    top["_physical"] = physical.fillna(median_physical)

    # Bubble size now represents football activity/physicality, NOT price.
    top["_bubble_size"] = (
        np.clip(
            top["_physical"] - 30.0,
            5.0,
            None,
        )
        * 12.0
    )

    color_values = pd.to_numeric(
        top["xa_p90"],
        errors="coerce",
    )

    median_color = (
        float(color_values.median())
        if color_values.notna().any()
        else 50.0
    )

    top["_creative_color"] = color_values.fillna(median_color)

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    scatter = ax.scatter(
        top["role_fit"],
        top["attacking_creation_index"],
        s=top["_bubble_size"],
        c=top["_creative_color"],
        cmap="viridis",
        alpha=0.76,
        edgecolors="black",
        linewidths=0.8,
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.015,
    )

    cbar.set_label(
        "xA per 90"
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    for _, row in top.iterrows():
        name = str(row["player_name"])
        ref = is_hall(name)

        score = row.get("jaissle_fb_score", np.nan)

        base_label = (
            f"{name} [Hall ref]"
            if ref
            else name
        )

        label = (
            f"{base_label} ({score:.1f})"
            if pd.notna(score)
            else base_label
        )

        ax.annotate(
            label,
            (
                row["role_fit"],
                row["attacking_creation_index"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5 if ref else 8,
            fontweight="bold" if ref else "normal",
        )

    ax.set_xlabel(
        "Vertical Wingback Fit"
    )

    ax.set_ylabel(
        "Attacking Creation Index"
    )

    ax.set_title(
        "Newcastle Left-Back Recruitment — Jaissle FB Score\n"
        "Bubble size = physical activity | Color = creative output"
    )

    fig.text(
        0.01,
        0.046,
        "Attacking Creation Index: 30% xA, 25% key passes, 20% accurate crosses, "
        "15% successful passes, 10% touches.",
        fontsize=7.5,
        ha="left",
    )

    fig.text(
        0.01,
        0.020,
        "Bubble size: duels won + aerials won + recoveries + tackles. "
        "This chart prioritizes attacking creation plus the physical activity required by the tactical model.",
        fontsize=7.5,
        ha="left",
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    fig.savefig(
        OUT_CHART,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


# ======================================================================================
# REPORT
# ======================================================================================

def save_html(df: pd.DataFrame):
    import html

    rows = []

    for _, row in df.head(80).iterrows():
        name = str(row.get("player_name", ""))

        if is_hall(name):
            name += " [Newcastle ref]"

        club = row.get("club_name", "")
        if pd.isna(club):
            club = ""

        age = row.get("age", np.nan)
        age_text = "" if pd.isna(age) else str(int(round(float(age))))

        value = row.get("market_value_eur_m", np.nan)
        value_text = "" if pd.isna(value) else f"€{float(value):.1f}M"

        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{html.escape(str(club))}</td>"
            f"<td>{html.escape(str(row.get('league_slug', '')))}</td>"
            f"<td>{age_text}</td>"
            f"<td>{float(row['role_fit']):.1f}</td>"
            f"<td>{float(row['jaissle_fb_score']):.1f}</td>"
            f"<td>{float(row['attacking_creation_index']):.1f}</td>"
            f"<td>{float(row['xa_p90']):.2f}</td>"
            f"<td>{float(row['physical_activity_index']):.1f}</td>"
            f"<td>{'' if pd.isna(row.get('progressive_carries_p90')) else format(float(row.get('progressive_carries_p90')), '.2f')}</td>" 
            f"<td>{'' if pd.isna(row.get('xa_p90')) else format(float(row.get('xa_p90')), '.2f')}</td>" 
            f"<td>{'' if pd.isna(row.get('key_passes_p90')) else format(float(row.get('key_passes_p90')), '.2f')}</td>" 
            f"<td>{'' if pd.isna(row.get('duels_won_p90')) else format(float(row.get('duels_won_p90')), '.2f')}</td>" 
            f"<td>{value_text}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Left-Back Progressive Recruitment</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; background:#f5f7f9; color:#17202a; }}
.note {{ background:white; padding:14px; border-left:4px solid #566573; margin:16px 0; line-height:1.5; }}
table {{ width:100%; border-collapse:collapse; background:white; font-size:12px; }}
th {{ background:#17202a; color:white; padding:8px; text-align:left; }}
td {{ padding:8px; border-bottom:1px solid #e5e7e9; }}
tr:nth-child(even) {{ background:#fafafa; }}
</style>
</head>
<body>

<h1>Left-Back / Wingback Recruitment — Creation + Physical Profile</h1>

<div class="note">
<strong>Why this view:</strong> age is useful for squad planning, but it does not describe the tactical requirement.
This analysis instead measures whether a left-back can create from advanced zones and support
an aggressive pressing system with physical activity.<br><br>

<strong>Attacking Creation Index:</strong>
30% xA, 25% key passes, 20% accurate crosses, 15% successful passes, 10% touches.<br><br>

<strong>Jaissle FB Score:</strong>
40% WB Fit, 30% Attacking Creation, 30% Physical Activity.<br><br>

<strong>Physical Activity Index:</strong>
40% duels won, 20% aerials won, 25% recoveries, 15% tackles.
</div>

<table>
<thead>
<tr>
<th>Player</th>
<th>Club</th>
<th>League</th>
<th>Age</th>
<th>WB Fit</th>
<th>Prog-Creation</th>
<th>xA/90</th>
<th>Physical Activity</th>
<th>Prog Carries/90</th>
<th>xA/90</th>
<th>Chances/90</th>
<th>Duels Won/90</th>
<th>Value</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>

</body>
</html>
"""

    OUT_HTML.write_text(
        doc,
        encoding="utf-8",
    )


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = build_pool()

    export_cols = [
        c
        for c in [
            "player_name",
            "club_name",
            "league_slug",
            "age",
            "minutes",
            "role_fit",
            "attacking_creation_index",
            "xa_p90",
            "physical_activity_index",
                                                "xa_p90",
            "key_passes_p90",
            "accurate_crosses_p90",
            "duels_won_p90",
            "aerials_won_p90",
            "recoveries_p90",
            "tackles_won_p90",
            "market_value_eur_m",
            "natural_position",
            "tm_sub_position",
        ]
        if c in result.columns
    ]

    result[export_cols].to_csv(
        OUT_CSV,
        index=False,
    )

    save_html(result)
    build_chart(result)

    print()
    print("=" * 120)
    print("TOP LEFT-BACKS — CREATION / PHYSICAL VIEW")
    print("=" * 120)

    display_cols = [
        "player_name",
        "club_name",
        "age",
        "role_fit",
        "jaissle_fb_score",
        "attacking_creation_index",
        "physical_activity_index",
        "xa_p90",
        "key_passes_p90",
        "accurate_crosses_p90",
        "duels_won_p90",
        "recoveries_p90",
        "interceptions_p90",
        "tackles_won_p90",
        "market_value_eur_m",
    ]

    display_cols = [
        c
        for c in display_cols
        if c in result.columns
    ]

    # Defensive safeguard: no duplicate source columns.
    display_cols = list(dict.fromkeys(display_cols))

    display = (
        result.loc[:, display_cols]
        .head(30)
        .copy()
    )

    display = display.rename(
        columns={
            "player_name": "Player",
            "club_name": "Club",
            "age": "Age",
            "role_fit": "WB Fit",
            "jaissle_fb_score": "Jaissle FB Score",
            "attacking_creation_index": "Creation",
            "physical_activity_index": "Physical",
            "xa_p90": "xA/90",
            "key_passes_p90": "Key Passes/90",
            "accurate_crosses_p90": "Acc Crosses/90",
            "duels_won_p90": "Duels Won/90",
            "recoveries_p90": "Recoveries/90",
            "interceptions_p90": "Interceptions/90",
            "tackles_won_p90": "Tackles Won/90",
            "market_value_eur_m": "Value €M",
        }
    )

    numeric_cols = [
        "Age",
        "WB Fit",
        "Jaissle FB Score",
        "Creation",
        "Physical",
        "xA/90",
        "Key Passes/90",
        "Acc Crosses/90",
        "Duels Won/90",
        "Recoveries/90",
        "Interceptions/90",
        "Tackles Won/90",
        "Value €M",
    ]

    for col in numeric_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(
                display[col],
                errors="coerce",
            ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(" ", OUT_CSV)
    print(" ", OUT_HTML)
    print(" ", OUT_CHART)


if __name__ == "__main__":
    main()
