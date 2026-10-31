from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = ROOT / "scripts" / "bsd" / "analyze" / "role_scouting"
OUT_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"

BENCHMARK_SCRIPT = "build_newcastle_historical_benchmarks.py"
BENCHMARK_CSV = OUT_DIR / "newcastle_jaissle_role_benchmarks.csv"
OPTIMAL_BENCHMARK_CSV = OUT_DIR / "newcastle_jaissle_optimal_roles.csv"

MAX_TARGET_VALUE_M = 70.0
REVIEW_SCORE_THRESHOLD = 52.5
REVIEW_FIT_THRESHOLD = 52.5

# Squad-management policy:
# - Under-23 players are developmental and should not be auto-sold.
# - Recent/new signings are protected while adapting to Jaissle.
# - Jacob Ramsey is explicitly touchable despite being a relatively recent signing
#   because his role remains unclear and his market value could be reallocated.
PROTECTED_DEVELOPMENTAL_PLAYERS = {
    "Aladji Bamba",
    "Lewis Miley",
    "Sean Steur",
    "Bazoumana Toure",
    "Matias Fernandez-Pardo",
    "William Osula",
}

PROTECTED_NEW_SIGNINGS = {
    "Anthony Elanga",
    "Malick Thiaw",
    "Nico Gonzalez",
    "Tino Livramento",
    "Yoane Wissa",
    "Amar Dedić",
}

TOUCHABLE_OVERRIDES = {
    "Jacob Ramsey",
}

# Players who should appear in the squad summary even if the historical
# recruitment benchmark layer cannot score them (GK is outside recruitment;
# Dedić's 2025/26 Portugal data is outside the six-league historical universe).
MANUAL_SQUAD_ROWS = [
    {"Player": "Nick Pope", "Age": 34, "Squad Position": "GK", "Display Role": "Goalkeeper", "Decision Flag": "KEEP / SQUAD CONTEXT"},
    {"Player": "Lukáš Horníček", "Age": 24, "Squad Position": "GK", "Display Role": "Sweeper / Build-Up GK", "Decision Flag": "KEEP / ROLE FIT"},
    {"Player": "Jaouen", "Age": 20, "Squad Position": "GK", "Display Role": "Goalkeeper", "Decision Flag": "DEVELOPMENTAL / PROTECTED"},
    {"Player": "Mark Gillespie", "Age": 34, "Squad Position": "GK", "Display Role": "Goalkeeper", "Decision Flag": "KEEP / SQUAD CONTEXT"},
    {"Player": "Amar Dedić", "Age": 24, "Squad Position": "FB/WB", "Display Role": "Vertical Full-Back / Wingback", "Decision Flag": "NEW SIGNING / ADAPTATION"},
]

# FIFA-style summary order.
POSITION_ORDER = {
    "GK": 0,
    "CB": 1,
    "FB/WB": 2,
    "CDM": 3,
    "CM": 4,
    "CAM": 5,
    "W": 6,
    "ST": 7,
}

ROLE_JOBS = [
    {
        "name": "Creative 10 / CAM",
        "role_key": "creative_cam",
        "script": "analyze_cam.py",
        "csv": "creative_10_jaissle_detailed.csv",
        "score": "jaissle_cam_score",
        "fit": "role_fit",
    },
    {
        "name": "Focal Striker",
        "role_key": "focal_striker",
        "script": "analyze_striker.py",
        "csv": "focal_striker_jaissle_detailed.csv",
        "score": "jaissle_striker_score",
        "fit": "role_fit",
    },
    {
        "name": "High-Line Center Back",
        "role_key": "center_back",
        "script": "analyze_centerback.py",
        "csv": "center_back_jaissle_detailed.csv",
        "score": "jaissle_cb_score",
        "fit": "role_fit",
    },
    {
        "name": "Workhorse Central Midfielder",
        "role_key": "advanced_pivot",
        "script": "analyze_workhorse_cm.py",
        "csv": "workhorse_cm_jaissle_detailed.csv",
        "score": "jaissle_workhorse_score",
        "fit": "role_fit",
    },
    {
        "name": "Left Wingback / Fullback",
        "role_key": "vertical_wingback",
        "script": "analyze_leftback.py",
        "csv": "left_wingback_recruitment_progressive.csv",
        "score": "jaissle_fb_score",
        "fit": "role_fit",
    },
    {
        "name": "Winger",
        "role_key": "vertical_winger",
        "script": "analyze_winger.py",
        "csv": "winger_jaissle_archetypes.csv",
        "score": "best_winger_score",
        "fit": "hybrid_score",
        "archetype": "best_winger_archetype",
    },
]


def header(title: str):
    print()
    print("=" * 140)
    print(title)
    print("=" * 140)


def run_script(filename: str, label: str) -> bool:
    path = SCRIPT_DIR / filename
    if not path.exists():
        print(f"[MISSING] {path}")
        return False

    header(f"RUNNING: {label}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        text=True,
    )
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAILED'}] {label}")
    return ok


def external_targets(job: dict) -> pd.DataFrame:
    path = OUT_DIR / job["csv"]
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if job["score"] not in df.columns:
        return pd.DataFrame()

    df[job["score"]] = pd.to_numeric(df[job["score"]], errors="coerce")
    value = pd.to_numeric(
        df.get("market_value_eur_m"), errors="coerce"
    )
    eligible = value.isna() | value.le(MAX_TARGET_VALUE_M)

    if "is_newcastle_benchmark" in df.columns:
        bench = (
            df["is_newcastle_benchmark"].astype(str)
            .str.lower().isin(["true", "1", "yes"])
        )
        eligible &= ~bench

    df = df[eligible].copy()
    df = df.sort_values(job["score"], ascending=False).head(5)

    data = {
        "Role": job["name"],
        "Role Key": job["role_key"],
        "Player": df.get("player_name"),
        "Club": df.get("club_name"),
        "Age": df.get("age"),
        "Role Score": df[job["score"]],
        "Role Fit": pd.to_numeric(df.get(job["fit"]), errors="coerce"),
        "Value €M": value.loc[df.index],
    }

    if job.get("archetype") and job["archetype"] in df.columns:
        data["Best Archetype"] = df[job["archetype"]]

    return pd.DataFrame(data)


def benchmark_rows() -> pd.DataFrame:
    """
    Retention decisions use ONE row per Newcastle player:
    the role in which that player scores best.

    The full benchmark CSV still exists for role-by-role scouting,
    but poor secondary-role scores do not trigger replacement reviews.
    """
    if not OPTIMAL_BENCHMARK_CSV.exists():
        return pd.DataFrame()

    raw = pd.read_csv(OPTIMAL_BENCHMARK_CSV)

    if raw.empty:
        return pd.DataFrame()

    raw["Role Score"] = pd.to_numeric(
        raw.get("optimal_jaissle_score"),
        errors="coerce",
    )

    raw["Role Fit"] = pd.to_numeric(
        raw.get("optimal_role_fit"),
        errors="coerce",
    )

    # If the detailed score was unavailable, the base role fit is the
    # best comparable fallback.
    raw["Role Score"] = raw["Role Score"].where(
        raw["Role Score"].notna(),
        raw["Role Fit"],
    )

    raw["Value €M"] = pd.to_numeric(
        raw.get("market_value_eur_m"),
        errors="coerce",
    )

    role_name_map = {
        job["role_key"]: job["name"]
        for job in ROLE_JOBS
    }

    raw["Role"] = (
        raw["optimal_role_key"]
        .map(role_name_map)
        .fillna(raw.get("optimal_role"))
    )

    out = pd.DataFrame({
        "Role": raw["Role"],
        "Role Key": raw["optimal_role_key"],
        "Player": raw["player_name"],
        "Historical Match": raw.get("historical_player_name"),
        "Historical League": raw.get("historical_league"),
        "Historical Minutes": raw.get("historical_minutes"),
        "Confidence": raw.get("sample_confidence"),
        "Age": raw.get("age"),
        "Role Score": raw["Role Score"],
        "Role Fit": raw["Role Fit"],
        "Coverage %": pd.to_numeric(
            raw.get("optimal_metric_coverage"),
            errors="coerce",
        ),
        "Value €M": raw["Value €M"],
        "Optimal Role": raw.get("optimal_role"),
        "Best Winger Archetype": raw.get("best_winger_archetype"),
    })

    out["Display Role"] = out.apply(display_optimal_role, axis=1)
    out["Squad Position"] = out.apply(summary_position_bucket, axis=1)
    out["Position Order"] = out["Squad Position"].map(POSITION_ORDER).fillna(99)

    return out


def player_is_protected(row: pd.Series) -> tuple[bool, str]:
    name = str(row["Player"])
    age = pd.to_numeric(pd.Series([row.get("Age")]), errors="coerce").iloc[0]

    if name in TOUCHABLE_OVERRIDES:
        return False, ""

    if name in PROTECTED_NEW_SIGNINGS:
        return True, "NEW SIGNING / ADAPTATION"

    if name in PROTECTED_DEVELOPMENTAL_PLAYERS:
        return True, "DEVELOPMENTAL / PROTECTED"

    if pd.notna(age) and age < 23:
        return True, "U23 DEVELOPMENTAL"

    return False, ""


def decision(row: pd.Series) -> str:
    protected, reason = player_is_protected(row)
    if protected:
        return reason

    confidence = str(row.get("Confidence", "")).upper()
    if confidence == "LOW" and str(row.get("Player")) not in TOUCHABLE_OVERRIDES:
        return "MONITOR / LOW SAMPLE"

    score = row["Role Score"]
    fit = row["Role Fit"]

    score_bad = pd.notna(score) and score < REVIEW_SCORE_THRESHOLD
    fit_bad = pd.notna(fit) and fit < REVIEW_FIT_THRESHOLD

    if score_bad and fit_bad:
        return "REPLACE / SELL REVIEW"
    if score_bad or fit_bad:
        return "DEPTH / VALUE REVIEW"
    if pd.notna(score) or pd.notna(fit):
        return "KEEP / ROLE FIT"
    return "MANUAL REVIEW"


def summary_position_bucket(row: pd.Series) -> str:
    role_key = str(row.get("Role Key", ""))
    role = str(row.get("Optimal Role", row.get("Role", "")))
    player = str(row.get("Player", ""))

    # Explicit player/category handling for current squad presentation.
    if player in {"Sven Botman", "Fabian Schar", "Malick Thiaw", "Dan Burn"}:
        return "CB"

    if player in {"Lewis Hall", "Tino Livramento", "Amar Dedić"}:
        return "FB/WB"

    if player in {"Nico Gonzalez"}:
        return "CDM"

    if player in {
        "Aladji Bamba", "Joelinton", "Jacob Ramsey",
        "Joe Willock", "Lewis Miley", "Sean Steur",
    }:
        # Creative 10 can still be their best score, but these are midfielders
        # in squad-structure terms unless explicitly used as a CAM.
        if role_key == "creative_cam" and player not in {"Nico Gonzalez"}:
            return "CM"
        return "CM"

    if role_key == "creative_cam":
        return "CAM"

    if role_key == "vertical_winger":
        return "W"

    if role_key == "focal_striker":
        return "ST"

    if role_key == "center_back":
        return "CB"

    if role_key == "vertical_wingback":
        return "FB/WB"

    if role_key == "advanced_pivot":
        return "CM"

    return "CM"


def display_optimal_role(row: pd.Series) -> str:
    role_key = str(row.get("Role Key", ""))

    if role_key == "vertical_winger":
        archetype = row.get("Best Winger Archetype")
        if pd.notna(archetype) and str(archetype).strip():
            return str(archetype)

    return str(row.get("Optimal Role", row.get("Role", "")))


def replacement_view(
    external: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    if external.empty or benchmarks.empty:
        return pd.DataFrame()

    review = benchmarks[
        benchmarks["Decision Flag"].isin(
            ["REPLACE / SELL REVIEW", "DEPTH / VALUE REVIEW"]
        )
    ].copy()

    rows = []
    for _, current in review.iterrows():
        candidates = external[
            external["Role Key"].eq(current["Role Key"])
        ].copy()

        candidates = candidates[
            candidates["Role Score"].ge(REVIEW_SCORE_THRESHOLD)
            & candidates["Role Fit"].ge(REVIEW_FIT_THRESHOLD)
        ]

        for _, target in candidates.head(3).iterrows():
            current_value = current["Value €M"]
            target_value = target["Value €M"]
            row = {
                "Role": current["Role"],
                "NUFC Player": current["Player"],
                "Decision Flag": current["Decision Flag"],
                "NUFC Score": current["Role Score"],
                "NUFC Fit": current["Role Fit"],
                "NUFC Value €M": current_value,
                "Replacement": target["Player"],
                "Replacement Score": target["Role Score"],
                "Replacement Fit": target["Role Fit"],
                "Replacement Value €M": target_value,
                "Potential Value Freed €M": (
                    current_value - target_value
                    if pd.notna(current_value) and pd.notna(target_value)
                    else pd.NA
                ),
            }

            if "Best Archetype" in target.index:
                row["Replacement Archetype"] = target["Best Archetype"]

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    header("NEWCASTLE JAISSLE MASTER RECRUITMENT PIPELINE")
    print(f"External target cap: €{MAX_TARGET_VALUE_M:.0f}M")
    print(
        f"Review thresholds: score < {REVIEW_SCORE_THRESHOLD:.1f}, "
        f"fit < {REVIEW_FIT_THRESHOLD:.1f}"
    )
    print("Protected: U23 developmental players + recent signings")
    print("Touchable override: Jacob Ramsey")

    statuses = []

    # First generate the external role pools, because the benchmark builder
    # uses those pools for identical component standardization.
    for job in ROLE_JOBS:
        ok = run_script(job["script"], job["name"])
        statuses.append({
            "Stage": job["name"],
            "Script": job["script"],
            "Status": "OK" if ok else "FAILED",
        })

    # Then bridge the current Newcastle roster to its 2025/26 historical rows
    # and calculate the detailed Jaissle scores using the same role pools.
    bench_ok = run_script(
        BENCHMARK_SCRIPT,
        "Newcastle Historical Benchmark Bridge",
    )
    statuses.append({
        "Stage": "Newcastle Historical Benchmark Bridge",
        "Script": BENCHMARK_SCRIPT,
        "Status": "OK" if bench_ok else "FAILED",
    })

    external_frames = [
        external_targets(job) for job in ROLE_JOBS
    ]
    external_frames = [x for x in external_frames if not x.empty]
    external = (
        pd.concat(external_frames, ignore_index=True)
        if external_frames else pd.DataFrame()
    )

    benchmarks = benchmark_rows()
    if not benchmarks.empty:
        benchmarks["Decision Flag"] = benchmarks.apply(decision, axis=1)

    manual_rows = pd.DataFrame(MANUAL_SQUAD_ROWS)
    if not manual_rows.empty:
        manual_rows["Role"] = manual_rows["Display Role"]
        manual_rows["Role Key"] = pd.NA
        manual_rows["Historical Match"] = pd.NA
        manual_rows["Historical League"] = pd.NA
        manual_rows["Historical Minutes"] = pd.NA
        manual_rows["Confidence"] = "MANUAL"
        manual_rows["Role Score"] = pd.NA
        manual_rows["Role Fit"] = pd.NA
        manual_rows["Coverage %"] = pd.NA
        manual_rows["Value €M"] = pd.NA
        manual_rows["Optimal Role"] = manual_rows["Display Role"]
        manual_rows["Best Winger Archetype"] = pd.NA
        manual_rows["Position Order"] = (
            manual_rows["Squad Position"].map(POSITION_ORDER).fillna(99)
        )

        existing_names = (
            set(benchmarks["Player"].astype(str))
            if not benchmarks.empty
            else set()
        )
        manual_rows = manual_rows[
            ~manual_rows["Player"].astype(str).isin(existing_names)
        ]

        benchmarks = pd.concat(
            [benchmarks, manual_rows],
            ignore_index=True,
            sort=False,
        )

    if not benchmarks.empty:
        benchmarks = benchmarks.sort_values(
            ["Position Order", "Squad Position", "Player"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    replacements = replacement_view(external, benchmarks)

    # Central Newcastle benchmark view. Individual role scripts may have their own
    # benchmark labels, but this bridge is the single source of truth.
    if BENCHMARK_CSV.exists():
        long_bench = pd.read_csv(BENCHMARK_CSV)
        if not long_bench.empty:
            role_label_map = {job["role_key"]: job["name"] for job in ROLE_JOBS}
            long_bench["Role"] = long_bench["role_key"].map(role_label_map)
            long_bench["Score"] = pd.to_numeric(
                long_bench.get("jaissle_role_score"), errors="coerce"
            )
            long_bench["Fit"] = pd.to_numeric(
                long_bench.get("role_fit"), errors="coerce"
            )
            long_bench["Archetype"] = (
                long_bench["best_winger_archetype"]
                if "best_winger_archetype" in long_bench.columns
                else pd.NA
            )
            header("NEWCASTLE BENCHMARKS BY ROLE — CENTRAL BRIDGE")
            fields = [
                "Role", "player_name", "Archetype", "Score", "Fit",
                "historical_minutes", "sample_confidence", "market_value_eur_m"
            ]
            fields = [c for c in fields if c in long_bench.columns]
            bench_show = long_bench[fields].copy().rename(columns={
                "player_name": "Player",
                "historical_minutes": "Minutes",
                "sample_confidence": "Confidence",
                "market_value_eur_m": "Value €M",
            })
            print(
                bench_show.sort_values(
                    ["Role", "Score"], ascending=[True, False], na_position="last"
                ).to_string(index=False)
            )

    header(f"REALISTIC EXTERNAL TARGETS (≤ €{MAX_TARGET_VALUE_M:.0f}M)")
    print(
        external.to_string(index=False)
        if not external.empty
        else "No external targets available."
    )

    header("NEWCASTLE SQUAD SUMMARY — OPTIMAL ROLE / ARCHETYPE")
    if benchmarks.empty:
        print("No Newcastle benchmark rows available.")
    else:
        cols = [
            "Squad Position", "Player", "Display Role",
            "Historical Match", "Historical League",
            "Historical Minutes", "Confidence", "Role Score", "Role Fit",
            "Coverage %", "Value €M", "Decision Flag",
        ]
        print(benchmarks[cols].to_string(index=False))

    header("REPLACEMENT / VALUE REVIEW")
    print(
        replacements.to_string(index=False)
        if not replacements.empty
        else "No players currently trigger a replacement/value review."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    external.to_csv(
        OUT_DIR / "jaissle_all_roles_external_top5.csv",
        index=False,
    )
    benchmarks.to_csv(
        OUT_DIR / "jaissle_newcastle_benchmarks_all_roles.csv",
        index=False,
    )
    replacements.to_csv(
        OUT_DIR / "jaissle_replacement_review.csv",
        index=False,
    )

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(
        OUT_DIR / "jaissle_pipeline_status.csv",
        index=False,
    )

    def table(df):
        return (
            df.to_html(index=False, float_format=lambda x: f"{x:.2f}")
            if not df.empty
            else "<p>No rows available.</p>"
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Newcastle Jaissle Recruitment Master Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin:30px; background:#f4f6f7; color:#17202a; }}
.card {{ background:white; padding:18px; margin-bottom:22px; border-left:5px solid #34495e; }}
table {{ width:100%; border-collapse:collapse; background:white; margin-bottom:28px; font-size:12px; }}
th {{ background:#17202a; color:white; padding:8px; }}
td {{ padding:7px; border-bottom:1px solid #ddd; }}
</style>
</head>
<body>
<h1>Newcastle United — Jaissle Recruitment Master Report</h1>
<div class="card">
<strong>External target ceiling:</strong> €{MAX_TARGET_VALUE_M:.0f}M.<br>
<strong>Review threshold:</strong> Jaissle score or role fit below {REVIEW_SCORE_THRESHOLD:.1f}.<br><br>
Newcastle benchmarks are resolved from the full 2025/26 historical universe,
not from the filtered recruitment shortlist. Retention decisions use each player's
highest-scoring plausible Jaissle role; poor scores in secondary positions do not
trigger a sale review. Wingers are displayed by their best archetype (Wide Creator,
Vertical / Box Winger, or Hybrid) rather than a generic vertical-winger label.
U23 developmental players and recent signings are protected from automatic sell
recommendations while they develop or adapt. Jacob Ramsey is an explicit touchable
override because his role remains unclear. Leadership, mentoring, injury history
and dressing-room influence remain qualitative decision factors.
</div>
<h2>Realistic External Targets</h2>
{table(external)}
<h2>Newcastle Historical Benchmarks</h2>
{table(benchmarks)}
<h2>Replacement / Value Review</h2>
{table(replacements)}
<h2>Pipeline Status</h2>
{table(status_df)}
</body>
</html>"""

    report = OUT_DIR / "jaissle_recruitment_master_report.html"
    report.write_text(html, encoding="utf-8")

    print()
    print("Saved master report:")
    print(" ", report)


if __name__ == "__main__":
    main()
