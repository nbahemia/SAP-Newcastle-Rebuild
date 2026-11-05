from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================
# PATHS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]
SHORTLIST_DIR = ROOT / "data" / "processed" / "bsd" / "shortlist"

RECRUITMENT_FILE = SHORTLIST_DIR / "recruitment_value_analysis.csv"
PRIORITY_FILE = SHORTLIST_DIR / "priority_target_full_check.csv"
YOUNG_CB_FILE = SHORTLIST_DIR / "young_cb_recruitment.csv"
WINGER_FILE = SHORTLIST_DIR / "bona_fide_winger_recruitment.csv"

OUT = SHORTLIST_DIR / "final_shortlist_analysis.csv"
OUT_METRICS = SHORTLIST_DIR / "final_shortlist_manual_metrics.csv"

VIS_DIR = SHORTLIST_DIR / "final_shortlist_visualizations"

TERMINAL_WIDTH = 108


# ======================================================================================
# FINAL WORKING SHORTLIST
# ======================================================================================

SHORTLIST = {
    "Nikola Krstović": {
        "position": "ST",
        "role": "Focal Striker",
        "project_fee": 33.0,
        "priority": "CORE",
        "status": "BUY",
    },
    "Kenan Yıldız": {
        "position": "LW / AM",
        "role": "Premium Winger / Creator",
        "project_fee": 90.0,
        "priority": "CORE",
        "status": "BUY",
    },
    "Rayan Aït-Nouri": {
        "position": "LB / LWB",
        "role": "Creative Left Back",
        "project_fee": 30.0,
        "priority": "CORE",
        "status": "BUY",
    },
    "Lazar Samardžić": {
        "position": "CAM / 10",
        "role": "Creative 10",
        "project_fee": 20.0,
        "priority": "CORE",
        "status": "BUY",
    },
    "Bart van Rooij": {
        "position": "RB",
        "role": "RB Competition",
        "project_fee": 13.0,
        "priority": "CORE",
        "status": "BUY",
    },
    "Filip Lissah": {
        "position": "RB / CB",
        "role": "Developmental Defensive Depth",
        "project_fee": 1.0,
        "priority": "CORE VALUE",
        "status": "BUY",
    },
    "Chibuike Nwaiwu": {
        "position": "CB",
        "role": "Young CB / Succession",
        "project_fee": 10.0,
        "priority": "OPTIONAL",
        "status": "SHORTLIST",
    },
    "Francisco Conceição": {
        "position": "RW / AM",
        "role": "Creator RW / CAM Depth",
        "project_fee": 30.0,
        "priority": "OPTIONAL",
        "status": "SHORTLIST",
    },
}

ALIASES = {
    "Nikola Krstović": ["N. Krstović", "Nikola Krstovic"],
    "Kenan Yıldız": ["K. Yıldız", "Kenan Yildiz"],
    "Rayan Aït-Nouri": ["R. Aït-Nouri", "Rayan Ait-Nouri"],
    "Lazar Samardžić": ["L. Samardžić", "Lazar Samardzic"],
    "Bart van Rooij": ["B. van Rooij"],
    "Filip Lissah": ["F. Lissah"],
    "Chibuike Nwaiwu": ["C. Nwaiwu"],
    "Francisco Conceição": ["F. Conceição", "Francisco Conceicao"],
}

MANUAL_METRICS = {
    "Chibuike Nwaiwu": {
        "Minutes": 893,
        "Passes/90": 55.94,
        "Pass %": 89.1,
        "Accurate Long Balls/90": 3.63,
        "Long Ball %": 60.0,
        "Line-Breaking Passes/90": 8.47,
        "Chances Created/90": 0.20,
        "Duels Won/90": 5.04,
        "Duel Win %": 63.3,
        "Aerials Won/90": 1.61,
        "Aerial Win %": 64.0,
        "Touches/90": 78.91,
        "Tackles/90": 1.61,
        "Interceptions/90": 1.71,
        "Recoveries/90": 5.04,
        "Defensive Contributions/90": 9.88,
        "Clearances/90": 5.14,
        "Dribbled Past/90": 0.81,
    },
    "Francisco Conceição": {
        "Minutes": 432,
        "xG/90": 0.27,
        "xA/90": 0.22,
        "Shots/90": 2.92,
        "Shots on Target/90": 1.25,
        "Passes/90": 37.29,
        "Pass %": 86.5,
        "Accurate Long Balls/90": 1.25,
        "Long Ball %": 85.7,
        "Line-Breaking Passes/90": 4.58,
        "Chances Created/90": 1.67,
        "Big Chances Created/90": 0.42,
        "Successful Crosses/90": 1.67,
        "Successful Dribbles/90": 1.46,
        "Dribble Success %": 43.8,
        "Touches/90": 64.58,
        "Touches In Box/90": 7.50,
        "Dispossessed/90": 0.83,
        "Recoveries/90": 3.33,
        "Possession Won Final 3rd/90": 0.42,
        "Defensive Contributions/90": 1.04,
    },
}


# ======================================================================================
# HELPERS
# ======================================================================================

def normalize_name(value) -> str:
    if value is None or pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.strip()


def alias_set(canonical: str) -> set[str]:
    names = [canonical, *ALIASES.get(canonical, [])]
    return {normalize_name(name) for name in names if normalize_name(name)}


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] Could not read {path.name}: {exc}")
        return pd.DataFrame()


def find_match(df: pd.DataFrame, canonical: str, player_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    player_col = next((col for col in player_cols if col in df.columns), None)
    if player_col is None:
        return pd.DataFrame()

    wanted = alias_set(canonical)
    return df[df[player_col].astype(str).map(normalize_name).isin(wanted)].copy()


def first_number(row: pd.Series | dict, candidates: list[str]) -> float:
    for col in candidates:
        if col in row:
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def first_text(row: pd.Series | dict, candidates: list[str]) -> str:
    for col in candidates:
        if col in row:
            value = row.get(col)
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none", "<na>"}:
                    return text
    return ""


def fmt_num(value, digits=2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_money(value) -> str:
    if pd.isna(value):
        return "—"
    return f"€{float(value):.1f}m"


def short_text(value, width: int) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = str(value).strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def section(title: str) -> None:
    print()
    print("=" * TERMINAL_WIDTH)
    print(title)
    print("=" * TERMINAL_WIDTH)


def subsection(title: str) -> None:
    print()
    print(title)
    print("-" * min(len(title), TERMINAL_WIDTH))


def best_recruitment_row(recruitment: pd.DataFrame, canonical: str) -> pd.Series | None:
    matched = find_match(recruitment, canonical, ["Player"])
    if matched.empty:
        return None

    if "Recruitment Suitability Score" in matched.columns:
        matched["Recruitment Suitability Score"] = pd.to_numeric(
            matched["Recruitment Suitability Score"],
            errors="coerce",
        )
        matched = matched.sort_values(
            "Recruitment Suitability Score",
            ascending=False,
            na_position="last",
        )
    return matched.iloc[0]


def best_priority_row(priority: pd.DataFrame, canonical: str) -> pd.Series | None:
    matched = find_match(priority, canonical, ["Target", "Matched Player"])
    if matched.empty:
        return None
    return matched.iloc[0]


def best_cb_row(young_cb: pd.DataFrame, canonical: str) -> pd.Series | None:
    matched = find_match(young_cb, canonical, ["Player"])
    if matched.empty:
        return None
    return matched.iloc[0]


def best_winger_row(winger: pd.DataFrame, canonical: str) -> pd.Series | None:
    matched = find_match(winger, canonical, ["Player"])
    if matched.empty:
        return None
    return matched.iloc[0]


# ======================================================================================
# FINAL SHORTLIST TABLE
# ======================================================================================

def build_final_shortlist(
    recruitment: pd.DataFrame,
    priority: pd.DataFrame,
    young_cb: pd.DataFrame,
    winger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for canonical, config in SHORTLIST.items():
        r = best_recruitment_row(recruitment, canonical)
        p = best_priority_row(priority, canonical)
        cb = best_cb_row(young_cb, canonical)
        w = best_winger_row(winger, canonical)

        source_rows = [row for row in [r, p, cb, w] if row is not None]

        def pick_number(candidates: list[str]) -> float:
            for source in source_rows:
                value = first_number(source, candidates)
                if pd.notna(value):
                    return value
            return np.nan

        def pick_text(candidates: list[str]) -> str:
            for source in source_rows:
                value = first_text(source, candidates)
                if value:
                    return value
            return ""

        row = {
            "Player": canonical,
            "Position": config["position"],
            "Role": config["role"],
            "Priority": config["priority"],
            "Status": config["status"],
            "Project Fee €M": float(config["project_fee"]),
            "Club": pick_text(["Current Club", "TM / Current Club", "Club"]),
            "Age": pick_number(["Age"]),
            "Model Role Score": pick_number(["Target Score", "Jaissle CB", "Winger Recruitment Score"]),
            "Upgrade": pick_number(["Upgrade"]),
            "Suitability": pick_number(["Recruitment Suitability Score"]),
            "Value Score": pick_number(["Value Score"]),
            "Young CB Score": pick_number(["Young CB Recruitment Score"]),
            "Winger Score": pick_number(["Winger Recruitment Score"]),
            "Goal Threat": pick_number(["Goal Threat"]),
            "Creation": pick_number(["Creation"]),
            "1v1 / Progression": pick_number(["1v1 / Progression"]),
            "xG/90": pick_number(["xG/90"]),
            "xA/90": pick_number(["xA/90"]),
            "Shots/90": pick_number(["Shots/90"]),
            "Recoveries/90": pick_number(["Recoveries/90"]),
            "Duels Won/90": pick_number(["Duels Won/90"]),
        }

        manual = MANUAL_METRICS.get(canonical, {})
        for key, value in manual.items():
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def build_manual_metrics_file() -> pd.DataFrame:
    rows = []
    for player, metrics in MANUAL_METRICS.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "Player": player,
                    "Metric": metric,
                    "Value": value,
                    "Source": "Manual FotMob current-season sample",
                }
            )
    return pd.DataFrame(rows)


# ======================================================================================
# VISUALIZATION BUILDERS
# ======================================================================================

def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    annotate: bool = True,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    work = df[[x_col, y_col]].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[y_col])

    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(work[x_col].astype(str), work[y_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)

    if annotate:
        for i, v in enumerate(work[y_col].tolist()):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    pairs = [(label, value) for label, value in zip(labels, values) if pd.notna(value)]
    if not pairs:
        return

    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)

    for i, v in enumerate(y):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_visualizations(
    recruitment: pd.DataFrame,
    young_cb: pd.DataFrame,
    winger: pd.DataFrame,
    final: pd.DataFrame,
) -> list[Path]:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    # ----------------------------------------------------------
    # 1) Core signings by suitability
    # ----------------------------------------------------------
    core = final[final["Status"].eq("BUY")].copy()
    if not core.empty:
        core = core.sort_values("Suitability", ascending=False, na_position="last")
        path = VIS_DIR / "core_signings_suitability.png"
        save_bar_chart(
            core,
            x_col="Player",
            y_col="Suitability",
            title="Core Signings — Recruitment Suitability",
            ylabel="Suitability Score",
            out_path=path,
        )
        saved_paths.append(path)

    # ----------------------------------------------------------
    # 2) Core signings by project fee
    # ----------------------------------------------------------
    if not core.empty:
        path = VIS_DIR / "core_signings_fees.png"
        save_bar_chart(
            core.sort_values("Project Fee €M", ascending=False),
            x_col="Player",
            y_col="Project Fee €M",
            title="Core Signings — Project Fees",
            ylabel="Fee (€m)",
            out_path=path,
        )
        saved_paths.append(path)

    # ----------------------------------------------------------
    # 3) CB shortlist comparison
    # ----------------------------------------------------------
    if not young_cb.empty:
        cb_names = [
            "Chibuike Nwaiwu",
            "Ryan Flamingo",
            "Ousmane Diao",
            "Ahmetcan Kaplan",
            "S. Sané",
        ]

        cb_rows = []
        for name in cb_names:
            match = find_match(young_cb, name, ["Player"])
            if not match.empty:
                cb_rows.append(match.iloc[0])

        if cb_rows:
            cb_df = pd.DataFrame(cb_rows).copy()

            # Normalize common columns
            for col in [
                "Young CB Recruitment Score",
                "Jaissle CB",
                "Build-Up",
                "Front-Foot",
                "Physical",
                "Value €M",
            ]:
                if col in cb_df.columns:
                    cb_df[col] = pd.to_numeric(cb_df[col], errors="coerce")

            if "Player" in cb_df.columns:
                path = VIS_DIR / "cb_shortlist_model_score.png"
                save_bar_chart(
                    cb_df.sort_values("Young CB Recruitment Score", ascending=False),
                    x_col="Player",
                    y_col="Young CB Recruitment Score",
                    title="CB Shortlist — Model Score",
                    ylabel="Young CB Recruitment Score",
                    out_path=path,
                )
                saved_paths.append(path)

                # Multi-metric separate charts
                for col, filename, title in [
                    ("Jaissle CB", "cb_shortlist_jaissle_cb.png", "CB Shortlist — Jaissle CB"),
                    ("Build-Up", "cb_shortlist_build_up.png", "CB Shortlist — Build-Up"),
                    ("Front-Foot", "cb_shortlist_front_foot.png", "CB Shortlist — Front-Foot"),
                    ("Physical", "cb_shortlist_physical.png", "CB Shortlist — Physical"),
                    ("Value €M", "cb_shortlist_value.png", "CB Shortlist — Market Value"),
                ]:
                    if col in cb_df.columns:
                        path = VIS_DIR / filename
                        save_bar_chart(
                            cb_df.sort_values(col, ascending=False, na_position="last"),
                            x_col="Player",
                            y_col=col,
                            title=title,
                            ylabel=col,
                            out_path=path,
                        )
                        saved_paths.append(path)

    # ----------------------------------------------------------
    # 4) Nwaiwu manual per-90 profile vs shortlisted CB peers
    # ----------------------------------------------------------
    nwaiwu_manual = MANUAL_METRICS.get("Chibuike Nwaiwu", {})
    if nwaiwu_manual:
        cb_peer_metrics = {
            "Chibuike Nwaiwu": nwaiwu_manual.get("Line-Breaking Passes/90"),
            "Ryan Flamingo": np.nan,
            "Ousmane Diao": 6.25,
            "Ahmetcan Kaplan": np.nan,
            "S. Sané": np.nan,
        }
        path = VIS_DIR / "cb_shortlist_line_breaking_manual.png"
        save_metric_comparison_chart(
            list(cb_peer_metrics.keys()),
            list(cb_peer_metrics.values()),
            title="CB Shortlist — Line-Breaking Passes/90 (manual where available)",
            ylabel="Line-Breaking Passes / 90",
            out_path=path,
        )
        saved_paths.append(path)

    # ----------------------------------------------------------
    # 5) Winger/creator shortlist
    # ----------------------------------------------------------
    if not winger.empty:
        wing_names = [
            "Kenan Yıldız",
            "Francisco Conceição",
            "Nico Williams",
            "Julio Enciso",
            "Antonio Nusa",
            "S. Ørjasæter",
        ]

        wing_rows = []
        for name in wing_names:
            match = find_match(winger, name, ["Player"])
            if not match.empty:
                wing_rows.append(match.iloc[0])

        if wing_rows:
            wing_df = pd.DataFrame(wing_rows).copy()

            for col in [
                "Winger Recruitment Score",
                "Goal Threat",
                "Creation",
                "1v1 / Progression",
                "Value €M",
            ]:
                if col in wing_df.columns:
                    wing_df[col] = pd.to_numeric(wing_df[col], errors="coerce")

            for col, filename, title in [
                ("Winger Recruitment Score", "winger_shortlist_model_score.png", "Winger Shortlist — Model Score"),
                ("Creation", "winger_shortlist_creation.png", "Winger Shortlist — Creation"),
                ("1v1 / Progression", "winger_shortlist_1v1.png", "Winger Shortlist — 1v1 / Progression"),
                ("Goal Threat", "winger_shortlist_goal_threat.png", "Winger Shortlist — Goal Threat"),
                ("Value €M", "winger_shortlist_value.png", "Winger Shortlist — Market Value"),
            ]:
                if col in wing_df.columns and "Player" in wing_df.columns:
                    path = VIS_DIR / filename
                    save_bar_chart(
                        wing_df.sort_values(col, ascending=False, na_position="last"),
                        x_col="Player",
                        y_col=col,
                        title=title,
                        ylabel=col,
                        out_path=path,
                    )
                    saved_paths.append(path)

    # ----------------------------------------------------------
    # 6) Creative 10 peers
    # ----------------------------------------------------------
    if not recruitment.empty:
        cam_df = recruitment[recruitment["Role"].astype(str).eq("Creative 10")].copy()
        if not cam_df.empty:
            cam_names = [
                "Lazar Samardžić",
                "Francisco Conceição",
                "Martin Baturina",
                "Mikkel Damsgaard",
                "C. De Ketelaere",
            ]
            cam_rows = []
            for name in cam_names:
                match = find_match(cam_df, name, ["Player"])
                if not match.empty:
                    cam_rows.append(match.iloc[0])

            if cam_rows:
                cam_comp = pd.DataFrame(cam_rows).copy()
                for col in ["Target Score", "Recruitment Suitability Score", "Value Score", "Current Value €M"]:
                    if col in cam_comp.columns:
                        cam_comp[col] = pd.to_numeric(cam_comp[col], errors="coerce")

                for col, filename, title in [
                    ("Target Score", "creative10_shortlist_target_score.png", "Creative 10 Shortlist — Target Score"),
                    ("Recruitment Suitability Score", "creative10_shortlist_suitability.png", "Creative 10 Shortlist — Suitability"),
                    ("Value Score", "creative10_shortlist_value_score.png", "Creative 10 Shortlist — Value Score"),
                    ("Current Value €M", "creative10_shortlist_market_value.png", "Creative 10 Shortlist — Market Value"),
                ]:
                    if col in cam_comp.columns and "Player" in cam_comp.columns:
                        path = VIS_DIR / filename
                        save_bar_chart(
                            cam_comp.sort_values(col, ascending=False, na_position="last"),
                            x_col="Player",
                            y_col=col,
                            title=title,
                            ylabel=col,
                            out_path=path,
                        )
                        saved_paths.append(path)

    # ----------------------------------------------------------
    # 7) RB / fullback comparison using final shortlist rows
    # ----------------------------------------------------------
    rb_rows = final[
        final["Player"].isin(["Bart van Rooij", "Filip Lissah", "Rayan Aït-Nouri"])
    ].copy()

    if not rb_rows.empty:
        rb_rows["Project Fee €M"] = pd.to_numeric(rb_rows["Project Fee €M"], errors="coerce")
        rb_rows["Model Role Score"] = pd.to_numeric(rb_rows["Model Role Score"], errors="coerce")
        rb_rows["Suitability"] = pd.to_numeric(rb_rows["Suitability"], errors="coerce")

        for col, filename, title in [
            ("Project Fee €M", "fullback_shortlist_fees.png", "Fullback Targets — Project Fees"),
            ("Model Role Score", "fullback_shortlist_model_score.png", "Fullback Targets — Model Role Score"),
            ("Suitability", "fullback_shortlist_suitability.png", "Fullback Targets — Suitability"),
        ]:
            path = VIS_DIR / filename
            save_bar_chart(
                rb_rows.sort_values(col, ascending=False, na_position="last"),
                x_col="Player",
                y_col=col,
                title=title,
                ylabel=col,
                out_path=path,
            )
            saved_paths.append(path)

    # ----------------------------------------------------------
    # 8) Manual performance charts for Nwaiwu and Conceição
    # ----------------------------------------------------------
    path = VIS_DIR / "nwaiwu_manual_cb_profile.png"
    save_metric_comparison_chart(
        ["LineBreaks", "Pass %", "Duels", "Aerial %", "Interceptions", "Recoveries"],
        [
            nwaiwu_manual.get("Line-Breaking Passes/90"),
            nwaiwu_manual.get("Pass %"),
            nwaiwu_manual.get("Duel Win %"),
            nwaiwu_manual.get("Aerial Win %"),
            nwaiwu_manual.get("Interceptions/90"),
            nwaiwu_manual.get("Recoveries/90"),
        ],
        title="Nwaiwu — Manual FotMob Sample",
        ylabel="Metric Value",
        out_path=path,
    )
    saved_paths.append(path)

    conceicao_manual = MANUAL_METRICS.get("Francisco Conceição", {})
    path = VIS_DIR / "conceicao_manual_attacking_profile.png"
    save_metric_comparison_chart(
        ["xG", "xA", "Shots", "Chances", "Crosses", "Dribbles", "TouchesBox"],
        [
            conceicao_manual.get("xG/90"),
            conceicao_manual.get("xA/90"),
            conceicao_manual.get("Shots/90"),
            conceicao_manual.get("Chances Created/90"),
            conceicao_manual.get("Successful Crosses/90"),
            conceicao_manual.get("Successful Dribbles/90"),
            conceicao_manual.get("Touches In Box/90"),
        ],
        title="Conceição — Manual FotMob Sample",
        ylabel="Per 90",
        out_path=path,
    )
    saved_paths.append(path)

    return saved_paths


# ======================================================================================
# PRINTS
# ======================================================================================

def print_core(df: pd.DataFrame) -> None:
    subsection("CORE TRANSFER PLAN")
    core = df[df["Status"].eq("BUY")].copy()

    print(f"  {'Player':<22} {'Pos':<9} {'Fee':>8} {'Score':>7} {'Upgrade':>7} {'Suit':>7}")
    print("  " + "-" * 67)

    for _, row in core.iterrows():
        print(
            f"  {short_text(row.get('Player'), 22):<22} "
            f"{short_text(row.get('Position'), 9):<9} "
            f"{fmt_money(row.get('Project Fee €M')):>8} "
            f"{fmt_num(row.get('Model Role Score')):>7} "
            f"{fmt_num(row.get('Upgrade')):>7} "
            f"{fmt_num(row.get('Suitability')):>7}"
        )

    total = pd.to_numeric(core["Project Fee €M"], errors="coerce").sum()
    print()
    print(f"  Core spend: {fmt_money(total)}")


def print_optional(df: pd.DataFrame) -> None:
    subsection("OPTIONAL FINAL ADDITIONS")
    optional = df[df["Status"].eq("SHORTLIST")].copy()

    for _, row in optional.iterrows():
        print()
        print(f"  {row['Player']} — {row['Position']} — {fmt_money(row['Project Fee €M'])}")
        print(f"    Role: {row['Role']}")

        if row["Player"] == "Chibuike Nwaiwu":
            print(
                f"    CB model {fmt_num(row.get('Young CB Score'))} | "
                f"Pass {fmt_num(row.get('Pass %'), 1)}% | "
                f"Line breaks {fmt_num(row.get('Line-Breaking Passes/90'))}/90"
            )
            print(
                f"    Duels {fmt_num(row.get('Duel Win %'), 1)}% | "
                f"Aerials {fmt_num(row.get('Aerial Win %'), 1)}% | "
                f"Interceptions {fmt_num(row.get('Interceptions/90'))}/90 | "
                f"Recoveries {fmt_num(row.get('Recoveries/90'))}/90"
            )

        elif row["Player"] == "Francisco Conceição":
            print(
                f"    Winger score {fmt_num(row.get('Winger Score'))} | "
                f"Creation {fmt_num(row.get('Creation'))} | "
                f"1v1 {fmt_num(row.get('1v1 / Progression'))}"
            )
            print(
                f"    xG {fmt_num(row.get('xG/90'))}/90 | "
                f"xA {fmt_num(row.get('xA/90'))}/90 | "
                f"Shots {fmt_num(row.get('Shots/90'))}/90 | "
                f"Chances {fmt_num(row.get('Chances Created/90'))}/90"
            )
            print(
                f"    Line breaks {fmt_num(row.get('Line-Breaking Passes/90'))}/90 | "
                f"Touches in box {fmt_num(row.get('Touches In Box/90'))}/90"
            )


def print_budget(df: pd.DataFrame) -> None:
    subsection("BUDGET")

    starting_budget = 150.0
    sales = 113.5
    core_spend = pd.to_numeric(df.loc[df["Status"].eq("BUY"), "Project Fee €M"], errors="coerce").sum()
    optional_spend = pd.to_numeric(df.loc[df["Status"].eq("SHORTLIST"), "Project Fee €M"], errors="coerce").sum()

    total_available = starting_budget + sales
    after_core = total_available - core_spend
    after_all = total_available - core_spend - optional_spend

    print(f"  Starting budget      {fmt_money(starting_budget)}")
    print(f"  Confirmed sales      {fmt_money(sales)}")
    print(f"  Total available      {fmt_money(total_available)}")
    print()
    print(f"  Core six             {fmt_money(core_spend)}")
    print(f"  Remaining after core {fmt_money(after_core)}")
    print()
    print(f"  + Nwaiwu             {fmt_money(SHORTLIST['Chibuike Nwaiwu']['project_fee'])}")
    print(f"  + Conceição          {fmt_money(SHORTLIST['Francisco Conceição']['project_fee'])}")
    print(f"  Remaining after all  {fmt_money(after_all)}")


def print_summary(df: pd.DataFrame) -> None:
    subsection("FINAL SHORTLIST")
    print(f"  {'Player':<22} {'Position':<10} {'Fee':>8} {'Priority':<10} Status")
    print("  " + "-" * 73)

    for _, row in df.iterrows():
        print(
            f"  {short_text(row.get('Player'), 22):<22} "
            f"{short_text(row.get('Position'), 10):<10} "
            f"{fmt_money(row.get('Project Fee €M')):>8} "
            f"{short_text(row.get('Priority'), 10):<10} "
            f"{row.get('Status')}"
        )


def print_visuals(saved_paths: list[Path]) -> None:
    subsection("VISUALIZATIONS")
    if not saved_paths:
        print("  No charts saved.")
        return

    for path in saved_paths:
        print(f"  {path}")


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:
    section("NEWCASTLE FINAL SHORTLIST ANALYSIS")

    recruitment = safe_read(RECRUITMENT_FILE)
    priority = safe_read(PRIORITY_FILE)
    young_cb = safe_read(YOUNG_CB_FILE)
    winger = safe_read(WINGER_FILE)

    final = build_final_shortlist(
        recruitment=recruitment,
        priority=priority,
        young_cb=young_cb,
        winger=winger,
    )
    manual_metrics = build_manual_metrics_file()

    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    final.to_csv(OUT, index=False)
    manual_metrics.to_csv(OUT_METRICS, index=False)

    saved_paths = build_visualizations(
        recruitment=recruitment,
        young_cb=young_cb,
        winger=winger,
        final=final,
    )

    print_core(final)
    print_optional(final)
    print_budget(final)
    print_summary(final)
    print_visuals(saved_paths)

    section("FILES SAVED")
    print(f"  Final shortlist -> {OUT}")
    print(f"  Manual metrics  -> {OUT_METRICS}")
    print(f"  Visualizations  -> {VIS_DIR}")


if __name__ == "__main__":
    main()
