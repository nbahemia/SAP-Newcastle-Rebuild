from pathlib import Path
import argparse
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import PyPizza, Radar


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED = PROJECT_ROOT / "data" / "processed" / "statsbomb"
ROLE_ANALYSIS = PROCESSED / "role_analysis"

VISUALIZATIONS = PROJECT_ROOT / "visualizations" / "player_profiles"

PERCENTILES_FILE = PROCESSED / "player_percentiles.parquet"


# ---------------------------------------------------------
# Role Configuration
# ---------------------------------------------------------

ROLE_CONFIG = {
    "striker": {
        "title": "Jaissle Line-Leading No. 9",
        "shortlist_file": ROLE_ANALYSIS / "striker_scouting_shortlist.parquet",
        "score_column": "striker_fit_score",
        "metrics": [
            "xg_percentile",
            "shots_percentile",
            "shots_in_box_percentile",
            "goals_percentile",
            "attacking_third_pressures_percentile",
            "duels_percentile",
        ],
        "raw_metrics": [
            "xg_per90",
            "shots_per90",
            "shots_in_box_per90",
            "goals_per90",
            "attacking_third_pressures_per90",
            "duels_per90",
        ],
        "labels": [
            "xG / 90",
            "Shots / 90",
            "Shots in Box / 90",
            "Goals / 90",
            "Att. 3rd Pressures / 90",
            "Duels / 90",
        ],
    },

    "fullback": {
        "title": "Vertical Two-Way Fullback",
        "shortlist_file": ROLE_ANALYSIS / "left_back_scouting_shortlist.parquet",
        "score_column": "fullback_fit_score",
        "metrics": [
            "progressive_carries_percentile",
            "carries_into_final_third_percentile",
            "progressive_passes_percentile",
            "passes_into_final_third_percentile",
            "recoveries_percentile",
            "duels_percentile",
        ],
        "raw_metrics": [
            "progressive_carries_per90",
            "carries_into_final_third_per90",
            "progressive_passes_per90",
            "passes_into_final_third_per90",
            "recoveries_per90",
            "duels_per90",
        ],
        "labels": [
            "Progressive Carries / 90",
            "Carries into Final 3rd / 90",
            "Progressive Passes / 90",
            "Passes into Final 3rd / 90",
            "Recoveries / 90",
            "Duels / 90",
        ],
    },

    "controller": {
        "title": "Controller / Progressive 6",
        "shortlist_file": ROLE_ANALYSIS / "controller_scouting_shortlist.parquet",
        "score_column": "controller_fit_score",
        "metrics": [
            "progressive_passes_percentile",
            "passes_into_final_third_percentile",
            "successful_passes_percentile",
            "progressive_carries_percentile",
            "recoveries_percentile",
            "interceptions_percentile",
        ],
        "raw_metrics": [
            "progressive_passes_per90",
            "passes_into_final_third_per90",
            "successful_passes_per90",
            "progressive_carries_per90",
            "recoveries_per90",
            "interceptions_per90",
        ],
        "labels": [
            "Progressive Passes / 90",
            "Passes into Final 3rd / 90",
            "Successful Passes / 90",
            "Progressive Carries / 90",
            "Recoveries / 90",
            "Interceptions / 90",
        ],
    },
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_filename(value: str) -> str:
    value = value.strip().lower()
    cleaned = []

    for char in value:
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_"}:
            cleaned.append("_")

    result = "".join(cleaned)

    while "__" in result:
        result = result.replace("__", "_")

    return result.strip("_")


def wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(
        textwrap.wrap(
            label,
            width=width,
        )
    )


def validate_columns(
    df: pd.DataFrame,
    columns: list[str],
    context: str,
):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns for {context}:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def get_player_row(
    players: pd.DataFrame,
    player_name: str,
) -> pd.Series:

    exact = players[
        players["player"].str.casefold()
        == player_name.casefold()
    ]

    if len(exact) == 1:
        return exact.iloc[0]

    contains = players[
        players["player"]
        .str.contains(
            player_name,
            case=False,
            na=False,
            regex=False,
        )
    ]

    if len(contains) == 1:
        return contains.iloc[0]

    if len(contains) > 1:
        raise ValueError(
            f"\n'{player_name}' matched multiple players:\n"
            + "\n".join(
                f"  - {name}"
                for name in contains["player"].tolist()
            )
        )

    raise ValueError(
        f"\nCould not find player: {player_name}"
    )


def load_role_data(
    role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if role not in ROLE_CONFIG:
        raise ValueError(
            f"Unknown role '{role}'."
        )

    config = ROLE_CONFIG[role]

    if not PERCENTILES_FILE.exists():
        raise FileNotFoundError(
            f"Percentiles file not found:\n"
            f"{PERCENTILES_FILE}"
        )

    if not config["shortlist_file"].exists():
        raise FileNotFoundError(
            f"Shortlist file not found:\n"
            f"{config['shortlist_file']}"
        )

    players = pd.read_parquet(
        PERCENTILES_FILE
    )

    shortlist = pd.read_parquet(
        config["shortlist_file"]
    )

    validate_columns(
        players,
        [
            "player",
            "team",
            "minutes",
            *config["metrics"],
            *config["raw_metrics"],
        ],
        f"{role} profile data",
    )

    validate_columns(
        shortlist,
        [
            "player",
            config["score_column"],
        ],
        f"{role} shortlist",
    )

    return players, shortlist, config


def choose_players(
    players: pd.DataFrame,
    shortlist: pd.DataFrame,
    config: dict,
    top_n: int,
    names: list[str] | None,
) -> list[str]:

    if names:
        resolved = []

        for name in names:
            row = get_player_row(
                players,
                name,
            )
            resolved.append(
                row["player"]
            )

        return resolved

    return (
        shortlist
        .sort_values(
            config["score_column"],
            ascending=False,
        )
        .head(top_n)["player"]
        .tolist()
    )


# ---------------------------------------------------------
# Individual PyPizza
# ---------------------------------------------------------

def plot_player_pizza(
    row: pd.Series,
    config: dict,
    output_dir: Path,
):

    labels = [
        wrap_label(label, 17)
        for label in config["labels"]
    ]

    values = [
        int(round(float(row[column])))
        if pd.notna(row[column])
        else 0
        for column in config["metrics"]
    ]

    raw_values = [
        float(row[column])
        if pd.notna(row[column])
        else np.nan
        for column in config["raw_metrics"]
    ]

    baker = PyPizza(
        params=labels,
        background_color="#F4F1EA",
        straight_line_color="#222222",
        straight_line_lw=1,
        last_circle_color="#222222",
        last_circle_lw=1,
        other_circle_color="#B5B5B5",
        other_circle_lw=0.6,
        inner_circle_size=18,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(10, 10),
        param_location=108,
        slice_colors=["#2F6B4F"] * len(values),
        value_colors=["#F4F1EA"] * len(values),
        value_bck_colors=["#1D1D1D"] * len(values),
        kwargs_slices={
            "edgecolor": "#F4F1EA",
            "linewidth": 1.8,
        },
        kwargs_params={
            "color": "#202020",
            "fontsize": 10,
            "va": "center",
        },
        kwargs_values={
            "color": "#F4F1EA",
            "fontsize": 10,
            "bbox": {
                "edgecolor": "#1D1D1D",
                "facecolor": "#1D1D1D",
                "boxstyle": "round,pad=0.25",
                "lw": 1,
            },
        },
    )

    fig.text(
        0.5,
        0.975,
        row["player"],
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#202020",
    )

    fig.text(
        0.5,
        0.945,
        (
            f"{config['title']} | "
            f"{row['team']} | "
            f"{float(row['minutes']):.0f} min"
        ),
        ha="center",
        va="top",
        fontsize=11,
        color="#4A4A4A",
    )

    fig.text(
        0.5,
        0.918,
        "Percentiles vs observed positional peers",
        ha="center",
        va="top",
        fontsize=9,
        color="#6A6A6A",
    )

    # Raw per-90 footer
    raw_text = "   |   ".join(
        (
            f"{label.replace(chr(10), ' ')}: "
            f"{raw:.2f}"
            if pd.notna(raw)
            else f"{label.replace(chr(10), ' ')}: —"
        )
        for label, raw in zip(
            labels,
            raw_values,
        )
    )

    fig.text(
        0.5,
        0.035,
        raw_text,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        wrap=True,
    )

    output_file = (
        output_dir
        / f"{safe_filename(row['player'])}_pizza.png"
    )

    fig.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight",
        facecolor="#F4F1EA",
    )

    plt.close(fig)

    return output_file


# ---------------------------------------------------------
# Two-Player Radar Comparison
# ---------------------------------------------------------

def plot_player_comparison(
    row_a: pd.Series,
    row_b: pd.Series,
    config: dict,
    output_dir: Path,
):

    params = [
        wrap_label(label, 16)
        for label in config["labels"]
    ]

    min_range = [0] * len(params)
    max_range = [100] * len(params)

    radar = Radar(
        params=params,
        min_range=min_range,
        max_range=max_range,
        num_rings=5,
        ring_width=1,
        center_circle_radius=1,
    )

    values_a = [
        float(row_a[column])
        if pd.notna(row_a[column])
        else 0.0
        for column in config["metrics"]
    ]

    values_b = [
        float(row_b[column])
        if pd.notna(row_b[column])
        else 0.0
        for column in config["metrics"]
    ]

    fig, ax = radar.setup_axis(
        figsize=(10, 10),
        facecolor="#F4F1EA",
    )

    radar.draw_circles(
        ax=ax,
        facecolor="#E2DDD4",
        edgecolor="#C6C0B6",
    )

    radar.draw_radar_compare(
        values_a,
        values_b,
        ax=ax,
        kwargs_radar={
            "facecolor": "#2F6B4F",
            "alpha": 0.40,
            "edgecolor": "#2F6B4F",
            "linewidth": 2,
        },
        kwargs_compare={
            "facecolor": "#9A6B3F",
            "alpha": 0.35,
            "edgecolor": "#9A6B3F",
            "linewidth": 2,
        },
    )

    radar.draw_range_labels(
        ax=ax,
        fontsize=8,
        color="#666666",
    )

    radar.draw_param_labels(
        ax=ax,
        fontsize=10,
        color="#222222",
    )

    fig.text(
        0.5,
        0.965,
        f"{config['title']} Comparison",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color="#202020",
    )

    fig.text(
        0.5,
        0.935,
        "Percentiles vs observed positional peers",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )

    fig.text(
        0.28,
        0.07,
        f"{row_a['player']} — {row_a['team']}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2F6B4F",
    )

    fig.text(
        0.72,
        0.07,
        f"{row_b['player']} — {row_b['team']}",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#9A6B3F",
    )

    output_file = (
        output_dir
        / (
            "comparison_"
            f"{safe_filename(row_a['player'])}_vs_"
            f"{safe_filename(row_b['player'])}.png"
        )
    )

    fig.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight",
        facecolor="#F4F1EA",
    )

    plt.close(fig)

    return output_file


# ---------------------------------------------------------
# Terminal Summary
# ---------------------------------------------------------

def print_summary(
    players: pd.DataFrame,
    names: list[str],
    config: dict,
):

    rows = []

    for name in names:

        row = get_player_row(
            players,
            name,
        )

        rows.append(
            {
                "Player": row["player"],
                "Team": row["team"],
                "Min": int(
                    round(
                        float(row["minutes"])
                    )
                ),
                **{
                    label: round(
                        float(row[column]),
                        1,
                    )
                    for label, column in zip(
                        config["labels"],
                        config["metrics"],
                    )
                },
            }
        )

    summary = pd.DataFrame(rows)

    print()
    print(summary.to_string(index=False))


# ---------------------------------------------------------
# Role Runner
# ---------------------------------------------------------

def run_role(
    role: str,
    top_n: int,
    names: list[str] | None,
    compare: bool,
):

    players, shortlist, config = load_role_data(
        role
    )

    selected = choose_players(
        players=players,
        shortlist=shortlist,
        config=config,
        top_n=top_n,
        names=names,
    )

    output_dir = (
        VISUALIZATIONS
        / role
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print(
        f"{config['title']} PROFILES".center(90)
    )
    print("=" * 90)

    rows = []

    for player_name in selected:

        row = get_player_row(
            players,
            player_name,
        )

        rows.append(row)

        output_file = plot_player_pizza(
            row=row,
            config=config,
            output_dir=output_dir,
        )

        print(
            f"Created: {output_file.name}"
        )

    if compare:

        if len(rows) != 2:
            print(
                "\nComparison radar skipped: "
                "--compare requires exactly 2 selected players."
            )

        else:
            comparison = plot_player_comparison(
                row_a=rows[0],
                row_b=rows[1],
                config=config,
                output_dir=output_dir,
            )

            print(
                f"Created: {comparison.name}"
            )

    print_summary(
        players=players,
        names=selected,
        config=config,
    )

    print()
    print(
        f"Saved to: {output_dir}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create football scouting profiles with "
            "mplsoccer PyPizza and Radar charts."
        )
    )

    parser.add_argument(
        "--role",
        choices=[
            "striker",
            "fullback",
            "controller",
            "all",
        ],
        default="all",
        help=(
            "Role to generate. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help=(
            "Generate profiles for the top N candidates. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--players",
        nargs="+",
        default=None,
        help=(
            "Specific player names. "
            "Only valid when a single role is selected."
        ),
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Create a comparison radar when exactly "
            "two players are selected."
        ),
    )

    args = parser.parse_args()

    if args.players and args.role == "all":
        raise ValueError(
            "--players requires a specific --role."
        )

    if args.role == "all":

        for role in [
            "striker",
            "fullback",
            "controller",
        ]:
            run_role(
                role=role,
                top_n=args.top_n,
                names=None,
                compare=False,
            )

    else:

        run_role(
            role=args.role,
            top_n=args.top_n,
            names=args.players,
            compare=args.compare,
        )


if __name__ == "__main__":
    main()
