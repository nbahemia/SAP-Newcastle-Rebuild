from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS / SETTINGS
# ======================================================================================

ROOT = Path(__file__).resolve().parents[4]
BSD_DIR = ROOT / "data" / "processed" / "bsd"
SHORTLIST_DIR = BSD_DIR / "shortlist"

OUT = SHORTLIST_DIR / "squad_depth_analysis.csv"

NEWCASTLE_BENCHMARKS_FILE = SHORTLIST_DIR / "newcastle_jaissle_role_benchmarks.csv"
NEWCASTLE_OPTIMAL_ROLES_FILE = SHORTLIST_DIR / "newcastle_jaissle_optimal_roles.csv"

MAX_TARGET_VALUE_EUR_M = 100.0

ROLE_ORDER = [
    "High-Line CB",
    "Left WB / FB",
    "Right WB / FB",
    "Holding Controller",
    "Advanced / Workhorse 8",
    "Creative 10",
    "Wide Creator",
    "Vertical / Box Winger",
    "Hybrid Winger",
    "Focal Striker",
]


# ======================================================================================
# CURRENT NEWCASTLE ROLE MAP
#
# This is intentionally explicit.
#
# The squad-depth report should describe the squad we actually believe Newcastle have,
# rather than simply sorting every player by the highest score found in a source file.
#
# Important current interpretation:
#   RB1        = Amar Dedic
#   Controller = Lewis Miley
#   No. 8      = Aladji Bamba / Joelinton, with Ramsey as rotation
#   Steur      = developmental midfield option
# ======================================================================================

CURRENT_ROLE_MAP: dict[str, list[str]] = {
    "Sweeper / Build-Up GK": [
        "Lukáš Horníček",
        "Nick Pope",
        "Robin Roefs",       # harmless if absent; score remains blank
    ],
    "High-Line CB": [
        "Fabian Schär",
        "Sven Botman",
        "Malick Thiaw",
    ],
    "Left WB / FB": [
        "Lewis Hall",
        "Dan Burn",
    ],
    "Right WB / FB": [
        "Amar Dedić",
        "Tino Livramento",
        "Lewis Miley",
    ],
    "Holding Controller": [
        "Lewis Miley",
        "Nico González",
        "Sean Steur",
    ],
    "Advanced / Workhorse 8": [
        "Aladji Bamba",
        "Joelinton",
        "Jacob Ramsey",
    ],
    "Creative 10": [
        "Yoane Wissa",
        "Jacob Ramsey",
        "Bazoumana Touré",
    ],
    "Wide Creator": [
        "Bazoumana Touré",
        "Matias Fernandez-Pardo",
        "Anthony Elanga",
    ],
    "Vertical / Box Winger": [
        "Matias Fernandez-Pardo",
        "Bazoumana Touré",
        "Harvey Barnes",
    ],
    "Hybrid Winger": [
        "Bazoumana Touré",
        "Matias Fernandez-Pardo",
        "Anthony Elanga",
    ],
    "Focal Striker": [
        "Yoane Wissa",
        "William Osula",
        "Nick Woltemade",
    ],
}


# ======================================================================================
# ROLE SOURCE CONFIG
# ======================================================================================

ROLE_SOURCE_CONFIG = {
    "High-Line CB": {
        "file": "center_back_jaissle_detailed.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["jaissle_cb_score", "Jaissle CB", "Score"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Left WB / FB": {
        "file": "left_wingback_recruitment_progressive.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["jaissle_fb_score", "_jaissle_fb_score", "Jaissle FB Score"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
        "rebuild_left_fb_score": True,
    },
    "Right WB / FB": {
        "file": "right_back_jaissle_detailed.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["jaissle_fb_score", "Jaissle RB", "Score"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Holding Controller": {
        "file": "holding_controller_jaissle_detailed.csv",
        "player_cols": ["Player", "player_name"],
        "score_cols": ["Jaissle Controller", "jaissle_controller_score", "Score"],
        "club_cols": ["Club", "club_name"],
        "value_cols": ["Value €M", "market_value_eur_m"],
    },
    "Advanced / Workhorse 8": {
        "file": "workhorse_cm_jaissle_detailed.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["jaissle_workhorse_score", "Jaissle Workhorse", "Score"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Creative 10": {
        "file": "creative_10_jaissle_detailed.csv",
        "player_cols": ["Player", "player_name"],
        "score_cols": ["CAM Fit", "creative_10_score", "Score"],
        "club_cols": ["Club", "club_name"],
        "value_cols": ["Value €M", "market_value_eur_m"],
    },
    "Wide Creator": {
        "file": "winger_jaissle_archetypes.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["wide_creator_score", "Wide Creator"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Vertical / Box Winger": {
        "file": "winger_jaissle_archetypes.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["vertical_box_score", "Vertical / Box"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Hybrid Winger": {
        "file": "winger_jaissle_archetypes.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["hybrid_score", "Hybrid"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
    },
    "Focal Striker": {
        "file": "focal_striker_jaissle_detailed.csv",
        "player_cols": ["player_name", "Player"],
        "score_cols": ["jaissle_striker_score", "Jaissle Striker", "Score"],
        "club_cols": ["club_name", "Club"],
        "value_cols": ["market_value_eur_m", "Value €M"],
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


def name_aliases(value) -> set[str]:
    normalized = normalize_name(value)
    if not normalized:
        return set()

    parts = normalized.split()
    aliases = {normalized}

    if len(parts) >= 2:
        surname = " ".join(parts[1:])
        aliases.add(f"{parts[0][0]} {surname}")
        aliases.add(surname)

    return aliases


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {
        normalize_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]

    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] Could not read {path.name}: {exc}")
        return pd.DataFrame()


def numeric_value(value) -> float:
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


def player_match_mask(series: pd.Series, player: str) -> pd.Series:
    wanted = name_aliases(player)

    return series.map(
        lambda value:
        bool(name_aliases(value) & wanted)
    )


def is_newcastle_club(value) -> bool:
    return "newcastle" in normalize_name(value)


def is_current_reference_row(row: pd.Series) -> bool:
    for col in row.index:
        col_key = normalize_name(col)

        if (
            "benchmark" in col_key
            or "sample type" in col_key
            or "status" == col_key
        ):
            value = normalize_name(row.get(col, ""))
            if (
                "current" in value
                or "reference" in value
                or "newcastle" in value
            ):
                return True

    return False


# ======================================================================================
# COMMON ROLE SCORE PREPARATION
# ======================================================================================

def prepare_role_source(role: str) -> pd.DataFrame:
    config = ROLE_SOURCE_CONFIG[role]
    path = SHORTLIST_DIR / config["file"]

    df = safe_read_csv(path)
    if df.empty:
        return df

    player_col = find_col(df, config["player_cols"])
    club_col = find_col(df, config["club_cols"])
    value_col = find_col(df, config["value_cols"])
    score_col = find_col(df, config["score_cols"])

    out = df.copy()

    # The left-back file historically did not always save the final Jaissle FB
    # composite. Rebuild it from the same 40/30/30 structure when necessary.
    if config.get("rebuild_left_fb_score"):
        role_fit_col = find_col(out, ["role_fit", "WB Fit", "Role Fit"])
        creation_col = find_col(
            out,
            ["attacking_creation_index", "Creation"],
        )
        physical_col = find_col(
            out,
            ["physical_activity_index", "Physical"],
        )

        if (
            role_fit_col is not None
            and creation_col is not None
            and physical_col is not None
        ):
            rebuilt = (
                0.40 * pd.to_numeric(out[role_fit_col], errors="coerce")
                + 0.30 * pd.to_numeric(out[creation_col], errors="coerce")
                + 0.30 * pd.to_numeric(out[physical_col], errors="coerce")
            )

            if score_col is None:
                out["_depth_role_score"] = rebuilt
                score_col = "_depth_role_score"
            else:
                existing = pd.to_numeric(out[score_col], errors="coerce")
                out["_depth_role_score"] = existing.fillna(rebuilt)
                score_col = "_depth_role_score"

    if player_col is None or score_col is None:
        return pd.DataFrame()

    standardized = pd.DataFrame({
        "Player": out[player_col].fillna("").astype(str),
        "Score": pd.to_numeric(out[score_col], errors="coerce"),
        "Club": (
            out[club_col].fillna("").astype(str)
            if club_col is not None
            else ""
        ),
        "Value €M": (
            pd.to_numeric(out[value_col], errors="coerce")
            if value_col is not None
            else np.nan
        ),
    })

    # Preserve a row-level signal for CURRENT / REFERENCE entries.
    standardized["_reference_row"] = out.apply(
        is_current_reference_row,
        axis=1,
    ).to_numpy()

    return standardized


# ======================================================================================
# NEWCASTLE PLAYER SCORE LOOKUP
#
# Search Newcastle-specific benchmark files first, then the role-specific source.
# ======================================================================================

def score_from_newcastle_benchmark_files(
    player: str,
    role: str,
) -> float | None:
    wanted = name_aliases(player)

    for path in [
        NEWCASTLE_BENCHMARKS_FILE,
        NEWCASTLE_OPTIMAL_ROLES_FILE,
    ]:
        df = safe_read_csv(path)
        if df.empty:
            continue

        player_col = find_col(
            df,
            ["player_name", "Player", "player", "name"],
        )
        if player_col is None:
            continue

        matched = df[
            df[player_col].map(
                lambda value:
                bool(name_aliases(value) & wanted)
            )
        ].copy()

        if matched.empty:
            continue

        # Prefer a row for the requested tactical role if the file stores one.
        role_col = find_col(
            matched,
            ["role_key", "Role", "role", "Optimal Role", "Best Role"],
        )
        if role_col is not None:
            role_tokens = {
                normalize_name(role),
            }

            role_aliases = {
                "Right WB / FB": {
                    "right wb fb",
                    "vertical full back wingback",
                    "right back",
                },
                "Holding Controller": {
                    "holding controller",
                    "controller",
                    "deep controller",
                },
                "Advanced / Workhorse 8": {
                    "advanced workhorse 8",
                    "advanced pivot",
                    "workhorse",
                },
                "High-Line CB": {
                    "high line cb",
                    "center back",
                    "centre back",
                },
                "Left WB / FB": {
                    "left wb fb",
                    "vertical full back wingback",
                    "left back",
                },
            }
            role_tokens.update(
                role_aliases.get(role, set())
            )

            normalized_roles = (
                matched[role_col]
                .fillna("")
                .astype(str)
                .map(normalize_name)
            )
            role_mask = normalized_roles.isin(role_tokens)

            if role_mask.any():
                matched = matched.loc[role_mask].copy()

        score_col = find_col(
            matched,
            [
                "jaissle_role_score",
                "Jaissle Role Score",
                "Score",
                "Role Score",
                "Fit",
            ],
        )
        if score_col is None:
            continue

        scores = pd.to_numeric(
            matched[score_col],
            errors="coerce",
        ).dropna()

        if not scores.empty:
            return float(scores.max())

    return None


def score_from_role_file(
    player: str,
    role: str,
) -> float | None:
    df = prepare_role_source(role)
    if df.empty:
        return None

    matched = df[
        player_match_mask(
            df["Player"],
            player,
        )
    ].copy()

    if matched.empty:
        return None

    scores = pd.to_numeric(
        matched["Score"],
        errors="coerce",
    ).dropna()

    if scores.empty:
        return None

    return float(scores.max())


def current_player_score(
    player: str,
    role: str,
) -> float:
    # Newcastle-specific bridge is most reliable for current squad players.
    score = score_from_newcastle_benchmark_files(
        player,
        role,
    )
    if score is not None:
        return score

    score = score_from_role_file(
        player,
        role,
    )
    if score is not None:
        return score

    return np.nan


# ======================================================================================
# KNOWN SCORE CORRECTIONS / ROLE-SPECIFIC SOURCES
#
# These are not invented ratings. They are the already-established Newcastle role scores
# from the project's historical benchmark bridge / dedicated controller model.
#
# Keeping them here prevents a stale source CSV from silently moving the depth chart back
# to an older interpretation.
# ======================================================================================

CURRENT_SCORE_OVERRIDES: dict[tuple[str, str], float] = {
    ("Right WB / FB", normalize_name("Amar Dedić")): 53.881443,
    ("Holding Controller", normalize_name("Lewis Miley")): 57.09,
    ("Advanced / Workhorse 8", normalize_name("Aladji Bamba")): 58.530327,
    ("Advanced / Workhorse 8", normalize_name("Joelinton")): 53.788676,
    ("Advanced / Workhorse 8", normalize_name("Jacob Ramsey")): 49.278852,
    ("Advanced / Workhorse 8", normalize_name("Sean Steur")): 45.786994,
}


def resolved_current_score(
    player: str,
    role: str,
) -> float:
    override = CURRENT_SCORE_OVERRIDES.get(
        (role, normalize_name(player))
    )
    if override is not None:
        return float(override)

    return current_player_score(
        player,
        role,
    )


# ======================================================================================
# EXTERNAL TARGET LOOKUP
# ======================================================================================

def best_external_target(
    role: str,
) -> tuple[str, float, float]:
    df = prepare_role_source(role)

    if df.empty:
        return "", np.nan, np.nan

    work = df.copy()

    # Remove Newcastle/current reference rows.
    external_mask = ~work["Club"].map(is_newcastle_club)
    external_mask &= ~work["_reference_row"].fillna(False)

    # Remove current Newcastle players even when historical club metadata is attached.
    current_aliases: set[str] = set()
    for players in CURRENT_ROLE_MAP.values():
        for player in players:
            current_aliases.update(
                name_aliases(player)
            )

    external_mask &= ~work["Player"].map(
        lambda value:
        bool(name_aliases(value) & current_aliases)
    )

    work = work.loc[external_mask].copy()

    value = pd.to_numeric(
        work["Value €M"],
        errors="coerce",
    )
    affordable = (
        value.isna()
        | (value <= MAX_TARGET_VALUE_EUR_M)
    )
    work = work.loc[affordable].copy()

    work["Score"] = pd.to_numeric(
        work["Score"],
        errors="coerce",
    )
    work = work.dropna(
        subset=["Score"],
    )

    if work.empty:
        return "", np.nan, np.nan

    work = work.sort_values(
        ["Score", "Value €M"],
        ascending=[False, True],
        na_position="last",
    )

    best = work.iloc[0]

    return (
        str(best["Player"]),
        float(best["Score"]),
        numeric_value(best["Value €M"]),
    )


# ======================================================================================
# DEPTH DIAGNOSIS
# ======================================================================================

def depth_risk(
    role: str,
    scores: list[float],
    external_score: float,
) -> str:
    valid_scores = [
        float(score)
        for score in scores
        if pd.notna(score)
    ]

    starter = valid_scores[0] if valid_scores else np.nan
    second = valid_scores[1] if len(valid_scores) >= 2 else np.nan

    gap = (
        external_score - starter
        if pd.notna(external_score) and pd.notna(starter)
        else np.nan
    )

    if role == "Sweeper / Build-Up GK":
        return "YELLOW — starter trusted, model confidence low"

    if role == "High-Line CB":
        if len(valid_scores) >= 3:
            return "GREEN — strong immediate depth; succession is the main question"
        return "YELLOW — adequate starters, limited depth"

    if role == "Left WB / FB":
        if len(valid_scores) >= 2:
            return "GREEN — multiple tactical options"
        if pd.notna(gap) and gap >= 4:
            return "YELLOW — upgrade / complementary profile opportunity"
        return "YELLOW — starter quality, shallow specialist depth"

    if role == "Right WB / FB":
        if len(valid_scores) >= 2 and pd.notna(second):
            if pd.notna(gap) and gap >= 6:
                return "YELLOW — starter established; competition can still improve role"
            return "GREEN — starter + internal cover"
        return "YELLOW — starter established, shallow specialist depth"

    if role == "Holding Controller":
        if len(valid_scores) >= 2:
            return "GREEN — Miley leads role with viable internal alternatives"
        return "YELLOW — controller depth uncertain"

    if role == "Advanced / Workhorse 8":
        if len(valid_scores) >= 3:
            return "GREEN — strong internal midfield depth"
        return "GREEN — starter + viable internal alternative"

    if role == "Creative 10":
        if pd.notna(gap) and gap >= 7:
            return "ORANGE — specialist creator upgrade available"
        return "YELLOW — internal options, specialist quality can improve"

    if role in {
        "Wide Creator",
        "Vertical / Box Winger",
        "Hybrid Winger",
    }:
        if pd.notna(gap) and gap >= 6:
            return "ORANGE — specialist winger quality gap"
        return "GREEN — adequate internal archetype coverage"

    if role == "Focal Striker":
        if pd.notna(gap) and gap >= 10:
            return "RED — major starter quality gap"
        if pd.notna(gap) and gap >= 5:
            return "ORANGE — meaningful striker upgrade available"
        return "GREEN — adequate striker quality / depth"

    return "GRAY — review manually"


# ======================================================================================
# BUILD DEPTH TABLE
# ======================================================================================

def build_depth_table() -> pd.DataFrame:
    rows: list[dict] = []

    for role in ROLE_ORDER:
        players = CURRENT_ROLE_MAP.get(
            role,
            [],
        )[:3]

        while len(players) < 3:
            players.append("")

        scores = [
            (
                resolved_current_score(
                    player,
                    role,
                )
                if player
                else np.nan
            )
            for player in players
        ]

        external_player, external_score, external_value = (
            best_external_target(role)
        )

        starter_score = scores[0]
        upgrade_gap = (
            external_score - starter_score
            if (
                pd.notna(external_score)
                and pd.notna(starter_score)
            )
            else np.nan
        )

        rows.append({
            "Role": role,
            "Choice 1": players[0],
            "Score 1": scores[0],
            "Choice 2": players[1],
            "Score 2": scores[1],
            "Choice 3": players[2],
            "Score 3": scores[2],
            "Best External": external_player,
            "External Score": external_score,
            "External Value €M": external_value,
            "Upgrade Gap": upgrade_gap,
            "Depth Risk": depth_risk(
                role,
                scores,
                external_score,
            ),
        })

    return pd.DataFrame(rows)


# ======================================================================================
# TERMINAL OUTPUT
# ======================================================================================

def fmt_score(value) -> str:
    if pd.isna(value):
        return "—"

    return f"{float(value):.2f}"


def fmt_value(value) -> str:
    if pd.isna(value):
        return "—"

    return f"€{float(value):.1f}m"


def split_status(value) -> tuple[str, str]:
    text = str(value or "").strip()

    if "—" in text:
        status, note = text.split("—", 1)
        return status.strip(), note.strip()

    return text, ""


def printable_player(value) -> bool:
    text = str(value or "").strip()

    return bool(
        text
        and text.lower() not in {"nan", "none", "<na>"}
    )


def print_depth_report(result: pd.DataFrame) -> None:
    width = 96

    print()
    print("=" * width)
    print("NEWCASTLE SQUAD DEPTH / QUALITY DIAGNOSIS")
    print("=" * width)

    for _, row in result.iterrows():
        role = str(row["Role"])

        print()
        print(role)
        print("-" * min(len(role), 55))

        for idx in range(1, 4):
            player = row.get(
                f"Choice {idx}",
                "",
            )
            score = row.get(
                f"Score {idx}",
                np.nan,
            )

            if not printable_player(player):
                continue

            label = (
                "Primary"
                if idx == 1
                else (
                    "Rotation"
                    if idx == 2
                    else "Depth"
                )
            )

            print(
                f"  {label:<9} "
                f"{str(player):<26} "
                f"{fmt_score(score):>6}"
            )

        external = row.get(
            "Best External",
            "",
        )
        external_score = row.get(
            "External Score",
            np.nan,
        )
        external_value = row.get(
            "External Value €M",
            np.nan,
        )
        gap = row.get(
            "Upgrade Gap",
            np.nan,
        )

        if printable_player(external):
            print(
                f"  External  "
                f"{str(external):<26} "
                f"{fmt_score(external_score):>6}"
                f"   Gap {fmt_score(gap):>6}"
                f"   {fmt_value(external_value):>8}"
            )

        status, note = split_status(
            row.get(
                "Depth Risk",
                "",
            )
        )

        print(
            f"  Status    {status}"
        )

        if note:
            print(
                f"  Note      {note}"
            )

    print()
    print("=" * width)
    print(
        f"Maximum external target value: "
        f"€{MAX_TARGET_VALUE_EUR_M:.0f}m"
    )
    print(
        f"Saved: {OUT}"
    )
    print("=" * width)


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:
    result = build_depth_table()

    # Round saved numerical output while retaining NaN values.
    numeric_cols = [
        "Score 1",
        "Score 2",
        "Score 3",
        "External Score",
        "External Value €M",
        "Upgrade Gap",
    ]

    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            ).round(2)

    SHORTLIST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT,
        index=False,
    )

    print_depth_report(
        result
    )


if __name__ == "__main__":
    main()
