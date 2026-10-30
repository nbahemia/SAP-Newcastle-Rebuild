from __future__ import annotations

from pathlib import Path
import duckdb


# ======================================================================================
# PATHS / CONFIG
# ======================================================================================

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "bsd"

POOL_FILE = PROCESSED_DIR / "europe_2025_26_player_percentiles.parquet"
PHYSICAL_FILE = PROCESSED_DIR / "player_physical_profiles.parquet"

OUTPUT_ALL = PROCESSED_DIR / "europe_2025_26_role_shortlist.parquet"
OUTPUT_CAM = PROCESSED_DIR / "europe_2025_26_cam_shortlist.parquet"
OUTPUT_CONTROLLER = PROCESSED_DIR / "europe_2025_26_controller_shortlist.parquet"
OUTPUT_STRIKER = PROCESSED_DIR / "europe_2025_26_toney_striker_shortlist.parquet"
OUTPUT_FULLBACK = PROCESSED_DIR / "europe_2025_26_fullback_shortlist.parquet"

MIN_MINUTES = 900
MAX_AGE = 29
MAX_MARKET_VALUE_EUR = 70_000_000


ROLE_WEIGHTS = {
    "creative_cam": {
        "position_groups": {"AM", "DM/CM"},
        "metrics": {
            "xa_per90_percentile": 0.20,
            "key_passes_per90_percentile": 0.18,
            "accurate_opposition_half_passes_per90_percentile": 0.15,
            "touches_per90_percentile": 0.10,
            "progressive_carries_per90_percentile": 0.10,
            "xg_per90_percentile": 0.08,
            "shots_per90_percentile": 0.05,
            "recoveries_per90_percentile": 0.07,
            "interceptions_per90_percentile": 0.04,
            "duels_won_per90_percentile": 0.03,
        },
    },

    "controller_midfielder": {
        "position_groups": {"DM/CM"},
        "metrics": {
            "successful_passes_per90_percentile": 0.18,
            "pass_accuracy_pct_percentile": 0.08,
            "accurate_opposition_half_passes_per90_percentile": 0.18,
            "total_long_balls_per90_percentile": 0.12,
            "progressive_carries_per90_percentile": 0.10,
            "touches_per90_percentile": 0.10,
            "recoveries_per90_percentile": 0.08,
            "interceptions_per90_percentile": 0.06,
            "duels_won_per90_percentile": 0.05,
            "xa_per90_percentile": 0.05,
        },
    },

    "toney_striker": {
        "position_groups": {"ST"},
        "metrics": {
            "xg_per90_percentile": 0.15,
            "shots_per90_percentile": 0.12,
            "shots_on_target_per90_percentile": 0.08,
            "box_touches_per90_percentile": 0.13,
            "aerials_won_per90_percentile": 0.12,
            "duels_won_per90_percentile": 0.10,
            "line_breaking_passes_per90_percentile": 0.08,
            "key_passes_per90_percentile": 0.07,
            "xa_per90_percentile": 0.05,
            "successful_passes_per90_percentile": 0.05,
            "touches_per90_percentile": 0.05,
        },
    },

    "vertical_fullback": {
        "position_groups": {"FB/WB"},
        "metrics": {
            "progressive_carries_per90_percentile": 0.22,
            "progressive_carry_distance_per90_percentile": 0.16,
            "accurate_opposition_half_passes_per90_percentile": 0.15,
            "total_crosses_per90_percentile": 0.10,
            "xa_per90_percentile": 0.07,
            "recoveries_per90_percentile": 0.10,
            "interceptions_per90_percentile": 0.06,
            "duels_won_per90_percentile": 0.05,
            "aerials_won_per90_percentile": 0.03,
            "tackles_won_per90_percentile": 0.06,
        },
    },
}


def quote(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def relation_columns(con: duckdb.DuckDBPyConnection, name: str) -> set[str]:
    return {
        row[0]
        for row in con.execute(f"DESCRIBE {name}").fetchall()
    }


def score_sql(
    available: set[str],
    metrics: dict[str, float],
) -> tuple[str, str]:
    score_terms = []
    weight_terms = []

    for metric, weight in metrics.items():
        if metric not in available:
            continue

        score_terms.append(
            f"CASE WHEN p.{metric} IS NOT NULL "
            f"THEN p.{metric} * {weight} ELSE 0 END"
        )
        weight_terms.append(
            f"CASE WHEN p.{metric} IS NOT NULL "
            f"THEN {weight} ELSE 0 END"
        )

    if not score_terms:
        return "NULL::DOUBLE", "0.0"

    numerator = " + ".join(score_terms)
    denominator = " + ".join(weight_terms)

    return (
        f"CASE WHEN ({denominator}) > 0 "
        f"THEN ({numerator}) / ({denominator}) ELSE NULL END",
        denominator,
    )


def main() -> None:
    if not POOL_FILE.exists():
        raise FileNotFoundError(
            f"Missing Europe 2025/26 pool:\n{POOL_FILE}\n\n"
            "Run build_europe_2025_26_pool.py first."
        )

    con = duckdb.connect()

    try:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW player_pool AS
            SELECT *
            FROM read_parquet({quote(POOL_FILE)});
            """
        )

        pool_cols = relation_columns(
            con,
            "player_pool",
        )

        if PHYSICAL_FILE.exists():
            con.execute(
                f"""
                CREATE OR REPLACE VIEW physical_profiles AS
                SELECT *
                FROM read_parquet({quote(PHYSICAL_FILE)});
                """
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE VIEW physical_profiles AS
                SELECT
                    NULL::BIGINT AS player_id,
                    NULL::VARCHAR AS player_name,
                    NULL::DOUBLE AS age,
                    NULL::DOUBLE AS height_cm,
                    NULL::DOUBLE AS weight_kg,
                    NULL::DOUBLE AS top_speed_kmh,
                    NULL::DOUBLE AS acceleration_score,
                    NULL::DOUBLE AS market_value_eur
                WHERE FALSE;
                """
            )

        physical_cols = relation_columns(
            con,
            "physical_profiles",
        )

        if (
            "player_id" in pool_cols
            and "player_id" in physical_cols
        ):
            join_sql = (
                "TRY_CAST(p.player_id AS BIGINT) "
                "= TRY_CAST(ph.player_id AS BIGINT)"
            )
        else:
            join_sql = (
                "lower(trim(p.player_name)) "
                "= lower(trim(ph.player_name))"
            )

        for role_name, config in ROLE_WEIGHTS.items():
            role_score_sql, coverage_sql = score_sql(
                pool_cols,
                config["metrics"],
            )

            positions = ", ".join(
                "'" + position.replace("'", "''") + "'"
                for position in sorted(
                    config["position_groups"]
                )
            )

            age_expr = (
                "COALESCE(TRY_CAST(ph.age AS DOUBLE), 999)"
                if "age" in physical_cols
                else "999"
            )

            market_expr = (
                "TRY_CAST(ph.market_value_eur AS DOUBLE)"
                if "market_value_eur" in physical_cols
                else "NULL::DOUBLE"
            )

            con.execute(
                f"""
                CREATE OR REPLACE VIEW shortlist_{role_name} AS
                SELECT
                    p.*,
                    {role_score_sql} AS role_score,
                    ({coverage_sql}) * 100.0 AS role_metric_coverage_pct,
                    TRY_CAST(ph.age AS DOUBLE) AS age,
                    TRY_CAST(ph.height_cm AS DOUBLE) AS height_cm,
                    TRY_CAST(ph.weight_kg AS DOUBLE) AS weight_kg,
                    TRY_CAST(ph.top_speed_kmh AS DOUBLE) AS top_speed_kmh,
                    TRY_CAST(ph.acceleration_score AS DOUBLE) AS acceleration_score,
                    {market_expr} AS market_value_eur,
                    '{role_name}' AS recruitment_role
                FROM player_pool p
                LEFT JOIN physical_profiles ph
                    ON {join_sql}
                WHERE
                    p.position_group IN ({positions})
                    AND TRY_CAST(p.minutes AS DOUBLE) >= {MIN_MINUTES}
                    AND (
                        ph.age IS NULL
                        OR TRY_CAST(ph.age AS DOUBLE) <= {MAX_AGE}
                    )
                    AND (
                        {market_expr} IS NULL
                        OR {market_expr} <= {MAX_MARKET_VALUE_EUR}
                    );
                """
            )

        output_map = {
            "creative_cam": OUTPUT_CAM,
            "controller_midfielder": OUTPUT_CONTROLLER,
            "toney_striker": OUTPUT_STRIKER,
            "vertical_fullback": OUTPUT_FULLBACK,
        }

        for role, path in output_map.items():
            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM shortlist_{role}
                    ORDER BY
                        role_metric_coverage_pct DESC,
                        role_score DESC,
                        minutes DESC
                )
                TO {quote(path)}
                (FORMAT PARQUET);
                """
            )

        con.execute(
            f"""
            COPY (
                SELECT * FROM shortlist_creative_cam
                UNION ALL BY NAME
                SELECT * FROM shortlist_controller_midfielder
                UNION ALL BY NAME
                SELECT * FROM shortlist_toney_striker
                UNION ALL BY NAME
                SELECT * FROM shortlist_vertical_fullback
            )
            TO {quote(OUTPUT_ALL)}
            (FORMAT PARQUET);
            """
        )

        print("=" * 120)
        print("EUROPE 2025/26 — ROLE SHORTLIST".center(120))
        print("=" * 120)

        coverage = con.execute(
            """
            SELECT
                league_slug,
                COUNT(*) AS eligible_players,
                COUNT(DISTINCT team_name) AS teams
            FROM player_pool
            GROUP BY league_slug
            ORDER BY league_slug;
            """
        ).df()

        print()
        print("Pool coverage:")
        print(coverage.to_string(index=False))

        for role in ROLE_WEIGHTS:
            print()
            print("-" * 120)
            print(role.upper().replace("_", " "))
            print("-" * 120)

            cols = relation_columns(
                con,
                f"shortlist_{role}",
            )

            preferred = [
                "player_name",
                "team_name",
                "league_name",
                "position_group",
                "minutes",
                "age",
                "height_cm",
                "top_speed_kmh",
                "market_value_eur",
                "role_score",
                "role_metric_coverage_pct",
            ]

            selected = [
                col
                for col in preferred
                if col in cols
            ]

            df = con.execute(
                f"""
                SELECT {", ".join(selected)}
                FROM shortlist_{role}
                ORDER BY
                    role_metric_coverage_pct DESC,
                    role_score DESC,
                    minutes DESC
                LIMIT 25;
                """
            ).df()

            if "role_score" in df.columns:
                df["role_score"] = df["role_score"].round(1)

            if "role_metric_coverage_pct" in df.columns:
                df["role_metric_coverage_pct"] = (
                    df["role_metric_coverage_pct"].round(1)
                )

            print(df.to_string(index=False))

        print()
        print("Saved:")
        print(OUTPUT_ALL)
        print(OUTPUT_CAM)
        print(OUTPUT_CONTROLLER)
        print(OUTPUT_STRIKER)
        print(OUTPUT_FULLBACK)

    finally:
        con.close()


if __name__ == "__main__":
    main()
