from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SHORTLIST = ROOT / "data" / "processed" / "bsd" / "shortlist"
EUROPE_PARQUET = ROOT / "data" / "processed" / "bsd" / "europe_2025_26_player_season_metrics.parquet"

FILES = {
    "center_back": SHORTLIST / "center_back_jaissle_detailed.csv",
    "workhorse_cm": SHORTLIST / "workhorse_cm_jaissle_detailed.csv",
    "left_wingback": SHORTLIST / "left_wingback_recruitment_progressive.csv",
    "winger": SHORTLIST / "winger_jaissle_archetypes.csv",
    "winger_extended": SHORTLIST / "winger_jaissle_extended.csv",
}

PLAYER_ALIASES = ["player_name", "Player", "Player Name", "name"]
CLUB_ALIASES = ["club_name", "Club", "display_club_name", "team_name"]
METRIC_ALIASES = {
    "pass_accuracy_pct": [
        "pass_accuracy_pct", "pass_accuracy", "passing_accuracy_pct",
        "successful_pass_pct", "passes_completed_pct",
    ],
    "successful_passes_per90": [
        "successful_passes_per90", "passes_completed_per90", "completed_passes_per90",
    ],
    "total_passes_per90": [
        "total_passes_per90", "passes_per90", "attempted_passes_per90",
    ],
    "key_passes_per90": [
        "key_passes_per90", "key_pass_per90", "keypasses_per90",
        "key_passes_p90", "chances_created_per90",
    ],
    "accurate_crosses_per90": [
        "accurate_crosses_per90", "crosses_completed_per90",
        "successful_crosses_per90", "accurate_cross_per90",
    ],
    "box_touches_per90": [
        "box_touches_per90", "touches_in_box_per90", "touches_opp_box_per90",
        "touches_opposition_box_per90", "opposition_box_touches_per90",
        "touches_in_opposition_box_per90",
    ],
}


def norm(s: str) -> str:
    return (str(s).replace("é","e").replace("í","i").replace("á","a")
            .replace("ö","o").replace("ü","u").replace("ç","c")
            .replace("č","c").replace("ć","c").replace("š","s")
            .replace("ž","z").strip().lower())


def first_existing(df, names):
    return next((c for c in names if c in df.columns), None)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def coalesce(raw, aliases):
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    for c in aliases:
        if c in raw.columns:
            vals = num(raw[c])
            out = out.where(out.notna(), vals)
    return out


def build_lookup():
    if not EUROPE_PARQUET.exists():
        print(f"[WARN] Missing historical parquet: {EUROPE_PARQUET}")
        return pd.DataFrame()
    raw = pd.read_parquet(EUROPE_PARQUET)
    pc = first_existing(raw, PLAYER_ALIASES)
    if pc is None:
        print("[WARN] No player-name column in historical parquet")
        return pd.DataFrame()
    cc = first_existing(raw, CLUB_ALIASES)
    lk = pd.DataFrame(index=raw.index)
    lk["player_key"] = raw[pc].map(norm)
    lk["club_key"] = raw[cc].map(norm) if cc else ""
    for metric, aliases in METRIC_ALIASES.items():
        lk[metric] = coalesce(raw, aliases)
    derived = np.where(lk["total_passes_per90"].gt(0),
                       100 * lk["successful_passes_per90"] / lk["total_passes_per90"],
                       np.nan)
    lk["pass_accuracy_pct"] = lk["pass_accuracy_pct"].where(
        lk["pass_accuracy_pct"].notna(), derived)
    return lk.groupby(["player_key","club_key"], dropna=False).median(numeric_only=True).reset_index()


def matched_metrics(df, lookup):
    if lookup.empty or df.empty:
        return pd.DataFrame(index=df.index)
    pc = first_existing(df, PLAYER_ALIASES)
    if pc is None:
        return pd.DataFrame(index=df.index)
    cc = first_existing(df, CLUB_ALIASES)
    keys = pd.DataFrame({
        "player_key": df[pc].map(norm),
        "club_key": df[cc].map(norm) if cc else "",
    }, index=df.index)
    exact = keys.merge(lookup, on=["player_key","club_key"], how="left", sort=False)
    exact.index = df.index
    by_player = lookup.groupby("player_key", dropna=False).median(numeric_only=True).reset_index()
    fb = keys[["player_key"]].merge(by_player, on="player_key", how="left", sort=False)
    fb.index = df.index
    for c in METRIC_ALIASES:
        if c in exact.columns and c in fb.columns:
            exact[c] = num(exact[c]).where(num(exact[c]).notna(), num(fb[c]))
    return exact


def fill_pass_pct(df, matched):
    repaired = 0
    for col in ["Pass %", "pass_accuracy_pct", "Pass Accuracy %"]:
        if col not in df.columns:
            continue
        src = num(matched.get("pass_accuracy_pct"))
        cur = num(df[col])
        if src is not None:
            mask = cur.isna() & src.notna()
            df.loc[mask, col] = src.loc[mask]
            repaired += int(mask.sum())
        succ = first_existing(df, ["Successful Passes/90","successful_passes_per90","successful_passes_p90"])
        total = first_existing(df, ["Total Passes/90","total_passes_per90","total_passes_p90"])
        if succ and total:
            der = pd.Series(np.where(num(df[total]).gt(0), 100*num(df[succ])/num(df[total]), np.nan), index=df.index)
            mask = num(df[col]).isna() & der.notna()
            df.loc[mask, col] = der.loc[mask]
            repaired += int(mask.sum())
    return repaired


def repair_or_mark_unsupported(df, matched, output_cols, metric):
    cols = [c for c in output_cols if c in df.columns]
    if not cols:
        return 0, 0
    src = num(matched.get(metric))
    repaired = unsupported = 0
    source_signal = src is not None and src.fillna(0).abs().sum() > 0
    for col in cols:
        cur = num(df[col])
        if source_signal:
            mask = cur.fillna(0).eq(0) & src.notna() & src.ne(0)
            df.loc[mask, col] = src.loc[mask]
            repaired += int(mask.sum())
        # If after attempted repair the whole field is still zero, mark it unsupported.
        now = num(df[col])
        nonmissing = now.dropna()
        if len(nonmissing) and nonmissing.abs().sum() == 0:
            unsupported += len(nonmissing)
            df.loc[nonmissing.index, col] = np.nan
    return repaired, unsupported


def repair_file(label, path, lookup):
    if not path.exists():
        print(f"[SKIP] {label}: {path.name} missing")
        return
    df = pd.read_csv(path)
    matched = matched_metrics(df, lookup)
    repaired = unsupported = 0
    if label in {"center_back","workhorse_cm"}:
        repaired += fill_pass_pct(df, matched)
    if label == "left_wingback":
        r,u = repair_or_mark_unsupported(df, matched,
            ["Key Passes/90","key_passes_per90","key_passes_p90"], "key_passes_per90")
        repaired += r; unsupported += u
        r,u = repair_or_mark_unsupported(df, matched,
            ["Acc Crosses/90","Accurate Crosses/90","accurate_crosses_per90","accurate_crosses_p90"],
            "accurate_crosses_per90")
        repaired += r; unsupported += u
    if label in {"winger","winger_extended"}:
        r,u = repair_or_mark_unsupported(df, matched,
            ["Key Passes/90","key_passes_per90","key_passes_p90"], "key_passes_per90")
        repaired += r; unsupported += u
        r,u = repair_or_mark_unsupported(df, matched,
            ["Box Touches/90","box_touches_per90","box_touches_p90"], "box_touches_per90")
        repaired += r; unsupported += u
    df.to_csv(path, index=False)
    print(f"[OK] {label}: repaired={repaired}, marked unsupported={unsupported}")


def main():
    print("="*110)
    print("METRIC OUTPUT REPAIR")
    print("="*110)
    lookup = build_lookup()
    for label, path in FILES.items():
        repair_file(label, path, lookup)
    print("Metric repair complete.")


if __name__ == "__main__":
    main()
