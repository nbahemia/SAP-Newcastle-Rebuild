from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
ROLE_DIR = ROOT / "scripts" / "bsd" / "analyze" / "role_scouting"
ANALYZE_DIR = ROOT / "scripts" / "bsd" / "analyze"
SHORTLIST = ROOT / "data" / "processed" / "bsd" / "shortlist"
VISUALIZATIONS = ROOT / "visualizations"

STAGES = [
    {
        "label": "High-line center-back scouting",
        "scripts": ["analyze_centerback.py"],
        "outputs": ["center_back_jaissle_detailed.csv"],
    },
    {
        "label": "Left wing-back / full-back scouting",
        "scripts": ["analyze_leftback.py"],
        "outputs": ["left_wingback_recruitment_progressive.csv"],
    },
    {
        "label": "Right wing-back / full-back scouting",
        "scripts": ["analyze_rightback.py"],
        "outputs": ["right_wingback_recruitment_progressive.csv"],
    },
    {
        "label": "Holding-pivot candidate screen",
        "scripts": ["analyze_controller.py"],
        "outputs": ["holding_controller_jaissle_detailed.csv"],
    },
    {
        "label": "Advanced / workhorse 8 scouting",
        "scripts": ["analyze_workhorse_cm.py"],
        "outputs": ["workhorse_cm_jaissle_detailed.csv"],
    },
    {
        "label": "Creative 10 scouting",
        "scripts": ["analyze_cam.py"],
        "outputs": ["creative_10_jaissle_detailed.csv"],
    },
    {
        "label": "Winger archetype scouting",
        "scripts": ["analyze_winger.py"],
        "outputs": ["winger_jaissle_archetypes.csv"],
    },
    {
        "label": "Focal striker scouting",
        "scripts": ["analyze_striker.py"],
        "outputs": ["focal_striker_jaissle_detailed.csv"],
    },
]

POST = ROLE_DIR / "run_jaissle_post_analysis.py"

FINAL_ANALYSIS_CANDIDATES = [
    ROLE_DIR / "build_final_shortlist_analysis.py",
    ANALYZE_DIR / "build_final_shortlist_analysis.py",
]

IMAGE_CREATION = ANALYZE_DIR / "run_jaissle_image_creation.py"

SHORTLIST_VIS_CANDIDATES = [
    ANALYZE_DIR / "plot_final_shortlist_vs_position.py",
    ROLE_DIR / "plot_final_shortlist_vs_position.py",
]


def find_script(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_role_script(names):
    for name in names:
        path = ROLE_DIR / name
        if path.exists():
            return path
    return None


def outputs_exist(names):
    return bool(names) and all(
        (SHORTLIST / name).exists()
        for name in names
    )


def run_stage(label, script):
    print()
    print("=" * 120)
    print(f"RUNNING: {label}")
    print("=" * 120)

    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: {script}"
        )

    print(
        f"[OK] {label} ({time.time() - t0:.1f}s)"
    )




def reset_visualizations():
    """Delete all previously generated visualization files before a fresh run."""
    if VISUALIZATIONS.exists():
        shutil.rmtree(VISUALIZATIONS)

    VISUALIZATIONS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[reset] Visualizations folder: {VISUALIZATIONS}")


def main():
    print("=" * 120)
    print("JAISSLE RECRUITMENT PIPELINE")
    print("=" * 120)
    print(f"Project root: {ROOT}")

    reset_visualizations()

    # 1. Rebuild positional scouting tables.
    for stage in STAGES:
        script = find_role_script(
            stage["scripts"]
        )

        if script is not None:
            print(
                f"[resolved] {stage['label']}: {script.name}"
            )
            run_stage(
                stage["label"],
                script,
            )

        elif outputs_exist(stage["outputs"]):
            print(
                f"[OK] Existing output retained: {stage['label']}"
            )

        else:
            raise FileNotFoundError(
                f"{stage['label']}: no analyzer and no required output found"
            )

    # 2. Build combined recruitment/value outputs.
    run_stage(
        "Jaissle post-analysis",
        POST,
    )

    # 3. Build our hand-picked final shortlist table.
    final_analysis = find_script(
        FINAL_ANALYSIS_CANDIDATES
    )

    if final_analysis is not None:
        run_stage(
            "Final shortlist analysis",
            final_analysis,
        )
    else:
        print(
            "[warn] build_final_shortlist_analysis.py not found; "
            "shortlist comparison charts may use an older CSV."
        )

    # 4. Existing role-profile visualization pack.
    if IMAGE_CREATION.exists():
        run_stage(
            "Role visualization pack",
            IMAGE_CREATION,
        )
    else:
        print(
            "[warn] run_jaissle_image_creation.py not found"
        )

    # 5. Guaranteed shortlist-vs-position market charts.
    shortlist_vis = find_script(
        SHORTLIST_VIS_CANDIDATES
    )

    if shortlist_vis is not None:
        run_stage(
            "Final shortlist positional comparisons",
            shortlist_vis,
        )
    else:
        print(
            "[warn] plot_final_shortlist_vs_position.py not found"
        )

    print()
    print("=" * 120)
    print("RECRUITMENT PIPELINE COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
