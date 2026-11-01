
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
ROLE_DIR = ROOT / "scripts" / "bsd" / "analyze" / "role_scouting"
SHORTLIST = ROOT / "data" / "processed" / "bsd" / "shortlist"

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
    }
]

POST = ROLE_DIR / "run_jaissle_post_analysis.py"


def find_script(names):
    for name in names:
        p = ROLE_DIR / name
        if p.exists():
            return p
    return None


def outputs_exist(names):
    return bool(names) and all((SHORTLIST / n).exists() for n in names)


def run_stage(label, script):
    print()
    print("=" * 120)
    print(f"RUNNING: {label}")
    print("=" * 120)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {script}")
    print(f"[OK] {label} ({time.time() - t0:.1f}s)")


def main():
    print("=" * 120)
    print("JAISSLE RECRUITMENT PIPELINE")
    print("=" * 120)
    print(f"Project root: {ROOT}")

    for stage in STAGES:
        script = find_script(stage["scripts"])
        if script is not None:
            print(f"[resolved] {stage['label']}: {script.name}")
            run_stage(stage["label"], script)
        elif outputs_exist(stage["outputs"]):
            print(f"[OK] Existing output retained: {stage['label']}")
        else:
            raise FileNotFoundError(
                f"{stage['label']}: no analyzer and no required output found"
            )

    run_stage("Jaissle post-analysis", POST)

    print()
    print("=" * 120)
    print("RECRUITMENT PIPELINE COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
