from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load raw events
# --------------------------------------------------

events = pd.read_parquet(RAW_DATA_DIR / "events.parquet")

print(f"Loaded {len(events)} events")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_location(location):
    if isinstance(location, np.ndarray):
        location = location.tolist()
        
    return location

def get_x(location):
    location = normalize_location(location)
        
    if isinstance(location, list):
        return location[0]
    
    return np.nan

def get_y(location):
    location = normalize_location(location)
        
    if isinstance(location, list):
        return location[1]
    
    return np.nan

def distance_to_goal(location):
    location = normalize_location(location)
        
    if not isinstance(location, list):
        return np.nan
    
    x, y = location
        
    return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)

# --------------------------------------------------
# Passing
# --------------------------------------------------

def build_passing_events(events):
    passes = events[events["type"] == "Pass"].copy()
    
    passes["start_x"] = passes["location"].apply(get_x)
    passes["start_y"] = passes["location"].apply(get_y)
    
    passes["end_x"] = passes["pass_end_location"].apply(get_x)
    passes["end_y"] = passes["pass_end_location"].apply(get_y)
    
    passes["successful"] = passes["pass_outcome"].isna()
    
    passes["start_distance_to_goal"] = passes["location"].apply(distance_to_goal)
    
    passes["end_distance_to_goal"] = passes["pass_end_location"].apply(distance_to_goal)
    
    passes["distance_progressed"] = passes["start_distance_to_goal"] - passes["end_distance_to_goal"]
    
    # Progressive passes 
    # Passes that move the ball forward at least 20 yards in own half or 10 yards in opposition half
    passes["progressive_pass"] = (
        passes["successful"] & passes["end_x"].gt(passes["start_x"]) & (
            ((passes["start_x"] < 60) & (passes["distance_progressed"] >= 20)) |
            ((passes["start_x"] >= 60) & (passes["distance_progressed"] >= 10))
        )
    )
    
    # Passes into final third
    passes["pass_into_final_third"] = (
        passes["successful"] & passes["end_x"].gt(passes["start_x"]) &
        passes["start_x"].lt(80) & passes["end_x"].ge(80)
    )
    
    return passes

# --------------------------------------------------
# Carries
# --------------------------------------------------

def build_carry_events(events):
    carries = events[events["type"] == "Carry"].copy()
    
    carries["start_x"] = carries["location"].apply(get_x)
    carries["start_y"] = carries["location"].apply(get_y)
    
    carries["end_x"] = carries["carry_end_location"].apply(get_x)
    carries["end_y"] = carries["carry_end_location"].apply(get_y)
    
    carries["start_distance_to_goal"] = carries["location"].apply(distance_to_goal)
    
    carries["end_distance_to_goal"] = carries["carry_end_location"].apply(distance_to_goal)
    
    carries["distance_progressed"] = carries["start_distance_to_goal"] - carries["end_distance_to_goal"]
    
    # Progressive carries 
    # Carries that move the ball forward at least 10 yards
    carries["progressive_carry"] = (
        (carries["end_x"] > carries["start_x"])
        & (carries["distance_progressed"] >= 10)
    )

    carries["carry_into_final_third"] = (
        (carries["start_x"] < 80)
        & (carries["end_x"] >= 80)
    )

    return carries

# --------------------------------------------------
# Defensive events
# --------------------------------------------------

def build_defensive_events(events):
    pressures = events[events["type"] == "Pressure"].copy()
    
    recoveries = events[events["type"] == "Ball Recovery"].copy()
    
    interceptions = events[events["type"] == "Interception"].copy()
    
    duels = events[events["type"] == "Duel"].copy()
    
    pressures["x"] = pressures["location"].apply(get_x)
    interceptions["x"] = interceptions["location"].apply(get_x)
    duels["x"] = duels["location"].apply(get_x)
    recoveries["x"] = recoveries["location"].apply(get_x)
    
    pressures["attacking_third_pressure"] = pressures["x"].ge(80)
    recoveries["attacking_third_recovery"] = recoveries["x"].ge(80)
    
    return pressures, recoveries, interceptions, duels
    
# --------------------------------------------------
# Run transformations
# --------------------------------------------------

passes = build_passing_events(events)

carries = build_carry_events(events)

pressures, recoveries, interceptions, duels = build_defensive_events(events)

#--------------------------------------------------
# Build player match metrics
#--------------------------------------------------

passing_metrics = (
    passes
    .groupby(["match_id", "team", "player"])
    .agg(
        successful_passes=("successful", "sum"),
        progressive_passes=("progressive_pass", "sum"),
        passes_into_final_third=("pass_into_final_third", "sum"),
    )
    .reset_index()
)

carrying_metrics = (
    carries
    .groupby(["match_id", "team", "player"])
    .agg(
        progressive_carries=("progressive_carry", "sum"),
        carries_into_final_third=("carry_into_final_third", "sum"),
    )
    .reset_index()
)

pressure_metrics = (
    pressures
    .groupby(["match_id", "team", "player"])
    .agg(
        pressures=("type", "size"),
        attacking_third_pressures=("attacking_third_pressure", "sum"),
    )
    .reset_index()
)

recovery_metrics = (
    recoveries
    .groupby(["match_id", "team", "player"])
    .agg(
        recoveries=("type", "size"),
        attacking_third_recoveries=("attacking_third_recovery", "sum"),
    )
    .reset_index()
)

interception_metrics = (
    interceptions
    .groupby(["match_id", "team", "player"])
    .size()
    .reset_index(name="interceptions")
)

duel_metrics = (
    duels
    .groupby(["match_id", "team", "player"])
    .size()
    .reset_index(name="duels")
)

player_match_metrics = passing_metrics.copy()

for df in [
    carrying_metrics,
    pressure_metrics,
    recovery_metrics,
    interception_metrics,
    duel_metrics,
]:
    player_match_metrics = player_match_metrics.merge(
        df,
        on=["match_id", "team", "player"],
        how="outer"
    )
    
metric_columns = [
    "successful_passes",
    "progressive_passes",
    "passes_into_final_third",
    "progressive_carries",
    "carries_into_final_third",
    "pressures",
    "attacking_third_pressures",
    "recoveries",
    "attacking_third_recoveries",
    "interceptions",
    "duels",
]

player_match_metrics[metric_columns] = (
    player_match_metrics[metric_columns]
    .fillna(0)
)
    
# --------------------------------------------------
# Save player match metrics
# --------------------------------------------------

player_match_metrics.to_parquet(
    PROCESSED_DATA_DIR / "player_match_metrics_raw.parquet",
    index=False
)

print(
    f"Saved {len(player_match_metrics):,} player-match rows"
)