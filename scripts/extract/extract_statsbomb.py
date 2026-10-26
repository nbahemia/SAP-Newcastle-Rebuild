from pathlib import Path

import pandas as pd
from statsbombpy import sb

COMPETITION_ID = 43
SEASON_ID = 106

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_matches():
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    
    matches.to_parquet(RAW_DATA_DIR / "matches.parquet", index=False)
    
    print(f"Saved {len(matches)} matches")
    
    return matches
    
def extract_events(matches):
    all_events = []
    
    for match_id in matches["match_id"]:
        print(f"Extracting match {match_id}")
        
        events = sb.events(match_id=match_id)
        
        events["match_id"] = match_id
        
        all_events.append(events)
        
    events_df = pd.concat(all_events, ignore_index=True)
    
    events_df.to_parquet(RAW_DATA_DIR / "events.parquet", index=False)
    
    print(f"Saved {len(events_df)} events")
    
def extract_lineups(matches):
    all_lineups = []

    for match_id in matches["match_id"]:
        print(f"Extracting lineups for match {match_id}")

        match_lineups = sb.lineups(match_id=match_id)

        for team, lineup in match_lineups.items():
            lineup = lineup.copy()

            lineup["match_id"] = match_id
            lineup["team"] = team

            all_lineups.append(lineup)

    lineups_df = pd.concat(
        all_lineups,
        ignore_index=True
    )

    lineups_df.to_parquet(
        RAW_DATA_DIR / "lineups.parquet",
        index=False
    )

    print(f"Saved {len(lineups_df):,} lineup rows")

    return lineups_df

def main():
    matches = extract_matches()

    extract_events(matches)
    extract_lineups(matches)
    
if __name__ == "__main__":
    main()