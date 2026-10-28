## Methodology

This project builds a football analytics pipeline for evaluating player and team performance using StatsBomb event data. The current implementation uses the 2022 FIFA World Cup as a proof-of-concept dataset before applying the same methodology to a full league-season dataset.

### Data Pipeline

The workflow follows:

- Raw event and lineup data
- Player-match metrics
- Corrected player minutes
- Per-90 metrics
- Tournament aggregates
- Position classification
- Position-specific percentiles

Player minutes are standardized to 90 minutes for regulation matches and 120 minutes for matches that include extra time. Substitution and lineup information are used to estimate each player's playing time.

### Player Metrics

The current player-level metrics focus on progression, possession advancement, pressing, and defensive activity.

Metrics include:

- Successful passes
- Progressive passes
- Passes into the final third
- Progressive carries
- Carries into the final third
- Pressures
- Attacking-third pressures
- Ball recoveries
- Attacking-third recoveries
- Interceptions
- Duels

A progressive pass is defined as a successful forward pass that meaningfully reduces the distance to goal. The required amount of progression varies depending on where the pass begins, with larger progression required from deeper areas of the pitch.

### Per-90 Metrics

Tournament-level per-90 statistics are calculated using total actions divided by total minutes played, multiplied by 90.

This avoids averaging individual match per-90 values, which could give short substitute appearances disproportionate influence.

### Position Classification

Players are grouped into broader positional categories:

- GK
- CB
- FB/WB
- DM/CM
- AM
- W
- ST

Position classification is based on where players actually played in the dataset rather than their listed or traditional position.

A player's primary position is determined by where they played the largest share of minutes played. The player's secondary position is retained when a player spends at least 10% of their minutes in another positional group.

### Percentile Rankings

Players are compared only against others in their primary position group.

This prevents statistically misleading comparisons between players with fundamentally different tactical responsibilities.

Players must play at least 180 minutes to be included in the percentile comparison pool. This threshold signals that the player has played roughly two full matches and reduces extreme results caused by very small samples. This threshold maintains a usable peer-group sizes within the data of the short tournament.

A 90th-percentile result means a player performed better than approximately 90% of players in their positional comparison group for that metric.

### Current Limitations

The current dataset is primarily intended to validate the analytical pipeline.

World Cup data has several limitations:

- Maximum sample size is only seven matches per player
- Many players have significantly fewer minutes
- National-team roles may differ from club roles
- Tournament tactics differ vastly from full-season league football
- Opposition quality varies substantially
- Some positional comparison groups are small

The final recruitment analysis will therefore use a larger league-season dataset where possible.