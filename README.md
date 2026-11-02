# SAP Newcastle Rebuild

This project is a data-driven Newcastle United rebuild built around Matthias Jaissle's tactical structure.

The goal is not to find the players with the highest generic ratings. The goal is to identify which tactical functions Newcastle need, measure how well the current squad fills those functions, and then find realistic recruitment targets who improve the squad without destroying the balance of the team or the transfer budget.

The project combines:

- historical 2025/26 player data
- tactical role analysis
- position-specific percentile comparisons
- Newcastle-specific role benchmarks
- league-strength adjustments
- market-value and age context
- visualization
- final manual validation using FotMob and football context

The final decision process is intentionally not fully automated. The code is used to reduce the player pool and identify strong profiles. Visualizations and FotMob checks are then used to validate whether the numbers actually describe the player we think they do.

---

## Project Question

The project started with a simple rebuild question:

> If Newcastle were rebuilding around Matthias Jaissle, which players should stay, which positions actually need improvement, and which recruitment targets best fit the tactical roles?

That became a more useful question:

> Does Newcastle have enough players capable of performing each tactical function in Jaissle's system?

This matters because two players can share the same listed position while performing completely different jobs.

For example:

- Nico González and a defensive ball-winner may both be listed as defensive midfielders
- Anthony Elanga and Francisco Conceição may both be right wingers
- Wissa and Krstović may both play striker

but they do not solve the same tactical problem.

The analysis therefore focuses on **roles before positions**.

---

# Current Tactical Model

The working structure is a fluid **4-2-3-1**.

The main tactical ideas are:

- a controlling No. 6 organizing first-phase possession
- a physical/mobile No. 8 supporting pressure and transitions
- aggressive fullbacks creating width
- a creative No. 10 connecting midfield and attack
- wingers providing different forms of threat rather than identical profiles
- a striker capable of occupying centre-backs and creating a central reference point
- a 4-4-2 defensive shape when the No. 10 joins the striker in the first pressing line

The final projected XI currently used for the rebuild is:

```text
                    Krstović

Yıldız             Samardžić             Conceição

              Nico González     Joelinton

Lewis Hall       Botman         Schär          Dedić

                    Horníček
```

Important depth and development pieces include:

- Leopold Querfeld — competitive CB rotation and succession option
- Filip Lissah — developmental CB/RB
- Malick Thiaw — high-level CB rotation / starter-quality depth
- Lewis Miley — midfield progression and controller depth
- Jacob Ramsey — CM/CAM rotation
- Rayan Aït-Nouri — attacking LB option
- Anthony Elanga — transition-focused RW
- Matías Fernández-Pardo — LW/RW/ST flexibility
- Bazoumana Touré — developmental LW
- Wissa / Osula — striker depth with different profiles

---

# Methodology

## 1. Tactical Analysis First

The project did not begin by sorting a player database.

The first step was understanding how Jaissle's team was functioning.

This was developed through:

- Newcastle squad analysis
- watching Newcastle season and tactical breakdowns
- looking at Jaissle's personnel usage
- identifying recurring responsibilities by position
- comparing those observations against player-level data
- using AI-assisted code and documentation to organize and test the analysis

From that, the squad was split into functional roles such as:

- Sweeper / Build-Up GK
- High-Line CB
- Left WB / FB
- Right WB / FB
- Holding Controller
- Advanced / Workhorse 8
- Creative 10
- Wide Creator
- Vertical / Box Winger
- Hybrid Winger
- Focal Striker

This role map became the base of the recruitment model.

---

## 2. Historical Data Pipeline

The active pipeline uses **2025/26 historical European player data** rather than the earlier StatsBomb World Cup proof-of-concept.

The current workflow is:

```text
BSD historical data
        ↓
season-level player metrics
        ↓
per-90 metrics
        ↓
position / role context
        ↓
percentiles
        ↓
Transfermarkt enrichment
        ↓
Newcastle role benchmarks
        ↓
recruitment scoring
        ↓
shortlists
        ↓
visualizations
        ↓
manual FotMob validation
        ↓
final recruitment decision
```

The main historical files are built around:

- season totals
- per-90 production
- positional context
- percentiles
- age
- club
- league
- Transfermarkt value

The current Europe-wide pool contains thousands of player-season rows, allowing comparisons against a much larger sample than the original tournament dataset.

---

## 3. Per-90 Metrics

Raw totals are converted into per-90 values using:

```text
per90 = total actions / total minutes × 90
```

This is preferred over averaging match-level per-90 values because very short substitute appearances can otherwise have too much influence.

Minimum-minute thresholds are used in different recruitment searches to reduce extreme small-sample results.

The exact threshold depends on the role and purpose of the shortlist.

For example, developmental searches can allow lower minutes than searches for immediate starters.

---

## 4. Position and Role Context

Players are not compared across the entire dataset.

Metrics are interpreted relative to role and positional responsibility.

A centre-back should not be rewarded or penalized using the same attacking expectations as a winger, and a controller should not be evaluated like a traditional ball-winning defensive midfielder.

The project therefore uses:

- position filters
- role-specific metric groups
- role-specific scoring
- Newcastle-specific benchmark players

The benchmark player is chosen intentionally.

A depth chart is useful for presentation, but it does not automatically determine who represents the tactical role in the model.

For example:

- Dedić is the right-back benchmark
- Nico González is the starting controller in the current tactical interpretation
- Joelinton is the physical/mobile central midfielder benchmark
- Newcastle's winger comparisons depend on the type of wide role being evaluated

This prevents stale squad ordering from becoming a statistical assumption.

---

## 5. Building the Role Weights

The role weights were not chosen by simply giving every available statistic equal value.

The process was:

1. identify what the role is expected to do tactically
2. identify which available metrics best represent those responsibilities
3. group related metrics into broader football concepts
4. compare the results against players whose style is already understood
5. adjust the weighting when the model produced rankings that did not make football sense

Examples of broader components include:

- progression
- creation
- goal threat
- 1v1 ability
- defensive activity
- aerial ability
- physical / high-line suitability
- pressing
- link play
- box threat

This was important because one of the biggest findings from the project was that a model can be mathematically consistent and still be wrong for the football question.

A high role score is therefore treated as:

> evidence that a player fits the statistical profile

not:

> proof that the player is definitely better.

---

## 6. Tactical Quality vs Recruitment Suitability

The project separates **football quality / role fit** from **recruitment suitability**.

A player can score highly in a role but still be a poor Newcastle target because:

- the current Newcastle player is already strong
- the player comes from a much weaker competitive environment
- the transfer fee is unrealistic
- the player is too old for the intended squad cycle
- the player recently transferred and is not realistically available

The recruitment suitability model developed into:

- **45% tactical quality**
- **25% improvement over Newcastle's current option**
- **25% league strength**
- **5% age / development**

An additional absolute-upgrade adjustment is used so a tiny numerical improvement does not look elite simply because it ranks highly within a weak candidate pool.

This was especially important when comparing positions where Newcastle already had strong starters.

---

## 7. League Strength

League strength became a major part of the model after early rankings over-rewarded strong statistical production from weaker leagues.

The current model gives meaningful weight to the competitive environment.

The scale is anchored with the Premier League at the top and then reduced across other leagues.

Example working values included:

- Premier League — 100
- Serie A — 88.9
- La Liga — 82.9
- Bundesliga — 80.0
- Ligue 1 — 72.9
- Primeira Liga — 70.0
- Championship — 67.0
- Eredivisie — 63.4
- Belgian Pro League — 63.0
- Turkish Süper Lig — 61.0
- Austrian Bundesliga — 58.0
- Scottish Premiership — 56.0
- Danish Superliga — 54.0

These values are not intended to be a universal ranking of world football.

They are a practical adjustment for this recruitment model.

The purpose is to avoid treating the same percentile production as equally proven in every competition.

---

## 8. Value Is Kept Separate

Transfer value is not heavily embedded inside the football-quality score.

This was intentional.

If value is mixed too strongly into the football score, a cheap average player can appear better than an expensive elite player simply because he is inexpensive.

Instead, the project separates:

### Recruitment Suitability

- Tactical quality — 45%
- Newcastle upgrade — 25%
- League strength — 25%
- Age — 5%

### Value Score

- Suitability — 45%
- Affordability — 25%
- Age / development — 15%
- Upgrade — 15%

That allows two different questions:

> Who is the best player for Newcastle?

and

> Who is the best value for Newcastle?

Those are not always the same answer.

---

# Player Metrics

The exact metrics vary by role.

The broader metric pool includes:

### Passing / Progression

- successful passes
- pass completion
- line-breaking passes
- progressive passes
- passes into the final third
- accurate long balls
- long-ball accuracy
- chances created
- xA

### Carrying / 1v1

- successful dribbles
- dribble success
- progressive carries
- carries into the final third
- touches
- penalty-area touches
- dispossessions

### Attacking

- goals
- xG
- non-penalty xG
- xGOT
- shots
- shots on target
- headed shots
- big chances
- box touches

### Defending

- tackles
- interceptions
- recoveries
- final-third recoveries / wins
- clearances
- blocked shots
- defensive contributions
- duels won
- duel win percentage
- aerials won
- aerial win percentage
- times dribbled past

The model does not force every role to use every metric.

---

# Visual Analysis

The code outputs are then converted into visual comparisons.

The visualization stage is used to:

- compare targets against Newcastle players
- identify where a target's score is actually coming from
- detect one-metric outliers
- compare different player archetypes
- see whether the final shortlist has enough tactical variety
- communicate the findings more clearly than a large CSV

The repository keeps a single root `visualizations/` tree for these outputs.

The visual stage is not just presentation.

It is part of the analysis.

A target that looks strong in one composite score can look much less convincing once the individual components are placed next to the current Newcastle player.

---

# Manual FotMob Validation

The final stage is manual.

Once the code reduces the pool, the best candidates are checked against FotMob.

This became necessary because several issues appeared during development:

- some missing metrics were being represented as zero
- some players appeared multiple times because of cross-file identity issues
- minute totals did not always match between generated outputs
- role scores could over-reward a player because of the comparison pool
- league strength could materially change how convincing a percentile looked
- a player's statistical score did not always describe the type of player Newcastle actually needed

FotMob is therefore used to check:

- minutes
- per-90 output
- percentile profile
- actual position
- role
- defensive workload
- passing style
- progression
- attacking involvement
- aerial ability
- duel efficiency
- current club context

The final shortlist is **code-informed, not code-decided**.

---

# Example: Centre-Back Search

The centre-back search became the clearest example of why the final manual stage matters.

An early model could rank players highly even when key build-up statistics were missing.

After auditing the pipeline, the CB evaluation emphasized:

- progression / build-up
- front-foot defending
- duel strength
- aerial strength
- mobility for a higher line
- league strength
- age / upside

The final football judgment then compared the actual profiles.

### Leopold Querfeld

The FotMob check showed:

- 7.89 line-breaking passes / 90
- 4.49 accurate long balls / 90
- 7.02 duels won / 90
- 60.3% duel win rate
- 4.26 aerial wins / 90
- 64.6% aerial win rate
- 1.20 interceptions / 90
- 7.32 clearances / 90

This showed a much more complete profile than the model score alone communicated.

Querfeld became the preferred succession CB because he combined:

- physical defending
- aerial dominance
- progressive passing
- Bundesliga experience
- age
- realistic cost

Filip Lissah remains a lower-cost developmental CB/RB option rather than the same type of signing.

---

# Example: Right-Wing Search

The right-wing search also moved away from forcing one archetype.

Newcastle already had verticality through Anthony Elanga.

The goal therefore became finding a winger who could add:

- 1v1 quality
- creation
- box involvement
- goal threat
- technical quality

rather than simply finding another fast transition runner.

Francisco Conceição emerged as a strong fit because his historical profile included:

- strong dribbling volume
- high shot volume
- good box involvement
- secondary creation
- line-breaking involvement
- pressing contribution

The final choice was therefore based on **squad balance**, not just his individual winger score.

---

# Current Recruitment Direction

The rebuild eventually moved toward:

### Incoming

- Nikola Krstović
- Kenan Yıldız
- Lazar Samardžić
- Rayan Aït-Nouri
- Bart van Rooij
- Francisco Conceição
- Leopold Querfeld

### Development / Depth

- Filip Lissah as a developmental CB/RB option

### Outgoing

- Tino Livramento
- Harvey Barnes
- Jacob Murphy
- Joe Willock

Jacob Ramsey was retained because his CM/CAM flexibility still provides useful squad depth.

The rebuild does not need to spend every available euro.

After the current planned moves, the remaining budget can be kept for future windows or opportunistic transfers.

---

# Main Findings

## 1. Tactical role matters more than listed position

The most important methodological change was moving from positional depth to functional depth.

---

## 2. Composite scores need context

A score is useful for reducing the player pool.

It should not replace checking the underlying metrics.

---

## 3. Missing data must remain missing

Missing values should not automatically become zero.

This was one of the most important technical lessons from the CB/RW audit.

---

## 4. League strength matters

Percentile performance is not equally transferable from every competition.

---

## 5. Newcastle's existing starter matters

Recruitment should measure improvement over the actual Newcastle benchmark, not just rank targets against each other.

---

## 6. Value and quality are different questions

The best bargain and the best player are not automatically the same player.

---

## 7. Squad balance matters

The final winger decision is a good example.

Newcastle did not need another version of Elanga.

The squad needed a different form of right-sided threat.

---

## 8. Manual review remains necessary

The final process is:

> **code → visualization → FotMob → football decision**

That is the intended methodology of the project.

---

# Repository Structure

```text
SAP-Newcastle-Rebuild/
├── data/
│   ├── external/
│   │   └── transfermarkt/
│   ├── raw/
│   │   └── bsd/
│   └── processed/
│       └── bsd/
│           ├── europe_2025_26_player_season_metrics.parquet
│           ├── europe_2025_26_player_percentiles.parquet
│           ├── europe_2025_26_player_percentiles_enriched.parquet
│           └── shortlist/
├── scripts/
│   └── bsd/
│       ├── extract/
│       ├── transform/
│       └── analyze/
├── visualizations/
├── reports/
├── README.md
├── Role_Analysis.md
└── requirements.txt
```

---

# Main Scripts

### Build the European historical pool

```powershell
python .\scripts\bsd\extract\build_europe_2025_2026_pool.py
```

### Rebuild the recruitment pipeline

```powershell
python .\scripts\bsd\transform\rebuild_europe_recruitment_pipeline.py
```

### Analyze Newcastle's current squad

```powershell
python .\scripts\bsd\analyze\analyze_newcastle_squad.py
```

### Run the Jaissle recruitment role pipeline

```powershell
python .\scripts\bsd\analyze\role_scouting\run_jaissle_recruitment_pipeline.py
```

### Generate the final CB / RW focus

```powershell
python .\scripts\bsd\analyze\find_final_cb_rw_targets.py
```

### Generate final visual comparisons

```powershell
python .\scripts\bsd\analyze\plot_final_shortlist_vs_position.py
```

---

# Limitations

The model still has limitations.

- event-provider metrics do not capture every tactical action
- positional labels can still be imperfect
- player roles change between clubs
- league-strength weights are model assumptions
- market values are not guaranteed transfer fees
- injuries and availability can change the practical value of a target
- percentile scores depend on the comparison pool
- some metrics require manual validation when source coverage is incomplete
- the current model does not fully estimate tactical chemistry or adaptation risk

Because of this, the final recruitment board should never be read as an automatic transfer recommendation.

It is a structured way to get from thousands of players to a small group worth seriously analyzing.
