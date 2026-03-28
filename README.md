# F1 Predictions

Predictive modeling for Formula 1 race results and F1 Fantasy optimization.

**Live predictions:** [nina-coder.github.io/f1-predictions](https://nina-coder.github.io/f1-predictions/)

## What This Is

A machine learning model that predicts F1 race results using 24 features — the same kinds of data a race engineer considers when building race strategy. The model runs 10,000 simulated races to estimate podium probabilities, then translates predictions into F1 Fantasy lineup recommendations.

Built for the 2026 season, which introduced massive regulation changes (new power units, active aerodynamics, smaller cars, 100% sustainable fuel). The entire competitive order has been reshuffled, making historical data less reliable and current-season data critical.

## Model v1.0+Q — 24 Features + Qualifying Integration

The model uses a **two-model approach**: car pace (from 2026 data only) and driver skill (from all 72 races, measured relative to the car). This ensures 2025 patterns don't bleed into 2026 predictions.

**Race weekend integration:** Practice (FP1/FP2/FP3) and qualifying data are pulled live via FastF1 and blended into the model. Car pace uses a 5-way blend (10% season + 15% FP1 + 20% FP2 + 20% FP3 + 35% qualifying). Qualifying provides actual grid positions (the #1 predictor) and live sector deltas. Simulations use position-dependent variance — front-row starters have less noise than the chaotic midfield pack.

### Features

| Category | Features | Description |
|----------|----------|-------------|
| **Car Performance** | car_pace, car_speed | 2026 constructor avg finish + straight-line speed |
| **Driver Skill** | teammate_delta, positions_gained_avg, driver_avg_finish | Car-independent skill metrics |
| **Race Starts** | first_lap_gain | Positions gained/lost on lap 1 |
| **Tire Strategy** | tire_degradation, avg_pit_stops, team_pit_strategy | Tire management + pit patterns |
| **Track-Specific** | sector1/2/3_delta, track_experience | Suzuka sector performance (S1 technical, S2 power, S3 mixed) |
| **Momentum** | momentum | Trend over last 5 races (current team only for team changers) |
| **Conditions** | air_temp, humidity, had_rain, wet_skill | Weather + historical wet performance |
| **Consistency** | consistency, dnf_rate | Lap time variation + reliability |
| **Context** | is_2026, team_changed, quali_race_gap | Regulation era + team change uncertainty |

### Key Methodological Choices

- **2026 data weighted 10x** — in a regulation-change year, current car performance matters far more than history
- **XGBoost gradient boosting** — gold standard for F1 prediction (77-82% winner accuracy in research)
- **Leave-one-out validation** — honest MAE of 2.39 positions (model never sees data it's tested on)
- **Safety car in simulations** — 33% probability at Suzuka, compresses the field
- **Team change flag** — 9 drivers switched teams; the model learns this adds uncertainty

### Outputs per race weekend

- Predicted finish order with feature breakdown
- Podium probabilities (10,000 simulated races)
- Rain scenario — how predictions shift in wet conditions
- Head-to-head teammate battle probabilities
- F1 Fantasy recommendations (expected points, value picks, boost + constructor recommendations)
- Personalized team analysis with transfer suggestions

## Model Evolution

| Version | Features | Key Addition |
|---------|----------|-------------|
| v0.1 | 3 | ELO baseline |
| v0.2 | 9 | Regulation-aware weighting |
| v0.3 | 9 | XGBoost ML |
| v0.4 | 15 | Two-model approach + weather |
| v0.5 | 15 | Speed traps + lap consistency |
| v0.6 | 18 | First lap + tire degradation + momentum |
| v0.7 | 21 | Suzuka sectors + safety car simulations |
| v0.8 | 21 | 2026 data weighted 10x |
| v0.9 | 22 | Honest validation + team change flag |
| v1.0 | 24 | Tire strategy + complete race engineer model |
| **v1.0+Q** | **24** | **Actual qualifying grid + live sectors + 5-way car pace + position-dependent sims** |

## 2026 Constructor Tiers (post-qualifying, Round 3 Japan)

| Tier | Teams | Qualifying Avg | Notes |
|------|-------|---------------|-------|
| 1 — Front-runners | Mercedes | P1.5 | Antonelli pole, Russell P2 |
| 2 — Strong midfield | Ferrari, McLaren | P4-5 | McLaren surged from Tier 5 (season P16!) |
| 3 — Midfield | Red Bull, Alpine, Audi, Racing Bulls | P9-12 | Verstappen only P11 (Q2 exit) |
| 4 — Lower midfield | Haas, Williams | P15-17 | Bearman P18 despite strong race pace |
| 5 — Backmarkers | Cadillac, Aston Martin | P19-22 | 3+ seconds off pole |

## F1 Fantasy

Full scoring system modeled:
- **Qualifying:** P1=10 ... P10=1, NC=-5
- **Race:** P1=25 ... P10=1, DNF=-20
- **Bonuses:** +1/position gained, +1/overtake, +10 fastest lap, +10 Driver of the Day
- **Constructors:** Both Q3=+10, both Q2=+5, pit stops <2.2s=+5-10
- **Budget:** $100M for 5 drivers + 2 constructors, 1 driver gets 2x boost

## Setup

```bash
git clone https://github.com/nina-coder/f1-predictions.git
cd f1-predictions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Tech Stack

- **XGBoost** — gradient boosting regression
- **FastF1** — F1 timing, telemetry, and weather data (2023+)
- **pandas** — data manipulation
- **scikit-learn** — validation metrics
- **matplotlib** — dark-themed visualizations
- **JupyterLab** — interactive notebooks
- **GitHub Pages** — published predictions (custom dark HTML template)

## Project Structure

```
f1-predictions/
├── notebooks/
│   └── 2026-R03-japan.ipynb     # Race prediction notebook
├── data/
│   ├── all_results.csv          # 72 races, 1,442 entries
│   ├── weather.csv              # Weather for all races
│   ├── lap_stats.csv            # Speed traps, pit stops, consistency
│   ├── advanced_features.csv    # First lap, tire deg, momentum
│   ├── suzuka_sectors.csv       # Suzuka S1/S2/S3 sector times
│   └── cache/                   # FastF1 cache (gitignored)
├── docs/
│   ├── index.html               # Published predictions page
│   └── custom.tpl               # Dark theme HTML template
├── requirements.txt
└── README.md
```

## Race Weekend Workflow

1. **Monday-Thursday:** Build/update notebook with latest data
2. **Friday:** Pull FP1/FP2 practice pace, update predictions
3. **Saturday:** Pull qualifying results, re-run with actual grid — publish update
4. **Sunday:** Watch race, post-race accuracy analysis
5. **Monday:** Update model with new data point, start next race notebook

## Roadmap

- [x] Japan GP predictions (Round 3) — v1.0
- [x] Practice session pace integration (FP1 3/26, FP2 3/27, FP3 3/28)
- [x] Actual F1 Fantasy driver prices
- [x] Post-qualifying update with actual grid positions, live sectors, 5-way blend (3/28)
- [ ] Post-race accuracy analysis (Sunday 3/29)
- [ ] Round 4 notebook
- [ ] Overtake probability modeling
- [ ] Enhanced safety car pit window timing
- [ ] Season-long accuracy tracking dashboard
