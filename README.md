# F1 Predictions

Predictive modeling for Formula 1 race results and F1 Fantasy optimization.

## What This Is

Jupyter notebooks that predict race outcomes, podium probabilities, and optimal F1 Fantasy lineups for each Grand Prix weekend. Built for the 2026 season — a regulation-change year with new power units, active aerodynamics, and completely reshuffled constructor performance.

## Methodology (v0.2)

The 2026 regulations are the biggest change in F1 since 2022. Pre-2026 historical data has **limited predictive value** for car performance because the pecking order has been completely reshuffled. The model accounts for this:

| Factor | Weight | Data Source |
|--------|--------|-------------|
| Constructor Strength | 50% | 2026 race results only |
| Driver Skill | 20% | Teammate deltas + racecraft |
| 2026 Form | 20% | Actual finishing positions |
| Track Factor | 10% | Historical track performance (discounted) |

### Outputs per race weekend:
- **Predicted finish order** — composite ranking of all drivers
- **Podium probabilities** — Monte Carlo simulation (10,000 races) with calibrated uncertainty
- **Head-to-head matchups** — teammate battle probabilities
- **F1 Fantasy recommendations** — expected points, value picks (pts/$M), boost candidates, constructor picks

### F1 Fantasy Scoring Modeled:
- Qualifying positions (P1=10 ... P10=1)
- Race positions (P1=25 ... P10=1)
- Positions gained (+1 per position)
- Overtakes (+1 each)
- Fastest lap (+10), Driver of the Day (+10)
- DNF penalty (-20 race, -10 sprint)
- Constructor bonuses (qualifying, pit stops)

## Setup

```bash
# Clone
git clone https://github.com/nina-coder/f1-predictions.git
cd f1-predictions

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## Dependencies

- **FastF1** — F1 timing and telemetry data (2023+)
- **pandas** — data manipulation
- **scikit-learn** — regression models
- **matplotlib** — visualizations
- **JupyterLab** — interactive notebook environment

## Project Structure

```
f1-predictions/
├── notebooks/           # One notebook per race weekend
│   └── 2026-R03-japan.ipynb
├── data/
│   └── cache/           # FastF1 data cache (gitignored)
├── models/              # Saved model artifacts (future)
├── requirements.txt
└── README.md
```

## 2026 Constructor Tiers (after Round 2)

| Tier | Teams | Avg Finish |
|------|-------|------------|
| 1 — Front-runners | Mercedes | P1.5 |
| 2 — Strong midfield | Ferrari | P3.5 |
| 3 — Midfield | Haas, Alpine, Racing Bulls | P9-10 |
| 4 — Lower midfield | Red Bull, Williams, Audi, Cadillac | P12-16 |
| 5 — Backmarkers | McLaren, Aston Martin | P16-18 |

## Roadmap

- [x] Japan GP predictions (Round 3)
- [ ] Post-qualifying update with actual grid positions
- [ ] Post-race accuracy analysis
- [ ] Weather integration
- [ ] Practice session pace data
- [ ] Tire strategy modeling
- [ ] Season-long tracking dashboard
