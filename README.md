# F1 Predictions

Predictive modeling for Formula 1 race results and F1 Fantasy optimization.

**Latest predictions:** [nina-coder.github.io/f1-predictions](https://nina-coder.github.io/f1-predictions/)

## What This Is

A machine learning model that predicts F1 race results using 24 features — the same kinds of data a race engineer considers when building race strategy. The model runs 10,000 simulated races to estimate podium probabilities, then translates predictions into F1 Fantasy lineup recommendations.

Built for the 2026 season, where massive regulation changes reshuffled the entire competitive order.

## How It Works

Two-model XGBoost approach: **car pace** (2026 data only, weighted 10x) and **driver skill** (all 72 races, measured relative to the car). Practice and qualifying data are pulled live via FastF1 and blended into predictions throughout the race weekend.

| Category | Features |
|----------|----------|
| **Car** | car_pace, car_speed |
| **Driver** | teammate_delta, positions_gained_avg, driver_avg_finish, first_lap_gain, momentum, quali_race_gap |
| **Tire** | tire_degradation, avg_pit_stops, team_pit_strategy |
| **Track** | sector1/2/3_delta, track_experience |
| **Conditions** | air_temp, humidity, had_rain, wet_skill |
| **Context** | is_2026, team_changed, consistency, dnf_rate |

Leave-one-out validation: **MAE 2.39 positions**.

## 2026 Season Results

Predicted winner from the post-qualifying model vs. the actual result. The model has called the winner in **4 of 4** races — but the honest test is whether it beats the naive baseline of "everyone finishes where they qualified." It only does so at the high-overtaking tracks. The **Edge** column is grid-baseline MAE minus model MAE (positive = model wins).

| Rnd | Race | Pred P1 | Actual | Win | Top 5 | Top 10 | Model MAE | Grid MAE | Edge |
|-----|------|---------|--------|:---:|-------|--------|-----------|----------|------|
| R03 | Japan | Antonelli | Antonelli | ✅ | 4/5 | 8/10 | 1.90 | 1.95 | **+0.05** |
| R04 | Miami | Antonelli | Antonelli | ✅ | 4/5 | 7/10 | 3.00 | 2.56 | −0.44 |
| R05 | Canada | Antonelli | Antonelli | ✅ | 2/5 | 6/10 | 3.88 | 3.94 | **+0.06** |
| R06 | Monaco | Antonelli | Antonelli | ✅ | 2/5 | 6/10 | 4.13 | 3.87 | −0.27 |

Per-race pages: [Japan](https://nina-coder.github.io/f1-predictions/r03-japan/) · [Miami](https://nina-coder.github.io/f1-predictions/r04-miami/) · [Canada](https://nina-coder.github.io/f1-predictions/r05-canada/) · [Monaco](https://nina-coder.github.io/f1-predictions/r06-monaco/)

*Rounds 1-2 (Australia, China) predate the model. Bahrain and Saudi Arabia were cancelled. R03 was a live pre-race prediction; R04-R06 are retrospectives reconstructed from each weekend's qualifying — no race data enters the model, so the accuracy check is fair. The current model is roughly a qualifying-order copy; closing the gap to the baseline is the active work.*

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
| **v1.0+Q** | **24** | **Live qualifying integration + 5-way car pace blend** |

## Run It Yourself

Requires Python 3.10+ and a [FastF1](https://docs.fastf1.dev/) cache directory.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Roadmap

- [x] Japan GP predictions (Round 3) — v1.0+Q
- [x] Practice + qualifying live integration
- [x] F1 Fantasy optimization with actual prices
- [x] Season results scoreboard + per-race archive pages
- [x] Post-race accuracy analysis (Rounds 3-6)
- [x] Reusable prediction library (`scripts/f1lib.py`) + one-command notebook generator
- [x] Live pre-race mode + DNF promotion in simulations
- [x] Race-day weather forecast input (Open-Meteo) with wet/dry probability blend
- [ ] Overtake probability modeling
- [ ] Enhanced safety car pit window timing
- [ ] Track-difficulty weighting (Monaco/Canada accuracy is the weak spot)
