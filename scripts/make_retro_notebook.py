"""Generate a prediction notebook for a 2026 race.

    python scripts/make_retro_notebook.py <round> <Event> <TrackLabel> <sectors_csv> <date_str> [--pre-race]

Examples:
    # retrospective (race has run) — includes the accuracy section
    python scripts/make_retro_notebook.py 4 Miami "Miami International Autodrome" miami_sectors.csv "May 1-3, 2026"

    # live, Saturday after qualifying (race not yet run) — stops before grading,
    # pulls the race-day forecast automatically (Open-Meteo, no API key)
    python scripts/make_retro_notebook.py 7 Barcelona "Circuit de Barcelona-Catalunya" barcelona_sectors.csv "June 12-14, 2026" --pre-race --weather auto

    # ...or pass the forecast by hand as temp,humidity,rain%  (e.g. 24C / 70% / 40% rain)
    python scripts/make_retro_notebook.py 7 Barcelona "Circuit de Barcelona-Catalunya" barcelona_sectors.csv "June 12-14, 2026" --pre-race --weather 24,70,40

Writes notebooks/2026-R0N-<event>.ipynb. Execute it with nbconvert to populate
outputs before committing. --weather is optional; without it the prediction uses
qualifying conditions as the race-day proxy.
"""
import sys
import json

# Modes: retrospective (default, race done) | --pre-race (post-quali, pre-race)
# | --pre-qualifying (early weekend, no quali yet — estimated grid)
PRE_RACE = '--pre-race' in sys.argv
PRE_QUALI = '--pre-qualifying' in sys.argv
# --weather auto | --weather T,H,R | (absent) -> stage default conditions
WEATHER = ''
_av = sys.argv[1:]
if '--weather' in _av:
    _i = _av.index('--weather')
    WEATHER = _av[_i + 1]
    del _av[_i:_i + 2]
args = [a for a in _av if a not in ('--pre-race', '--pre-qualifying')]

RND = int(args[0])
EVENT = args[1]
TRACK = args[2]
SECTORS = args[3]
DATE = args[4]
SLUG = EVENT.lower().replace(' ', '-')

if PRE_QUALI:
    TITLE_MD = f"""# \U0001f3ce️ Nina's F1 Predictions: 2026 {EVENT} Grand Prix (Round {RND})
## {TRACK} — {DATE}

**Model version:** v1.0+Q (early weekend, pre-qualifying)
**Confidence:** MEDIUM — grid is estimated, not yet set

### What This Is

A machine learning model that predicts Formula 1 race results from 24 features — the same kinds of data a race engineer considers when building race strategy. This is the **first cut of the weekend**, made before qualifying: the model trains on every race so far this season (2026 weighted 10x over 2023-2025 history) and estimates the grid from team pace plus driver form. It refreshes as practice and qualifying data arrive — the qualifying update is the biggest jump in accuracy.

> **The grid here is an estimate, so treat this as a baseline.** It will be replaced by the real qualifying result, and the race accuracy check added, as the weekend progresses.
"""
elif PRE_RACE:
    TITLE_MD = f"""# \U0001f3ce️ Nina's F1 Predictions: 2026 {EVENT} Grand Prix (Round {RND})
## {TRACK} — {DATE}

**Model version:** v1.0+Q (live, post-qualifying)
**Confidence:** HIGH

### What This Is

A machine learning model that predicts Formula 1 race results from 24 features — the same kinds of data a race engineer considers when building race strategy. The model trains on every race so far this season (2026 weighted 10x over 2023-2025 history), takes the **actual qualifying grid** and **live qualifying sector times** for this weekend, then runs 10,000 simulated races to estimate podium probabilities.

> **This is a live prediction made after qualifying, before the race.** The accuracy check will be added here once the race has run.
"""
else:
    TITLE_MD = f"""# \U0001f3ce️ Nina's F1 Predictions: 2026 {EVENT} Grand Prix (Round {RND})
## {TRACK} — {DATE}

**Model version:** v1.0+Q (retrospective)
**Confidence:** HIGH

### What This Is

A machine learning model that predicts Formula 1 race results from 24 features — the same kinds of data a race engineer considers when building race strategy. The model trains on every race through the round before this one (2026 weighted 10x over 2023-2025 history), takes the **actual qualifying grid** and **live qualifying sector times** for this weekend, then runs 10,000 simulated races to estimate podium probabilities.

> **This is a reconstructed prediction.** The model was rebuilt using only data available before the {EVENT} GP — nothing from the race itself enters the model — so the accuracy check in the final section is a fair test. It backfills the season scoreboard for a race weekend that ran before the model was caught up.
"""

SETUP = f"""import sys; sys.path.insert(0, '../scripts')
import f1lib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background'); plt.rcParams['figure.figsize'] = (10, 6)
f1lib.enable_cache('../data/cache')

EVENT, RND, TRACK = '{EVENT}', {RND}, '{TRACK}'
SECTORS = '../data/{SECTORS}'
PRE_RACE = {PRE_RACE}
WEATHER = '{WEATHER}'

df, weather, lap_stats, advanced, lk = f1lib.load_data('../data')
# If this round is already in the data use its slot; otherwise (pre-race, race not
# yet appended) train on everything available by targeting one past the last race.
_rows = df[(df.Year == 2026) & (df.Round == RND)]
target_seq = int(_rows['race_seq'].iloc[0]) if len(_rows) else int(df['race_seq'].max() + 1)

sec = f1lib.track_sector_deltas(SECTORS)
cp_season = f1lib.season_pace(df, 2026, target_seq)
speeds_2026 = lap_stats[lap_stats.Year == 2026].groupby('Team')['avg_speed_st'].mean().to_dict()

feat = f1lib.engineer_features(df, lap_stats, lk, sec, target_seq, cp_season)
model, loo_mae, residual_std = f1lib.train(feat, target_seq)

print('=' * 60)
print(f'  \U0001f916 XGBOOST v1.0 — 24 features, trained through Round {{RND-1}}')
print('=' * 60)
print(f'  Training rows: {{len(feat[feat.race_seq < target_seq]):,}}')
print(f'  Leave-one-out MAE: {{loo_mae:.2f}} positions (honest)')
print(f'  Residual StdDev:   {{residual_std:.2f}}')

imp = sorted(zip(f1lib.FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print('\\n  Top features:')
for f, s in imp[:8]:
    print(f'    {{f:25s}} {{s:.3f}}')
"""

# shared weather resolution, used by both the qualifying and pre-qualifying cells
WEATHER_BLOCK = """import fastf1
# Race-day weather. WEATHER is 'auto' (Open-Meteo forecast), 'temp,humidity,rain%'
# (manual), or '' (use the stage's default conditions).
forecast = None
if WEATHER == 'auto':
    _ev = fastf1.get_event(2026, RND)
    _race_date = str(pd.to_datetime(_ev['EventDate']).date())
    forecast = f1lib.fetch_forecast(EVENT, _race_date)
    if forecast:
        print(f"  \U0001f324️ Forecast (Open-Meteo, {_race_date}): {forecast['air_temp']:.0f}°C, "
              f"{forecast['humidity']:.0f}% RH, {forecast['rain_prob']:.0f}% rain")
    else:
        print('  ⚠️ Forecast unavailable — falling back to default conditions.')
elif WEATHER:
    _p = [float(x) for x in WEATHER.split(',')]
    forecast = {'air_temp': _p[0], 'humidity': _p[1], 'rain_prob': _p[2]}
    print(f"  \U0001f324️ Manual forecast: {forecast['air_temp']:.0f}°C, "
          f"{forecast['humidity']:.0f}% RH, {forecast['rain_prob']:.0f}% rain")
"""

CONDITIONS_PRINT = """
_cond = 'WET' if wk['rain'] else 'DRY'
print(f"\\n  \U0001f321️ Race conditions: {wk['temp']:.1f}°C, {wk['humidity']:.0f}% humidity, "
      f"{wk['rain_prob']:.0f}% rain -> {_cond}")
if 0 < wk['rain_prob'] < 100:
    print(f"  Predictions are a {wk['rain_prob']:.0f}% wet / {100-wk['rain_prob']:.0f}% dry blend.")
"""

QUALI_MD = f"""## 1. Qualifying & Blended Car Pace

Grid position is the model's single strongest predictor. This weekend's actual qualifying result sets the starting grid, and the team-level pace is a 30/70 blend of season-to-date form and qualifying performance. Live {EVENT} sector deltas from qualifying replace the historical track profile."""

QUALI = WEATHER_BLOCK + """
wk = f1lib.weekend(2026, EVENT, df, {'pace': cp_season, 'speed': speeds_2026},
                   pre_race=PRE_RACE, forecast=forecast)
wk['race_name'] = EVENT

grid_sorted = sorted(wk['grid'].items(), key=lambda x: x[1])
print('=' * 70)
print(f'  \U0001f3c1 QUALIFYING GRID — 2026 {EVENT} GP')
print('=' * 70)
print(f"\\n  {'Pos':>3s}  {'Driver':22s}  {'Team':18s}  {'Blend Pace':>10s}")
print('  ' + '-' * 60)
for name, pos in grid_sorted:
    t = wk['team_map'].get(name, '')
    print(f"  P{pos:>2d}  {name:22s}  {t:18s}  P{wk['car_pace'].get(t, 15):>6.1f}")
""" + CONDITIONS_PRINT

QUALI_PREQ_MD = f"""## 1. Season Pace & Estimated Grid

Qualifying hasn't run yet, so there is no real grid. The model estimates the starting order from team pace (season form blended with any practice that has run) plus each driver's season form, and uses {EVENT}'s historical sector profile. **The grid below is an estimate** — it will be replaced by the real qualifying result when this page is regenerated on Saturday, which is the single biggest accuracy jump of the weekend."""

QUALI_PREQ = WEATHER_BLOCK + """
wk = f1lib.weekend_preq(2026, EVENT, df, {'pace': cp_season, 'speed': speeds_2026}, sec,
                        forecast=forecast)
wk['race_name'] = EVENT

grid_sorted = sorted(wk['grid'].items(), key=lambda x: x[1])
print('=' * 70)
print(f'  \U0001f4ca ESTIMATED GRID — 2026 {EVENT} GP  [stage: {wk["stage"]}]')
print('=' * 70)
print(f"\\n  {'Est':>3s}  {'Driver':22s}  {'Team':18s}  {'Blend Pace':>10s}")
print('  ' + '-' * 60)
for name, pos in grid_sorted:
    t = wk['team_map'].get(name, '')
    print(f"  P{pos:>2d}  {name:22s}  {t:18s}  P{wk['car_pace'].get(t, 15):>6.1f}")
print("\\n  Grid is ESTIMATED from pace + form; real qualifying replaces it Saturday.")
""" + CONDITIONS_PRINT

PRED_MD = """## 2. Predicted Finish

Each driver's predicted finishing position, with the inputs that drove it: grid (actual qualifying), car (blended team pace), and the position change the model expects relative to the grid."""

PRED = """preds = f1lib.predict(df, feat, model, lk, wk)

print('=' * 80)
print(f'  \U0001f3c1 PREDICTED FINISH — 2026 {EVENT} GP')
print('=' * 80)
print(f"\\n  {'Pos':>3s}  {'Driver':22s}  {'Team':18s}  {'Grid':>4s}  {'Pred':>5s}  {'ΔGrid':>5s}")
print('  ' + '-' * 70)
for p in preds:
    d = p['Grid'] - p['Rank']
    ds = f'+{d}' if d > 0 else (f'{d}' if d < 0 else '=')
    print(f"  P{p['Rank']:>2d}  {p['Driver']:22s}  {p['Team']:18s}  P{p['Grid']:>2.0f}  P{p['Predicted']:>4.1f}  {ds:>5s}")
"""

SIM_MD = """## 3. Podium Probabilities (10,000 Simulated Races)

Rather than a single outcome, the model simulates the race 10,000 times with realistic randomness — and now models **retirements**: each driver can DNF with their season failure rate, which promotes everyone behind them. That matters most at high-attrition tracks, where a front-runner's mechanical failure reshuffles the order. Add a 33% safety-car chance and grid-dependent chaos (the midfield is messier than the front row), and the result is a probability for each driver."""

SIM = """sims = f1lib.simulate(preds, residual_std)
n = sims['n']
print(f"  Modeled retirements: {sum(sims['dnf'].values())/n:.1f} per race (driven by each driver's season DNF rate)\\n")
prob = [{'Driver': p['Driver'], 'Team': p['Team'], 'Grid': p['Grid'],
         'Win': sims['win'][p['Driver']]/n*100, 'Podium': sims['podium'][p['Driver']]/n*100,
         'Top5': sims['top5'][p['Driver']]/n*100, 'Top10': sims['top10'][p['Driver']]/n*100}
        for p in preds]
prob_df = pd.DataFrame(prob).sort_values('Win', ascending=False)

print('=' * 80)
print(f'  \U0001f3b2 PODIUM PROBABILITIES — 2026 {EVENT} GP')
print('=' * 80)
print(f"\\n  {'Driver':22s}  {'Team':18s}  {'Grid':>4s}  {'Win':>6s}  {'Podium':>7s}  {'Top5':>6s}")
print('  ' + '-' * 72)
for _, r in prob_df.head(12).iterrows():
    bar = '█' * int(r['Win']/2)
    print(f"  {r['Driver']:22s}  {r['Team']:18s}  P{r['Grid']:>2.0f}  {r['Win']:>5.1f}%  {r['Podium']:>6.1f}%  {r['Top5']:>5.1f}%  {bar}")

fig, ax = plt.subplots(figsize=(10, 7))
t12 = prob_df.head(12).sort_values('Podium')
colors = ['#ff006e' if p > 40 else '#8338ec' if p > 15 else '#3a86ff' if p > 5 else '#06ffa5' for p in t12['Podium']]
bars = ax.barh(t12['Driver'] + '  (' + t12['Team'] + ')', t12['Podium'], color=colors)
for b, v in zip(bars, t12['Podium']):
    ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=10)
ax.set_xlabel('Podium Probability (%)')
ax.set_title(f'2026 {EVENT} GP — Who Makes the Podium?', fontweight='bold')
plt.tight_layout(); plt.show()
"""

GRADE_MD = """## 4. Race Results & Model Accuracy

The race has run — time to grade the model. Predicted order against the official classification: position-by-position error, podium / top-5 / top-10 accuracy, and the standout calls and misses. These numbers feed the season scoreboard."""

GRADE = """g = f1lib.grade(preds, wk['race_results'])

print('=' * 85)
print(f'  \U0001f3c1 OFFICIAL RESULTS vs MODEL — 2026 {EVENT} GP')
print('=' * 85)
print(f"\\n  {'Fin':>3s}  {'Driver':22s}  {'Team':18s}  {'Pred':>4s}  {'Err':>4s}  {'Status'}")
print('  ' + '-' * 78)
for r in sorted(g['rows'], key=lambda r: (pd.isna(r['Actual']), r['Actual'] if pd.notna(r['Actual']) else 99)):
    es = f"{r['Err']:.0f}" if r['Err'] is not None else 'DNF'
    hit = '\U0001f3af' if r['Err'] == 0 else ''
    fin = f"P{r['Actual']:>2.0f}" if pd.notna(r['Actual']) else ' - '
    print(f"  {fin}  {r['Driver']:22s}  {r['Team']:18s}  P{r['PredRank']:>2d}  {es:>4s}  {r['Status']} {hit}")

print(f"\\n  \U0001f4ca SCORECARD")
print('  ' + '-' * 60)
print(f"  Predicted winner:  {g['winner_pred']}")
print(f"  Actual winner:     {g['winner_actual']}  {'✅' if g['winner_hit'] else '❌'}")
print(f"  Podium accuracy:   {g['podium']}/3")
print(f"  Top 5 accuracy:    {g['top5']}/5   (grid baseline: {g['grid_top5']}/5)")
print(f"  Top 10 accuracy:   {g['top10']}/10  (grid baseline: {g['grid_top10']}/10)")
print(f"  MAE (classified):  {g['mae']:.2f} positions  (validation expected {loo_mae:.2f})")
edge = g['model_edge']
verdict = 'model beats grid' if edge > 0.05 else ('grid beats model' if edge < -0.05 else 'tie')
print(f"\\n  \U0001f3c1 vs GRID BASELINE (does the model beat 'finish = qualifying order'?)")
print(f"  Grid-copy MAE:     {g['grid_mae']:.2f} positions")
print(f"  Model edge:        {edge:+.2f}  ->  {verdict}")

beats = sorted(g['clean'], key=lambda r: r['Err'])[:3]
misses = sorted(g['clean'], key=lambda r: -r['Err'])[:3]
print(f"\\n  \U0001f3af Best calls:   " + ', '.join(f"{r['Driver']} (P{r['PredRank']}→P{r['Actual']:.0f})" for r in beats))
print(f"  \U0001f4a5 Worst misses: " + ', '.join(f"{r['Driver']} (pred P{r['PredRank']}, fin P{r['Actual']:.0f})" for r in misses))
dnfs = [r for r in g['rows'] if r['DNF']]
if dnfs:
    print(f"  \U0001f527 DNFs: " + ', '.join(f"{r['Driver']}" for r in dnfs))

fig, ax = plt.subplots(figsize=(8, 8))
xs = [r['PredRank'] for r in g['clean']]; ys = [r['Actual'] for r in g['clean']]
ax.scatter(xs, ys, s=80, c='#06ffa5', alpha=0.8, zorder=3)
for r in g['clean']:
    ax.annotate(r['Driver'].split()[-1], (r['PredRank'], r['Actual']),
                textcoords='offset points', xytext=(6, 4), fontsize=8)
lim = max(max(xs), max(ys)) + 1
ax.plot([0, lim], [0, lim], color='#ff006e', linewidth=1, linestyle='--', alpha=0.7, label='Perfect prediction')
ax.set_xlabel('Predicted Position'); ax.set_ylabel('Actual Position')
ax.set_title(f"2026 {EVENT} GP — Predicted vs Actual (MAE {g['mae']:.2f})", fontweight='bold')
ax.legend(); plt.tight_layout(); plt.show()
"""


if PRE_QUALI:
    CLOSING_MD = f"""## 4. What Happens Next

*This is the pre-qualifying baseline for the {EVENT} GP. It refreshes through the weekend:*
- *Friday (after practice): rerun `--pre-qualifying` to fold in FP pace.*
- *Saturday (after qualifying): rerun `--pre-race --weather auto` for the real grid — the biggest accuracy jump.*
- *After the race: run `update_data.py` then rerun with no flag to add the official results and accuracy.*"""
else:
    CLOSING_MD = f"""## 4. Race Results & Model Accuracy

*Pending — the {EVENT} GP has not run yet. After the race, regenerate this notebook without `--pre-race` (the round will be in the data by then) to add the official results, the position-by-position accuracy, and the grid-baseline comparison.*"""


def cell(t, src):
    c = {'cell_type': t, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if t == 'code':
        c.update({'execution_count': None, 'outputs': []})
    return c


_quali_md = QUALI_PREQ_MD if PRE_QUALI else QUALI_MD
_quali = QUALI_PREQ if PRE_QUALI else QUALI
cells = [
    cell('markdown', TITLE_MD),
    cell('code', SETUP),
    cell('markdown', _quali_md), cell('code', _quali),
    cell('markdown', PRED_MD), cell('code', PRED),
    cell('markdown', SIM_MD), cell('code', SIM),
]
# the grading section requires the race to have run
cells += ([cell('markdown', GRADE_MD), cell('code', GRADE)]
          if not (PRE_RACE or PRE_QUALI) else [cell('markdown', CLOSING_MD)])

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}

out = f'notebooks/2026-R{RND:02d}-{SLUG}.ipynb'
json.dump(nb, open(out, 'w'), indent=1)
print('wrote', out)
