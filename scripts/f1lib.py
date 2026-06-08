"""Shared F1 prediction model — v1.0+Q logic, parameterized by track.

Used by the per-race retrospective notebooks (R04+). The Japanese GP notebook
(R03) keeps its own inline showcase code with the full FP1/FP2/FP3 blend.

Honest, forward-looking design: to predict round N, the model trains only on
races BEFORE round N (race_seq < target) plus all 2023-2025 history, with 2026
weighted 10x. The grid comes from round N's actual qualifying, and live
qualifying sector deltas replace historical track sectors. Nothing from round
N's race enters the model — so comparing the prediction to the actual result is
a fair test.
"""
import warnings
import numpy as np
import pandas as pd
import fastf1
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

FEATURE_COLS = ['grid_position', 'car_pace', 'car_speed', 'driver_avg_finish',
                'positions_gained_avg', 'teammate_delta', 'dnf_rate', 'consistency',
                'track_experience', 'wet_skill', 'first_lap_gain', 'tire_degradation',
                'momentum', 'quali_race_gap', 'sector1_delta', 'sector2_delta',
                'sector3_delta', 'air_temp', 'humidity', 'had_rain', 'avg_pit_stops',
                'team_pit_strategy', 'team_changed', 'is_2026']

TEAM_CHANGERS = ['Lewis Hamilton', 'Oliver Bearman', 'Isack Hadjar', 'Franco Colapinto',
                 'Gabriel Bortoleto', 'Sergio Perez', 'Valtteri Bottas', 'Arvid Lindblad']

RACE_PTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def enable_cache(path='../data/cache'):
    fastf1.Cache.enable_cache(path)
    import logging
    logging.getLogger('fastf1').setLevel(logging.ERROR)


def load_data(data_dir='../data'):
    """Load CSVs and build the driver/team lookups the feature builder needs."""
    df = pd.read_csv(f'{data_dir}/all_results.csv')
    weather = pd.read_csv(f'{data_dir}/weather.csv')
    lap_stats = pd.read_csv(f'{data_dir}/lap_stats.csv')
    advanced = pd.read_csv(f'{data_dir}/advanced_features.csv')

    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['DNF'] = df['Status'].apply(
        lambda x: 0 if x == 'Finished' or (isinstance(x, str) and 'Lap' in x) else 1)
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)
    df = df.merge(weather[['Year', 'Round', 'avg_air_temp', 'avg_track_temp',
                           'avg_humidity', 'avg_wind_speed', 'had_rain', 'rain_pct']],
                  on=['Year', 'Round'], how='left')

    # global race ordering
    df['race_id'] = df['Year'].astype(str) + '_' + df['Round'].astype(str)
    ro = df[['race_id', 'Year', 'Round']].drop_duplicates().sort_values(
        ['Year', 'Round']).reset_index(drop=True)
    ro['race_seq'] = range(len(ro))
    df = df.merge(ro[['race_id', 'race_seq']], on='race_id', how='left')

    lk = {
        'first_lap_avg': advanced.groupby('FullName')['first_lap_gain'].mean().to_dict(),
        'tire_deg_avg': advanced.groupby('FullName')['tire_degradation'].mean().to_dict(),
        'quali_race_gap': advanced.groupby('FullName')['quali_race_gap'].mean().to_dict(),
        'driver_consistency': lap_stats.groupby('FullName')['lap_time_std'].mean().to_dict(),
        'driver_pits': lap_stats.groupby('FullName')['n_pits'].mean().to_dict(),
    }
    # momentum: slope of last 5 finishes
    mom = {}
    for d in advanced['FullName'].unique():
        dh = advanced[advanced['FullName'] == d].sort_values(['Year', 'Round']).tail(5)
        pos = dh['finish'].dropna().values
        mom[d] = np.polyfit(range(len(pos)), pos, 1)[0] if len(pos) >= 3 else 0
    lk['momentum'] = mom
    return df, weather, lap_stats, advanced, lk


def track_sector_deltas(sectors_csv):
    """Per-driver sector-time deltas vs field mean, from a track's history."""
    s = pd.read_csv(sectors_csv)
    ds = s.groupby('FullName')[['avg_s1', 'avg_s2', 'avg_s3']].mean()
    for c in ['avg_s1', 'avg_s2', 'avg_s3']:
        ds[f'{c}_d'] = ds[c] - ds[c].mean()
    return (ds['avg_s1_d'].to_dict(), ds['avg_s2_d'].to_dict(), ds['avg_s3_d'].to_dict())


def season_pace(df, year, before_seq):
    """2026 team pace (avg finish) and speed using only races before target."""
    season = df[(df['Year'] == year) & (df['race_seq'] < before_seq)]
    cp = season.groupby('TeamName')['Position'].mean().to_dict()
    return cp


def engineer_features(df, lap_stats, lk, sec_deltas, target_seq, car_pace_season):
    """Build the 24-feature matrix using only rows up to and including target_seq.

    For 2026 rows, car_pace = season-to-date team avg finish (no leakage).
    Sector deltas come from the target track's driver history.
    """
    s1d, s2d, s3d = sec_deltas
    team_pits_2026 = lap_stats[lap_stats['Year'] == 2026].groupby('Team')['n_pits'].mean().to_dict()
    speeds_2026 = lap_stats[lap_stats['Year'] == 2026].groupby('Team')['avg_speed_st'].mean().to_dict()

    rows = []
    for rs in sorted(df[df['race_seq'] <= target_seq]['race_seq'].unique()):
        rd = df[df['race_seq'] == rs]
        for _, row in rd.iterrows():
            d, t, rn = row['FullName'], row['TeamName'], row['RaceName']
            h = df[(df['FullName'] == d) & (df['race_seq'] < rs)]
            ht = df[(df['TeamName'] == t) & (df['race_seq'] < rs)]
            l5, l10 = h.tail(5), h.tail(10)

            tm = df[(df['TeamName'] == t) & (df['FullName'] != d) & (df['race_seq'] < rs)]
            tmd = []
            for s in h.tail(5)['race_seq'].values:
                my = h[h['race_seq'] == s]['Position'].values
                t2 = tm[tm['race_seq'] == s]['Position'].values
                if len(my) and len(t2) and not np.isnan(my[0]) and not np.isnan(t2[0]):
                    tmd.append(t2[0] - my[0])

            wet = h[h['had_rain'] == True]
            wg = (wet['GridPosition'] - wet['Position']).mean() if len(wet) else 0

            is_changer = d in TEAM_CHANGERS and row['Year'] == 2026
            mom_window = h[h['TeamName'] == t].tail(5) if is_changer else l5
            mp = mom_window['Position'].dropna().values
            try:
                mom_val = np.polyfit(range(len(mp)), mp, 1)[0] if len(mp) >= 3 else 0
            except Exception:
                mom_val = 0

            if row['Year'] == 2026:
                cp = car_pace_season.get(t, 15)
            else:
                cp = ht.tail(5)['Position'].mean() if len(ht) else 12

            rows.append({
                'race_seq': rs, 'Year': row['Year'], 'Round': row['Round'],
                'RaceName': rn, 'FullName': d, 'TeamName': t,
                'grid_position': row['GridPosition'], 'finish_position': row['Position'],
                'car_pace': cp,
                'car_speed': speeds_2026.get(t, 290) if row['Year'] == 2026 else 290,
                'driver_avg_finish': l5['Position'].mean() if len(l5) else 12,
                'positions_gained_avg': (l5['GridPosition'] - l5['Position']).mean() if len(l5) else 0,
                'teammate_delta': np.mean(tmd) if len(tmd) else 0,
                'dnf_rate': l10['DNF'].mean() if len(l10) else 0.1,
                'consistency': lk['driver_consistency'].get(d, 1.5),
                'track_experience': len(h[h['RaceName'] == rn]),
                'wet_skill': wg,
                'first_lap_gain': lk['first_lap_avg'].get(d, 0),
                'tire_degradation': lk['tire_deg_avg'].get(d, 0.03),
                'momentum': mom_val,
                'avg_pit_stops': lk['driver_pits'].get(d, 1.5),
                'team_pit_strategy': team_pits_2026.get(t, 1.5) if row['Year'] == 2026 else 1.5,
                'quali_race_gap': lk['quali_race_gap'].get(d, 0),
                'sector1_delta': s1d.get(d, 0), 'sector2_delta': s2d.get(d, 0),
                'sector3_delta': s3d.get(d, 0),
                'air_temp': row.get('avg_air_temp', 22), 'humidity': row.get('avg_humidity', 50),
                'had_rain': 1 if row.get('had_rain', False) else 0,
                'team_changed': 1 if (row['Year'] == 2026 and d in TEAM_CHANGERS) else 0,
                'is_2026': 1 if row['Year'] == 2026 else 0,
                'DNF': row['DNF'],
            })
    return pd.DataFrame(rows).dropna(subset=['finish_position', 'grid_position'])


def train(feat_df, target_seq):
    """Train on finishers BEFORE the target race (2026 weighted 10x).

    Returns (model, loo_mae, residual_std). LOO MAE is leave-one-out across the
    2026 rounds present in the training window — honest because each held-out
    round is predicted by a model that never saw it.
    """
    train_df = feat_df[(feat_df['race_seq'] < target_seq) & (feat_df['DNF'] == 0)]
    X, y = train_df[FEATURE_COLS].fillna(0), train_df['finish_position']
    w = np.where(train_df['Year'] == 2026, 10, 1).astype(float)
    model = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y, sample_weight=w)

    loo_preds, loo_actual = [], []
    rounds_2026 = sorted(train_df[train_df['Year'] == 2026]['Round'].unique())
    for ho in rounds_2026:
        hold = train_df[(train_df['Year'] == 2026) & (train_df['Round'] == ho)]
        tr = train_df[~((train_df['Year'] == 2026) & (train_df['Round'] == ho))]
        if len(hold) == 0 or len(tr) == 0:
            continue
        Xt, yt = tr[FEATURE_COLS].fillna(0), tr['finish_position']
        wt = np.where(tr['Year'] == 2026, 10, 1).astype(float)
        m = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(Xt, yt, sample_weight=wt)
        loo_preds.extend(m.predict(hold[FEATURE_COLS].fillna(0)))
        loo_actual.extend(hold['finish_position'].values)

    if loo_actual:
        loo_mae = mean_absolute_error(loo_actual, loo_preds)
        residual_std = float(np.std(np.array(loo_actual) - np.array(loo_preds)))
    else:
        loo_mae, residual_std = 2.4, 2.4
    return model, loo_mae, residual_std


def weekend(year, event, df, season_speed):
    """Pull a completed weekend's qualifying for grid + live sector deltas + blended
    car pace, plus race weather. Falls back gracefully if practice is missing.

    Returns dict: grid, sector_s1/2/3, car_pace, car_speed, temp, humidity, rain.
    """
    season_pace_dict = season_speed['pace']
    season_spd_dict = season_speed['speed']

    quali = fastf1.get_session(year, event, 'Qualifying')
    quali.load()
    qres = quali.results
    grid = {r['FullName']: int(r['Position']) for _, r in qres.iterrows() if pd.notna(r['Position'])}
    team_map = dict(zip(qres['FullName'], qres['TeamName']))
    abbr_map = dict(zip(qres['Abbreviation'], qres['FullName']))

    # live qualifying sector deltas + top speed
    qbest = quali.laps.sort_values('LapTime').groupby('Driver').first()
    qs1, qs2, qs3, qspd = {}, {}, {}, {}
    for drv, row in qbest.iterrows():
        name = abbr_map.get(drv, drv)
        for store, col in ((qs1, 'Sector1Time'), (qs2, 'Sector2Time'), (qs3, 'Sector3Time')):
            if pd.notna(row[col]):
                store[name] = row[col].total_seconds()
        if pd.notna(row.get('SpeedST', np.nan)):
            qspd[name] = row['SpeedST']
    m1, m2, m3 = (np.mean(list(qs1.values())), np.mean(list(qs2.values())), np.mean(list(qs3.values())))
    sector_s1 = {d: t - m1 for d, t in qs1.items()}
    sector_s2 = {d: t - m2 for d, t in qs2.items()}
    sector_s3 = {d: t - m3 for d, t in qs3.items()}

    # qualifying-derived team pace
    qteam = {}
    for name, pos in grid.items():
        qteam.setdefault(team_map.get(name, ''), []).append(pos)
    quali_team_pos = {t: np.mean(v) for t, v in qteam.items()}

    # blend season pace with qualifying (practice sessions vary by weekend; use
    # the robust 2-way blend: season + qualifying, qualifying dominant)
    SEASON_W, Q_W = 0.30, 0.70
    car_pace, car_speed = {}, {}
    for t in set(list(season_pace_dict.keys()) + list(quali_team_pos.keys())):
        s = season_pace_dict.get(t, quali_team_pos.get(t, 15))
        qp = quali_team_pos.get(t, s)
        car_pace[t] = s * SEASON_W + qp * Q_W
        s_spd = season_spd_dict.get(t, 290)
        q_spd = np.mean([qspd.get(d, s_spd) for d in grid if team_map.get(d) == t and d in qspd]) \
            if any(team_map.get(d) == t for d in qspd) else s_spd
        car_speed[t] = s_spd * SEASON_W + q_spd * Q_W

    # race weather (actual conditions)
    race = fastf1.get_session(year, event, 'R')
    race.load()
    w = race.weather_data
    return {
        'grid': grid, 'team_map': team_map,
        'sector_s1': sector_s1, 'sector_s2': sector_s2, 'sector_s3': sector_s3,
        'car_pace': car_pace, 'car_speed': car_speed,
        'temp': w['AirTemp'].mean(), 'humidity': w['Humidity'].mean(),
        'rain': 1 if w['Rainfall'].any() else 0,
        'race_results': race.results, 'quali_results': qres,
    }


def predict(df, feat_df, model, lk, wk):
    """Predict the target race for the 2026 grid using actual qualifying grid +
    live sectors + blended pace. Returns a sorted list of dicts (best first)."""
    latest_2026 = df[df['Year'] == 2026]
    preds = []
    for d in wk['grid']:
        dh = feat_df[feat_df['FullName'] == d].sort_values('race_seq')
        t = wk['team_map'].get(d)
        if t is None and len(dh):
            t = dh.iloc[-1]['TeamName']
        base = {c: (dh.iloc[-1].get(c, 0) if len(dh) else 0) for c in FEATURE_COLS}
        te = len(dh[dh['RaceName'].str.contains(wk.get('race_name', ''), case=False, na=False)]) \
            if 'race_name' in wk else 0
        base.update({
            'grid_position': wk['grid'][d],
            'car_pace': wk['car_pace'].get(t, 15),
            'car_speed': wk['car_speed'].get(t, 290),
            'track_experience': te,
            'air_temp': wk['temp'], 'humidity': wk['humidity'], 'had_rain': wk['rain'],
            'is_2026': 1,
            'first_lap_gain': lk['first_lap_avg'].get(d, 0),
            'tire_degradation': lk['tire_deg_avg'].get(d, 0.03),
            'momentum': lk['momentum'].get(d, 0),
            'sector1_delta': wk['sector_s1'].get(d, 0),
            'sector2_delta': wk['sector_s2'].get(d, 0),
            'sector3_delta': wk['sector_s3'].get(d, 0),
        })
        X = pd.DataFrame([base])[FEATURE_COLS].fillna(0)
        preds.append({'Driver': d, 'Team': t, 'Grid': wk['grid'][d],
                      'Car': wk['car_pace'].get(t, 15), 'Predicted': float(model.predict(X)[0]),
                      'Features': base})
    preds.sort(key=lambda x: x['Predicted'])
    for i, p in enumerate(preds):
        p['Rank'] = i + 1
    return preds


def simulate(preds, residual_std, n_sims=10000, sc_prob=0.33, seed=42):
    """Monte-Carlo race sim with grid-dependent variance + safety car."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)

    def gv(g):
        if g <= 3:
            return residual_std * 0.7
        if g <= 7:
            return residual_std * 0.9
        if g <= 15:
            return residual_std * 1.2
        return residual_std * 1.0

    pt = defaultdict(list)
    wc = defaultdict(int); pc = defaultdict(int); t5 = defaultdict(int); t10 = defaultdict(int)
    for _ in range(n_sims):
        sc = rng.random() < sc_prob
        sim = []
        for p in preds:
            pred = p['Predicted'] + rng.normal(0, gv(p['Grid']))
            if sc and p['Grid'] > 5:
                pred -= rng.uniform(0, 1.5)
            sim.append((p['Driver'], pred))
        sim.sort(key=lambda x: x[1])
        for pos, (d, _) in enumerate(sim):
            pt[d].append(pos + 1)
            if pos == 0: wc[d] += 1
            if pos < 3: pc[d] += 1
            if pos < 5: t5[d] += 1
            if pos < 10: t10[d] += 1
    return {'pt': pt, 'win': wc, 'podium': pc, 'top5': t5, 'top10': t10, 'n': n_sims}


def grade(preds, race_results):
    """Compare predictions to official classification. Returns a scorecard dict."""
    actual_pos, actual_status = {}, {}
    for _, r in race_results.iterrows():
        actual_pos[r['FullName']] = r['Position']
        actual_status[r['FullName']] = str(r['Status'])
    pred_lookup = {p['Driver']: p for p in preds}

    rows = []
    for name, pos in actual_pos.items():
        p = pred_lookup.get(name)
        if p is None:
            continue
        status = actual_status[name]
        dnf = not (status == 'Finished' or 'Lap' in status)
        err = abs(p['Rank'] - pos) if (not dnf and pd.notna(pos)) else None
        rows.append({'Driver': name, 'Team': p['Team'], 'Grid': p['Grid'],
                     'PredRank': p['Rank'], 'Actual': pos, 'DNF': dnf, 'Err': err, 'Status': status})

    clean = [r for r in rows if r['Err'] is not None]
    actual_order = [n for n, _ in sorted(((n, p) for n, p in actual_pos.items() if pd.notna(p)),
                                         key=lambda x: x[1])]
    pred_order = [p['Driver'] for p in preds]
    return {
        'rows': rows, 'clean': clean,
        'mae': float(np.mean([r['Err'] for r in clean])) if clean else None,
        'winner_pred': pred_order[0], 'winner_actual': actual_order[0],
        'winner_hit': pred_order[0] == actual_order[0],
        'podium': len(set(pred_order[:3]) & set(actual_order[:3])),
        'top5': len(set(pred_order[:5]) & set(actual_order[:5])),
        'top10': len(set(pred_order[:10]) & set(actual_order[:10])),
        'pred_order': pred_order, 'actual_order': actual_order,
    }
