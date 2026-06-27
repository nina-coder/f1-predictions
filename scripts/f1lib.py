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


# Circuit coordinates + local timezone for race-day weather lookups. Keyed by the
# event name passed to weekend(). Add a row when you add a track.
CIRCUIT_COORDS = {
    'Japan': (34.843, 136.541, 'Asia/Tokyo'),
    'Miami': (25.958, -80.239, 'America/New_York'),
    'Canada': (45.500, -73.522, 'America/Toronto'),
    'Monaco': (43.735, 7.421, 'Europe/Monaco'),
    'Barcelona': (41.570, 2.261, 'Europe/Madrid'),
    'Austria': (47.220, 14.765, 'Europe/Vienna'),
}


def fetch_forecast(event, date_str, hour=15):
    """Race-day forecast from Open-Meteo (no API key). date_str is 'YYYY-MM-DD',
    hour is local race start (24h). Returns {air_temp, humidity, rain_prob} where
    rain_prob is 0-100, or None if the circuit is unmapped or the call fails."""
    import json
    from urllib.request import urlopen
    from urllib.parse import urlencode
    if event not in CIRCUIT_COORDS:
        return None
    lat, lon, tz = CIRCUIT_COORDS[event]
    q = urlencode({'latitude': lat, 'longitude': lon, 'timezone': tz,
                   'hourly': 'temperature_2m,relative_humidity_2m,precipitation_probability',
                   'start_date': date_str, 'end_date': date_str})
    try:
        with urlopen(f'https://api.open-meteo.com/v1/forecast?{q}', timeout=20) as r:
            h = json.load(r).get('hourly', {})
        target = f'{date_str}T{hour:02d}:00'
        i = h['time'].index(target) if target in h['time'] else len(h['time']) // 2
        return {'air_temp': h['temperature_2m'][i],
                'humidity': h['relative_humidity_2m'][i],
                'rain_prob': h['precipitation_probability'][i]}
    except Exception:
        return None


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


def weekend(year, event, df, season_speed, pre_race=False, forecast=None):
    """Pull a weekend's qualifying for grid + live sector deltas + blended car pace.

    Retrospective mode (default) also loads the race for actual weather + results.
    pre_race=True is for a Saturday run before the race has happened: it skips the
    race session, takes conditions from qualifying, and returns race_results=None.

    forecast (optional) = {air_temp, humidity, rain_prob (0-100)} overrides the
    race-day conditions — use it pre-race so the prediction reflects the actual
    Sunday forecast rather than qualifying conditions. rain_prob drives a wet/dry
    blend in predict().

    Returns dict: grid, sector_s1/2/3, car_pace, car_speed, temp, humidity, rain,
    rain_prob, race_results (None if pre_race), quali_results.
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

    if pre_race:
        # race hasn't run — use qualifying conditions as the best available proxy
        qw = quali.weather_data
        temp, humidity, rain, race_results = (
            qw['AirTemp'].mean(), qw['Humidity'].mean(), 1 if qw['Rainfall'].any() else 0, None)
    else:
        race = fastf1.get_session(year, event, 'R')
        race.load()
        w = race.weather_data
        temp, humidity, rain, race_results = (
            w['AirTemp'].mean(), w['Humidity'].mean(), 1 if w['Rainfall'].any() else 0, race.results)
    rain_prob = 100.0 if rain else 0.0
    if forecast:
        # an explicit race-day forecast overrides the proxy conditions
        temp, humidity = forecast['air_temp'], forecast['humidity']
        rain_prob = float(forecast['rain_prob'])
        rain = 1 if rain_prob >= 50 else 0
    return {
        'grid': grid, 'team_map': team_map,
        'sector_s1': sector_s1, 'sector_s2': sector_s2, 'sector_s3': sector_s3,
        'car_pace': car_pace, 'car_speed': car_speed,
        'temp': temp, 'humidity': humidity, 'rain': rain, 'rain_prob': rain_prob,
        'race_results': race_results, 'quali_results': qres,
    }


def _try_session(year, event, sess):
    """Load a session, returning it only if it has lap data; else None."""
    try:
        s = fastf1.get_session(year, event, sess)
        s.load()
        if s.laps is None or len(s.laps) == 0:
            return None
        return s
    except Exception:
        return None


def _practice_team_pos(session):
    """Team -> grid-position proxy from a practice session's best laps."""
    laps = session.laps
    best = laps.groupby('Driver')['LapTime'].min().dropna()
    name_map = dict(zip(session.results['Abbreviation'], session.results['FullName']))
    team_map = dict(zip(session.results['FullName'], session.results['TeamName']))
    gaps = {}
    for drv, lt in best.items():
        team = team_map.get(name_map.get(drv, drv))
        if team:
            gaps.setdefault(team, []).append(lt.total_seconds())
    order = sorted(gaps, key=lambda t: np.mean(gaps[t]))
    return {t: i * 2 + 1.5 for i, t in enumerate(order)}, team_map


def weekend_preq(year, event, df, season_speed, sec_deltas, forecast=None):
    """Early-weekend prediction with NO qualifying yet.

    Replaces the original notebook's progressive flow. Team pace is season form
    blended with whatever practice (FP1/FP2/FP3) has run; the starting grid is
    *estimated* from that pace plus driver season form; sector deltas come from
    the track's history (not live). As FP sessions appear the blend tightens, and
    once qualifying exists you switch to weekend() for the real grid.

    Returns the same dict shape as weekend(), plus 'stage' and 'estimated_grid',
    with race_results=None and quali_results=None.
    """
    s1d, s2d, s3d = sec_deltas
    season_pace_dict, season_spd_dict = season_speed['pace'], season_speed['speed']

    # current lineup + teams from the most recent completed 2026 race
    season = df[(df['Year'] == 2026) & df['Position'].notna()]
    last_seq = season['race_seq'].max()
    entry = df[(df['Year'] == 2026) & (df['race_seq'] == last_seq)]
    team_map = dict(zip(entry['FullName'], entry['TeamName']))
    drivers = list(team_map)
    season_avg_finish = season.groupby('FullName')['Position'].mean().to_dict()

    # blend season pace with available practice sessions (same weighting logic the
    # Japan notebook used as each session arrived)
    practice = [(s, _try_session(year, event, s)) for s in ('FP1', 'FP2', 'FP3')]
    practice = [(name, _practice_team_pos(sess)) for name, sess in practice if sess is not None]
    if not practice:
        weights, stage = {'season': 1.0}, 'pre-practice (season form)'
    elif len(practice) == 1:
        weights, stage = {'season': 0.6, 0: 0.4}, 'post-FP1'
    elif len(practice) == 2:
        weights, stage = {'season': 0.3, 0: 0.3, 1: 0.4}, 'post-FP2'
    else:
        weights, stage = {'season': 0.2, 0: 0.2, 1: 0.3, 2: 0.3}, 'post-FP3'
    # practice may carry the freshest team map (driver swaps)
    for _, (_, tmap) in practice:
        team_map.update({d: t for d, t in tmap.items() if d in team_map})

    teams = set(season_pace_dict) | {team_map[d] for d in drivers if d in team_map}
    car_pace = {}
    for t in teams:
        val = weights['season'] * season_pace_dict.get(t, 15)
        for i, (_, (tpos, _)) in enumerate(practice):
            val += weights[i] * tpos.get(t, season_pace_dict.get(t, 15))
        car_pace[t] = val
    car_speed = {t: season_spd_dict.get(t, 290) for t in teams}

    # estimate the grid: 70% car pace, 30% driver season form, ranked 1..N
    proxy = {}
    for d in drivers:
        t = team_map.get(d)
        tp = car_pace.get(t, 15)
        proxy[d] = 0.7 * tp + 0.3 * season_avg_finish.get(d, tp)
    grid = {d: i + 1 for i, (d, _) in enumerate(sorted(proxy.items(), key=lambda x: x[1]))}

    # conditions: forecast if given, else mild dry default
    if forecast:
        temp, humidity, rain_prob = forecast['air_temp'], forecast['humidity'], float(forecast['rain_prob'])
    else:
        temp, humidity, rain_prob = 22.0, 50.0, 0.0
    return {
        'grid': grid, 'estimated_grid': True, 'stage': stage, 'team_map': team_map,
        'sector_s1': {d: s1d.get(d, 0) for d in drivers},
        'sector_s2': {d: s2d.get(d, 0) for d in drivers},
        'sector_s3': {d: s3d.get(d, 0) for d in drivers},
        'car_pace': car_pace, 'car_speed': car_speed,
        'temp': temp, 'humidity': humidity, 'rain': 1 if rain_prob >= 50 else 0,
        'rain_prob': rain_prob, 'race_results': None, 'quali_results': None,
    }


def predict(df, feat_df, model, lk, wk):
    """Predict the target race for the 2026 grid using actual qualifying grid +
    live sectors + blended pace. Returns a sorted list of dicts (best first).

    When wk['rain_prob'] is between 0 and 100 the prediction is a probability
    blend of a dry run and a wet run (had_rain=1, humidity raised to 85), so a
    forecast like '40% rain' shifts the order toward wet-weather skill
    proportionally instead of flipping on a hard threshold."""
    rp = wk.get('rain_prob', 0.0) / 100.0
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
        dry = float(model.predict(pd.DataFrame([{**base, 'had_rain': 0}])[FEATURE_COLS].fillna(0))[0])
        if 0 < rp < 1:
            wet_feat = {**base, 'had_rain': 1, 'humidity': max(base['humidity'], 85)}
            wet = float(model.predict(pd.DataFrame([wet_feat])[FEATURE_COLS].fillna(0))[0])
            pred = (1 - rp) * dry + rp * wet
        else:
            pred = dry if rp == 0 else float(
                model.predict(pd.DataFrame([{**base, 'had_rain': 1,
                                             'humidity': max(base['humidity'], 85)}])[FEATURE_COLS].fillna(0))[0])
        preds.append({'Driver': d, 'Team': t, 'Grid': wk['grid'][d],
                      'Car': wk['car_pace'].get(t, 15), 'Predicted': pred, 'Features': base})
    preds.sort(key=lambda x: x['Predicted'])
    for i, p in enumerate(preds):
        p['Rank'] = i + 1
    return preds


def simulate(preds, residual_std, n_sims=10000, sc_prob=0.33, seed=42, dnf=True):
    """Monte-Carlo race sim with grid-dependent variance, safety car, and DNFs.

    Each sim, a driver retires with probability = their season dnf_rate; retirees
    are classified behind every finisher, so a front-runner's DNF promotes
    everyone behind. This is what lets the sim produce the position movement the
    point estimate misses on high-attrition tracks (Canada, Monaco). dnf_rate is
    clamped to [0.02, 0.40] to keep a single bad streak from dominating.
    """
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

    dnf_rate = {p['Driver']: min(0.40, max(0.02, p['Features'].get('dnf_rate', 0.1)))
                for p in preds}

    pt = defaultdict(list)
    wc = defaultdict(int); pc = defaultdict(int); t5 = defaultdict(int); t10 = defaultdict(int)
    dnf_count = defaultdict(int)
    for _ in range(n_sims):
        sc = rng.random() < sc_prob
        finishers, retirees = [], []
        for p in preds:
            d = p['Driver']
            if dnf and rng.random() < dnf_rate[d]:
                retirees.append(d)
                dnf_count[d] += 1
                continue
            pred = p['Predicted'] + rng.normal(0, gv(p['Grid']))
            if sc and p['Grid'] > 5:
                pred -= rng.uniform(0, 1.5)
            finishers.append((d, pred))
        finishers.sort(key=lambda x: x[1])
        # retirees fill the classified positions behind all finishers, worst-grid last
        order = [d for d, _ in finishers] + sorted(
            retirees, key=lambda d: next(p['Grid'] for p in preds if p['Driver'] == d))
        for pos, d in enumerate(order):
            pt[d].append(pos + 1)
            if pos == 0: wc[d] += 1
            if pos < 3: pc[d] += 1
            if pos < 5: t5[d] += 1
            if pos < 10: t10[d] += 1
    return {'pt': pt, 'win': wc, 'podium': pc, 'top5': t5, 'top10': t10,
            'dnf': dnf_count, 'n': n_sims}


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
    # grid baseline: the naive "everyone finishes where they qualified" predictor,
    # graded the same way as the model so we always know if the model adds value.
    grid_rank = {d: i + 1 for i, (d, _) in enumerate(
        sorted(((p['Driver'], p['Grid']) for p in preds), key=lambda x: x[1]))}
    grid_mae = float(np.mean([abs(grid_rank[r['Driver']] - r['Actual']) for r in clean])) if clean else None
    grid_order = [d for d, _ in sorted(grid_rank.items(), key=lambda x: x[1])]
    return {
        'rows': rows, 'clean': clean,
        'mae': float(np.mean([r['Err'] for r in clean])) if clean else None,
        'grid_mae': grid_mae,
        'model_edge': (grid_mae - float(np.mean([r['Err'] for r in clean]))) if clean else None,
        'grid_top5': len(set(grid_order[:5]) & set(actual_order[:5])),
        'grid_top10': len(set(grid_order[:10]) & set(actual_order[:10])),
        'winner_pred': pred_order[0], 'winner_actual': actual_order[0],
        'winner_hit': pred_order[0] == actual_order[0],
        'podium': len(set(pred_order[:3]) & set(actual_order[:3])),
        'top5': len(set(pred_order[:5]) & set(actual_order[:5])),
        'top10': len(set(pred_order[:10]) & set(actual_order[:10])),
        'pred_order': pred_order, 'actual_order': actual_order,
    }
