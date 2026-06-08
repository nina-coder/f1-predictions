"""Append a completed race weekend to the model's data CSVs.

Usage:
    python scripts/update_data.py 2026 3            # append 2026 Round 3
    python scripts/update_data.py 2026 3 --dry-run  # show what would be appended

Appends the Grand Prix (race session only, consistent with historical data) to:
    data/all_results.csv        - classification, grid, points, status
    data/weather.csv            - session weather aggregates
    data/lap_stats.csv          - per-driver speeds, pits, stints, consistency
    data/advanced_features.csv  - first-lap gain, tire degradation, quali/race gap

Note: tire_degradation is the mean within-stint lap-time slope (s/lap) using a
median*1.05 outlier filter on stints of 5+ laps. This matches the original
feature definition directionally (validated against 2026 R1 within ~0.06 s/lap).
"""
import sys
import numpy as np
import pandas as pd
import fastf1

DATA = 'data'


def load_race(year, rnd):
    fastf1.Cache.enable_cache(f'{DATA}/cache')
    ses = fastf1.get_session(year, rnd, 'R')
    ses.load()
    return ses


def results_rows(ses, year, rnd):
    rows = []
    for _, r in ses.results.iterrows():
        rows.append({
            'DriverNumber': int(r['DriverNumber']),
            'FullName': r['FullName'],
            'TeamName': r['TeamName'],
            'Position': r['Position'],
            'GridPosition': r['GridPosition'],
            'Points': r['Points'],
            'Status': r['Status'],
            'Year': year, 'Round': rnd,
            'RaceName': ses.event['EventName'],
            'Circuit': ses.event['Location'],
        })
    return pd.DataFrame(rows)


def weather_row(ses, year, rnd):
    w = ses.weather_data
    return pd.DataFrame([{
        'Year': year, 'Round': rnd, 'RaceName': ses.event['EventName'],
        'avg_air_temp': w['AirTemp'].mean(),
        'avg_track_temp': w['TrackTemp'].mean(),
        'avg_humidity': w['Humidity'].mean(),
        'avg_wind_speed': w['WindSpeed'].mean(),
        'had_rain': bool(w['Rainfall'].any()),
        'rain_pct': float(w['Rainfall'].mean() * 100),
    }])


def lap_stats_rows(ses, year, rnd, label):
    laps = ses.laps
    name_map = dict(zip(ses.results['Abbreviation'], ses.results['FullName']))
    team_map = dict(zip(ses.results['Abbreviation'], ses.results['TeamName']))
    rows = []
    for d in laps['Driver'].unique():
        dl = laps[laps['Driver'] == d]
        clean = dl['LapTime'].dropna().dt.total_seconds()
        if len(clean) == 0:
            continue
        med = clean.median()
        quick = clean[clean < med * 1.05]
        rows.append({
            'Year': year, 'Round': rnd, 'Label': label,
            'Driver': d, 'FullName': name_map.get(d, d), 'Team': team_map.get(d, ''),
            'avg_speed_i1': dl['SpeedI1'].mean(), 'avg_speed_i2': dl['SpeedI2'].mean(),
            'avg_speed_fl': dl['SpeedFL'].mean(), 'avg_speed_st': dl['SpeedST'].mean(),
            'top_speed': dl['SpeedST'].max(),
            'n_pits': int(dl['PitInTime'].notna().sum()),
            'n_stints': int(dl['Stint'].nunique()),
            'primary_compound': dl['Compound'].mode().iloc[0] if dl['Compound'].notna().any() else '',
            'lap_time_std': quick.std(),
            'avg_lap_seconds': quick.mean(),
            'total_laps': len(dl),
        })
    return pd.DataFrame(rows)


def tire_degradation(dl):
    slopes = []
    for st in dl['Stint'].dropna().unique():
        stint = dl[dl['Stint'] == st].sort_values('LapNumber')
        times = stint['LapTime'].dropna()
        if len(times) >= 5:
            secs = times.dt.total_seconds().values
            med = np.median(secs)
            mask = secs < med * 1.05
            if mask.sum() >= 4:
                x = np.arange(len(secs))[mask]
                slopes.append(np.polyfit(x, secs[mask], 1)[0])
    return float(np.mean(slopes)) if slopes else np.nan


def advanced_rows(ses, year, rnd):
    laps = ses.laps
    res = ses.results
    name_map = dict(zip(res['Abbreviation'], res['FullName']))
    # safety car: track status 4 (SC) or 6/7 (VSC) at any point
    status = laps['TrackStatus'].dropna().astype(str)
    had_sc = bool(status.str.contains('4|6|7').any())
    lap1 = laps[laps['LapNumber'] == 1]
    rows = []
    for _, r in res.iterrows():
        d = r['Abbreviation']
        dl = laps[laps['Driver'] == d]
        l1 = lap1[lap1['Driver'] == d]
        fl_gain = (float(r['GridPosition']) - float(l1.iloc[0]['Position'])
                   if len(l1) and pd.notna(l1.iloc[0]['Position']) else np.nan)
        qrg = (float(r['GridPosition']) - float(r['Position'])
               if pd.notna(r['Position']) else np.nan)
        rows.append({
            'Year': year, 'Round': rnd, 'RaceName': ses.event['EventName'],
            'FullName': r['FullName'], 'Team': r['TeamName'],
            'first_lap_gain': fl_gain,
            'tire_degradation': tire_degradation(dl),
            'quali_race_gap': qrg,
            'had_safety_car': had_sc,
            'grid': r['GridPosition'], 'finish': r['Position'],
        })
    return pd.DataFrame(rows)


def append(path, new, keys=('Year', 'Round')):
    df = pd.read_csv(path)
    mask = (df[keys[0]] == new[keys[0]].iloc[0]) & (df[keys[1]] == new[keys[1]].iloc[0])
    if mask.any():
        print(f'  {path}: rows for this event already present, replacing {mask.sum()}')
        df = df[~mask]
    out = pd.concat([df, new], ignore_index=True)
    out.to_csv(path, index=False)
    print(f'  {path}: +{len(new)} rows -> {len(out)} total')


if __name__ == '__main__':
    year, rnd = int(sys.argv[1]), int(sys.argv[2])
    dry = '--dry-run' in sys.argv
    ses = load_race(year, rnd)
    label = f"{ses.event['Location']} {year}"
    print(f"Loaded: {year} R{rnd} {ses.event['EventName']} ({ses.event['Location']})")

    res = results_rows(ses, year, rnd)
    wea = weather_row(ses, year, rnd)
    ls = lap_stats_rows(ses, year, rnd, label)
    adv = advanced_rows(ses, year, rnd)

    if dry:
        print(res.head(25).to_string())
        print(wea.to_string())
        print(ls.head(25).to_string())
        print(adv.head(25).to_string())
    else:
        append(f'{DATA}/all_results.csv', res)
        append(f'{DATA}/weather.csv', wea)
        append(f'{DATA}/lap_stats.csv', ls)
        append(f'{DATA}/advanced_features.csv', adv)
