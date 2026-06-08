"""Build a per-driver historical sector-time CSV for a track.

Usage:
    python scripts/build_sectors.py Miami data/miami_sectors.csv
    python scripts/build_sectors.py Monaco data/monaco_sectors.csv 2023 2024 2025

Mirrors data/suzuka_sectors.csv: one row per driver per year with average and
best sector times from race laps (median*1.05 outlier filter).
"""
import sys
import numpy as np
import pandas as pd
import fastf1

fastf1.Cache.enable_cache('data/cache')

event = sys.argv[1]
out_path = sys.argv[2]
years = [int(y) for y in sys.argv[3:]] or [2023, 2024, 2025]

rows = []
for year in years:
    try:
        ses = fastf1.get_session(year, event, 'R')
        ses.load()
    except Exception as e:
        print(f'  {year}: skipped ({e})')
        continue
    laps = ses.laps
    name_map = dict(zip(ses.results['Abbreviation'], ses.results['FullName']))
    team_map = dict(zip(ses.results['Abbreviation'], ses.results['TeamName']))
    for d in laps['Driver'].unique():
        dl = laps[laps['Driver'] == d]
        rec = {'Year': year, 'FullName': name_map.get(d, d), 'Team': team_map.get(d, '')}
        ok = True
        for i in (1, 2, 3):
            times = dl[f'Sector{i}Time'].dropna().dt.total_seconds()
            if len(times) < 3:
                ok = False
                break
            med = times.median()
            quick = times[times < med * 1.05]
            rec[f'avg_s{i}'] = quick.mean()
            rec[f'best_s{i}'] = times.min()
        if ok:
            rows.append(rec)
    print(f'  {year}: {ses.event["EventName"]} ok')

df = pd.DataFrame(rows)[['Year', 'FullName', 'Team',
                         'avg_s1', 'avg_s2', 'avg_s3',
                         'best_s1', 'best_s2', 'best_s3']]
df.to_csv(out_path, index=False)
print(f'{out_path}: {len(df)} rows across {df.Year.nunique()} years')
