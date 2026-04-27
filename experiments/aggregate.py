#!/usr/bin/env python3
"""
experiments/aggregate.py
========================
Aggregate all *_results.csv files produced by cluster jobs into a single
master comparison table.

Results are expected at:
    results/phase_{N}/{group_id}/{ticker_set}/{timeframe}/{strategy}_results.csv

Usage:
    python experiments/aggregate.py [--phase N] [--out master_results.csv]

Options:
    --phase N      Only include groups whose phase == N (default: all)
    --out PATH     Output path for master CSV (default: results/master_results.csv)
    --sort-by COL  Column to sort the summary table by (default: hmm_oos_sharpe_mean)
"""

from __future__ import annotations

import argparse
import os
import glob
import math
from typing import Optional

import pandas as pd
import yaml

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
CONFIGS = os.path.join(HERE, 'configs')


def load_group_phases() -> dict[str, int]:
    """Return {group_id: phase} from groups.yaml."""
    with open(os.path.join(CONFIGS, 'groups.yaml')) as f:
        cfg = yaml.safe_load(f)
    return {g['id']: g['phase'] for g in cfg['groups']}


def _mean(vals, clip=None):
    """Mean of non-null values.  If clip is set, winsorise to ±clip first."""
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if clip is not None:
        valid = [max(-clip, min(clip, v)) for v in valid]
    return sum(valid) / len(valid) if valid else float('nan')


def _win_rate(improved_col):
    vals = [bool(v) for v in improved_col if v is not None]
    return 100 * sum(vals) / len(vals) if vals else float('nan')


METRICS = [
    # (csv_col_prefix, display_label, higher_is_better)
    ('return',     'Return (%)',   True),
    ('annual',     'CAGR (%)',     True),
    ('sharpe',     'Sharpe',       True),
    ('calmar',     'Calmar',       True),
    ('dd',         'MaxDD (%)',    False),
    ('trades',     'Trades',       None),   # informational
    ('won_trades', 'Won Trades',   None),   # informational
]


def aggregate(phase_filter: Optional[int], out_path: str, sort_by: str):
    group_phases = load_group_phases()

    pattern = os.path.join(ROOT, 'results', 'phase_*', '*', '*', '*', '*_results.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f'No result CSVs found matching:\n  {pattern}')
        return

    rows = []
    missing = []
    for path in sorted(csv_files):
        # results/phase_{N}/{group_id}/{ticker_set}/{timeframe}/{strategy}_results.csv
        parts = os.path.normpath(path).split(os.sep)
        try:
            strategy  = os.path.splitext(parts[-1])[0].replace('_results', '')
            timeframe = parts[-2]
            ticker_set= parts[-3]
            group_id  = parts[-4]
        except IndexError:
            print(f'Warning: unexpected path structure: {path}')
            continue

        phase = group_phases.get(group_id)
        if phase_filter is not None and phase != phase_filter:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            missing.append((path, str(e)))
            continue

        if df.empty:
            missing.append((path, 'empty CSV'))
            continue

        agg = {
            'group_id':   group_id,
            'phase':      phase,
            'ticker_set': ticker_set,
            'timeframe':  timeframe,
            'strategy':   strategy,
            'n_windows':  len(df),
            'win_rate':   _win_rate(df.get('improved', [])),
        }

        # Sharpe values are winsorised at ±10 before averaging across windows.
        # A single extreme Sharpe (e.g. −57 from a 1-trade OOS window with
        # near-zero daily-return variance) would otherwise dominate the mean.
        SHARPE_CLIP = 10.0

        for col_suffix, _, _ in METRICS:
            for prefix in ('bl_oos', 'hmm_oos'):
                col = f'{prefix}_{col_suffix}'
                if col in df.columns:
                    clip = SHARPE_CLIP if col_suffix == 'sharpe' else None
                    agg[f'{col}_mean'] = _mean(df[col], clip=clip)

        # delta = hmm - baseline
        for col_suffix, _, _ in METRICS:
            bl_col  = f'bl_oos_{col_suffix}_mean'
            hmm_col = f'hmm_oos_{col_suffix}_mean'
            if bl_col in agg and hmm_col in agg:
                agg[f'delta_{col_suffix}'] = agg[hmm_col] - agg[bl_col]

        rows.append(agg)

    if not rows:
        print('No matching CSVs loaded.')
        return

    master = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    master.to_csv(out_path, index=False)
    print(f'Master results saved → {out_path}  ({len(master)} rows)')

    if missing:
        print(f'\nWarning: {len(missing)} files could not be loaded:')
        for path, err in missing:
            print(f'  {path}: {err}')

    # Print summary pivot: group_id × strategy, sorted by sort_by
    print_summary(master, sort_by)


def print_summary(master: pd.DataFrame, sort_by: str):
    print('\n' + '=' * 110)
    print('  AGGREGATE SUMMARY  (mean OOS metrics across windows)')
    print('=' * 110)

    col_w = 14
    hdr_cols = ['group_id', 'ticker_set', 'tf', 'strategy',
                'n_win', 'win%',
                'bl_sharpe', 'hmm_sharpe', 'Δsharpe',
                'bl_cagr%',  'hmm_cagr%',  'Δcagr%',
                'bl_dd%',    'hmm_dd%',    'Δdd%']
    print('  ' + '  '.join(f'{c:<{col_w}}' for c in hdr_cols))
    print('  ' + '-' * (len(hdr_cols) * (col_w + 2)))

    def _fmt(val, fmt='.3f'):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 'n/a'
        return format(val, fmt)

    if sort_by in master.columns:
        master = master.sort_values(sort_by, ascending=False)

    for _, r in master.iterrows():
        row = [
            str(r.get('group_id',   '')),
            str(r.get('ticker_set', '')),
            str(r.get('timeframe',  '')),
            str(r.get('strategy',   '')),
            str(int(r.get('n_windows', 0))),
            _fmt(r.get('win_rate'), '.1f'),
            _fmt(r.get('bl_oos_sharpe_mean')),
            _fmt(r.get('hmm_oos_sharpe_mean')),
            _fmt(r.get('delta_sharpe')),
            _fmt(r.get('bl_oos_annual_mean'),  '.2f'),
            _fmt(r.get('hmm_oos_annual_mean'), '.2f'),
            _fmt(r.get('delta_annual'),        '.2f'),
            _fmt(r.get('bl_oos_dd_mean'),      '.2f'),
            _fmt(r.get('hmm_oos_dd_mean'),     '.2f'),
            _fmt(r.get('delta_dd'),            '.2f'),
        ]
        print('  ' + '  '.join(f'{v:<{col_w}}' for v in row))

    print('=' * 110)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', type=int, default=None,
                        help='Only aggregate groups from this phase number')
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'master_results.csv'),
                        help='Output path for master CSV')
    parser.add_argument('--sort-by', default='hmm_oos_sharpe_mean', dest='sort_by',
                        help='Column to sort the printed summary table by')
    args = parser.parse_args()
    aggregate(phase_filter=args.phase, out_path=args.out, sort_by=args.sort_by)


if __name__ == '__main__':
    main()
