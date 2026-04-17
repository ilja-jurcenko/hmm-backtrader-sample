#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
walkforward-compare.py
======================
Run walk-forward HMM optimisation for multiple strategies in **parallel**,
save each output to a .txt file, then print a side-by-side summary.

Usage
-----
    python walkforward-compare.py \
        --ticker SPY \
        --wf-start 2015-01-01 --wf-end 2025-01-01 \
        --is-years 3 --oos-years 1 --step 1 --n-trials 50 \
        --strategies sma dema rsi macd \
        --out-dir ./wf_results

Each strategy runs as a separate subprocess so they execute in parallel.
Output files: wf_results/sma.txt, wf_results/dema.txt, etc.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
WF_SCRIPT = os.path.join(HERE, 'walkforward-hmm.py')
PYTHON = sys.executable


def _run_strategy(strategy: str, base_args: list[str], out_path: str) -> tuple[str, float, int]:
    """Run walkforward-hmm.py for one strategy, write output to out_path."""
    cmd = [PYTHON, WF_SCRIPT, '--strategy', strategy] + base_args
    t0 = time.perf_counter()
    with open(out_path, 'w') as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    return strategy, elapsed, result.returncode


# ---------------------------------------------------------------------------
# Summary scraper (parse key lines from the txt output)
# ---------------------------------------------------------------------------

def _scrape_summary(txt_path: str) -> dict:
    """Extract key numbers from a walkforward-hmm output file."""
    summary = {}
    try:
        with open(txt_path) as f:
            content = f.read()
    except FileNotFoundError:
        return {'error': 'file not found'}

    # Overall windows
    m = re.search(r'Total windows\s+:\s+(\d+)', content)
    if m:
        summary['windows'] = int(m.group(1))

    # HMM win rate
    m = re.search(r'HMM wins.*?:\s+(\d+)\s*/\s*(\d+)\s*\(([0-9.]+)%\)', content)
    if m:
        summary['hmm_wins']   = int(m.group(1))
        summary['win_rate']   = float(m.group(3))

    # Mean delta return
    m = re.search(r'Mean Δ return\s+:\s+([+-][0-9.]+)%', content)
    if m:
        summary['mean_delta_ret'] = float(m.group(1))

    # OOS metric comparison table rows
    metrics = {
        'Total Return (%)': 'total_return',
        'CAGR (%)':         'cagr',
        'Sharpe':           'sharpe',
        'Calmar':           'calmar',
        'Max DrawDown (%)': 'max_dd',
    }
    for label, key in metrics.items():
        pattern = rf'{re.escape(label)}\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([+-][0-9.]+)\s+(✓ YES|✗ NO)'
        m = re.search(pattern, content)
        if m:
            summary[f'bl_{key}']    = float(m.group(1).lstrip('+'))
            summary[f'hmm_{key}']   = float(m.group(2).lstrip('+'))
            summary[f'delta_{key}'] = float(m.group(3))
            summary[f'improved_{key}'] = (m.group(4) == '✓ YES')

    # Count improved metrics
    n_improved = sum(1 for k, v in summary.items()
                     if k.startswith('improved_') and v is True)
    summary['metrics_improved'] = n_improved

    return summary


# ---------------------------------------------------------------------------
# Pretty summary table
# ---------------------------------------------------------------------------

def _print_comparison(strategies: list[str], summaries: dict[str, dict]):
    W = 120
    print('\n\n' + '=' * W)
    print('  STRATEGY COMPARISON  –  Walk-Forward OOS Results (means across all windows)')
    print('=' * W)

    cols   = strategies
    c_w    = max(14, max(len(s) for s in strategies) + 2)
    lbl_w  = 24

    hdr = f'  {"Metric":<{lbl_w}}' + ''.join(f'{s.upper():>{c_w}}' for s in cols)
    print(hdr)
    print('  ' + '-' * (lbl_w + c_w * len(cols)))

    def _row(label, key, fmt='+.2f', suffix=''):
        line = f'  {label:<{lbl_w}}'
        for s in cols:
            v = summaries[s].get(key)
            if v is None:
                line += f'{"N/A":>{c_w}}'
            else:
                line += f'{format(v, fmt) + suffix:>{c_w}}'
        print(line)

    def _bool_row(label, key):
        line = f'  {label:<{lbl_w}}'
        for s in cols:
            v = summaries[s].get(key)
            cell = '✓ YES' if v is True else ('✗ NO' if v is False else 'N/A')
            line += f'{cell:>{c_w}}'
        print(line)

    _row('Mean Δ Return (%)',   'mean_delta_ret', '+.2f', '%')
    _row('Baseline Return (%)', 'bl_total_return', '.2f', '%')
    _row('HMM Return (%)',      'hmm_total_return', '.2f', '%')
    print()
    _row('Baseline CAGR (%)',   'bl_cagr',   '.2f', '%')
    _row('HMM CAGR (%)',        'hmm_cagr',  '.2f', '%')
    _row('Δ CAGR (%)',          'delta_cagr', '+.2f', '%')
    print()
    _row('Baseline Sharpe',     'bl_sharpe',    '.3f')
    _row('HMM Sharpe',          'hmm_sharpe',   '.3f')
    _row('Δ Sharpe',            'delta_sharpe', '+.3f')
    print()
    _row('Baseline MaxDD (%)',   'bl_max_dd',    '.2f', '%')
    _row('HMM MaxDD (%)',        'hmm_max_dd',   '.2f', '%')
    _row('Δ MaxDD (%)',          'delta_max_dd', '+.2f', '%')
    print()
    _row('HMM Win Rate (%)',     'win_rate',    '.1f', '%')
    _row('Metrics Improved',     'metrics_improved', 'd', '/5')

    print('=' * W)

    # Verdict: which strategy benefited most from HMM
    valid = {s: summaries[s] for s in strategies if 'mean_delta_ret' in summaries[s]}
    if valid:
        best = max(valid, key=lambda s: valid[s]['mean_delta_ret'])
        print(f'\n  Best HMM synergy : {best.upper()}  '
              f'(mean Δ return {valid[best]["mean_delta_ret"]:+.2f}% per OOS window)')
    print('=' * W + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run walk-forward HMM comparison across multiple strategies in parallel')

    p.add_argument('--strategies', nargs='+',
        default=['sma', 'dema', 'rsi', 'macd'],
        choices=['sma', 'dema', 'rsi', 'macd', 'hmm_mr'],
        help='Strategies to compare (run in parallel)')
    p.add_argument('--out-dir', default='./wf_results', dest='out_dir',
        help='Directory for output .txt files')

    # --- pass-through args to walkforward-hmm.py ---
    p.add_argument('--ticker',     nargs='+', default=['SPY'])
    p.add_argument('--wf-start',   default='2015-01-01', dest='wf_start')
    p.add_argument('--wf-end',     default='2025-01-01', dest='wf_end')
    p.add_argument('--is-years',   type=int,   default=3,  dest='is_years')
    p.add_argument('--oos-years',  type=int,   default=1,  dest='oos_years')
    p.add_argument('--step',       type=int,   default=1)
    p.add_argument('--n-trials',   type=int,   default=40, dest='n_trials')
    p.add_argument('--fast',       type=int,   default=10)
    p.add_argument('--slow',       type=int,   default=30)
    p.add_argument('--rsi-period',     type=int, default=14, dest='rsi_period')
    p.add_argument('--rsi-oversold',   type=int, default=30, dest='rsi_oversold')
    p.add_argument('--rsi-overbought', type=int, default=70, dest='rsi_overbought')
    p.add_argument('--macd-fast',   type=int, default=12, dest='macd_fast')
    p.add_argument('--macd-slow',   type=int, default=26, dest='macd_slow')
    p.add_argument('--macd-signal', type=int, default=9,  dest='macd_signal')
    p.add_argument('--stake',      type=int,   default=100)
    p.add_argument('--cash',       type=float, default=100_000.0)
    p.add_argument('--commission', type=float, default=0.001)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--hmm-score-threshold', type=float, default=None,
        dest='hmm_score_threshold')
    p.add_argument('--hmm-mr-z-threshold', type=float, default=0.0,
        dest='hmm_mr_z_threshold',
        help='HMM-MR: std-devs below state mean required before entry')

    return p.parse_args()


def _build_passthrough(cfg) -> list[str]:
    """Build the list of CLI args to forward to walkforward-hmm.py."""
    args = []
    args += ['--ticker']    + cfg.ticker
    args += ['--wf-start',  cfg.wf_start]
    args += ['--wf-end',    cfg.wf_end]
    args += ['--is-years',  str(cfg.is_years)]
    args += ['--oos-years', str(cfg.oos_years)]
    args += ['--step',      str(cfg.step)]
    args += ['--n-trials',  str(cfg.n_trials)]
    args += ['--fast',      str(cfg.fast)]
    args += ['--slow',      str(cfg.slow)]
    args += ['--rsi-period',     str(cfg.rsi_period)]
    args += ['--rsi-oversold',   str(cfg.rsi_oversold)]
    args += ['--rsi-overbought', str(cfg.rsi_overbought)]
    args += ['--macd-fast',   str(cfg.macd_fast)]
    args += ['--macd-slow',   str(cfg.macd_slow)]
    args += ['--macd-signal', str(cfg.macd_signal)]
    args += ['--stake',      str(cfg.stake)]
    args += ['--cash',       str(cfg.cash)]
    args += ['--commission', str(cfg.commission)]
    args += ['--seed',       str(cfg.seed)]
    if cfg.hmm_score_threshold is not None:
        args += ['--hmm-score-threshold', str(cfg.hmm_score_threshold)]
    args += ['--hmm-mr-z-threshold', str(cfg.hmm_mr_z_threshold)]
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    os.makedirs(cfg.out_dir, exist_ok=True)
    base_args = _build_passthrough(cfg)

    strategies = cfg.strategies
    out_paths  = {s: os.path.join(cfg.out_dir, f'{s}.txt') for s in strategies}

    print(f'\n{"=" * 70}')
    print(f'  Walk-Forward Strategy Comparison')
    print(f'  Strategies : {", ".join(s.upper() for s in strategies)}')
    print(f'  Tickers    : {", ".join(cfg.ticker)}')
    print(f'  Period     : {cfg.wf_start} → {cfg.wf_end}')
    print(f'  IS/OOS     : {cfg.is_years}y IS  /  {cfg.oos_years}y OOS  (step {cfg.step}y)')
    print(f'  Trials     : {cfg.n_trials} per window')
    print(f'  Output dir : {os.path.abspath(cfg.out_dir)}')
    print(f'{"=" * 70}')
    print(f'\n  Launching {len(strategies)} parallel workers …\n')

    t_start  = time.perf_counter()
    statuses = {}

    with ProcessPoolExecutor(max_workers=len(strategies)) as pool:
        futures = {
            pool.submit(_run_strategy, s, base_args, out_paths[s]): s
            for s in strategies
        }
        for fut in as_completed(futures):
            strategy, elapsed, rc = fut.result()
            status = '✓ OK' if rc == 0 else f'✗ FAILED (exit {rc})'
            statuses[strategy] = rc
            print(f'  [{strategy.upper():<5}] {status}  ({elapsed:.1f}s)  '
                  f'→ {out_paths[strategy]}')

    total_elapsed = time.perf_counter() - t_start
    print(f'\n  All strategies finished in {total_elapsed:.1f}s  '
          f'(wall-clock, ran in parallel)\n')

    # Build summary from scraped txt files
    summaries = {}
    for s in strategies:
        if statuses.get(s) == 0:
            summaries[s] = _scrape_summary(out_paths[s])
        else:
            summaries[s] = {'error': f'exit code {statuses[s]}'}

    # Print comparison table
    ok_strategies = [s for s in strategies if 'error' not in summaries[s]]
    if ok_strategies:
        _print_comparison(ok_strategies, summaries)
    else:
        print('  [ERROR] All strategies failed — check output files for details.')
        sys.exit(1)

    # Print per-file paths for easy access
    print('  Output files:')
    for s in strategies:
        print(f'    {s.upper():<6} → {out_paths[s]}')
    print()


if __name__ == '__main__':
    main()
