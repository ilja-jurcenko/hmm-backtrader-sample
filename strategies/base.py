#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Base regime-aware strategy
==========================
All concrete strategies in this package inherit from ``BaseRegimeStrategy``.

Sub-classes must implement:
    _init_indicators(d)  → populate self.inds[d] with required indicator objects
    _signal(d)           → return +1 (buy), -1 (sell), or 0 (no signal)

The base class handles:
    - Order and position sentinels (self.orders)
    - Regime gate (HMM or always-on)
    - notify_order / notify_trade callbacks
    - next() generic loop
    - stop() summary log
"""
from __future__ import annotations

import backtrader as bt


class BaseRegimeStrategy(bt.Strategy):
    """
    Abstract base for all HMM-aware crossover / indicator strategies.

    Concrete children override ``_init_indicators`` and ``_signal``.
    """

    params = dict(
        stake         = 100,        # shares per trade per instrument
        printlog      = True,       # print trade log to stdout
        use_hmm       = False,      # enable HMM regime gate
        regime_mode   = 'strict',   # 'strict' = block trades in unfav regime
                                    # 'size'   = reduce position size instead
        unfav_fraction = 0.25,      # fraction of stake used in unfavourable regime
                                    # (only used when regime_mode='size')
    )

    # ------------------------------------------------------------------ helpers

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    def _get_regime(self, d) -> float:
        """Return regime value: 1.0 (fully allowed), 0.0 (suppressed),
        or a float in [0,1] for score-based sizing."""
        if self.p.use_hmm and hasattr(d, 'regime'):
            return float(d.regime[0])
        return 1.0

    def _get_hmm_state(self, d) -> int:
        """Return the dominant HMM component index, or -1 if unavailable."""
        if self.p.use_hmm and hasattr(d, 'hmm_state'):
            return int(d.hmm_state[0])
        return -1

    def _get_hmm_score(self, d) -> float:
        """Return the composite score of the dominant HMM state, or 0.0."""
        if self.p.use_hmm and hasattr(d, 'hmm_score'):
            return float(d.hmm_score[0])
        return 0.0

    # ------------------------------------------------------------------ lifecycle

    def __init__(self):
        self.inds   = {}
        self.orders = {}
        # Per-regime trade tracking
        self._regime_entries = []   # (regime_value, size, dname, hmm_state, hmm_score)
        self._regime_bars    = []   # list of regime values for every bar
        for d in self.datas:
            self.orders[d] = None
            self.inds[d]   = {}
            self._init_indicators(d)

    def _init_indicators(self, d):
        """Populate self.inds[d].  Must be overridden by sub-classes."""
        raise NotImplementedError

    def _signal(self, d) -> int:
        """Return +1 = buy, -1 = sell, 0 = hold.  Must be overridden."""
        raise NotImplementedError

    # ------------------------------------------------------------------ events

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dname = order.data._name or order.data._dataname

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f'[{dname}] BUY  executed  price={order.executed.price:.2f}  '
                    f'cost={order.executed.value:.2f}  '
                    f'comm={order.executed.comm:.2f}')
            else:
                self.log(
                    f'[{dname}] SELL executed  price={order.executed.price:.2f}  '
                    f'cost={order.executed.value:.2f}  '
                    f'comm={order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'[{dname}] Order canceled / margin / rejected')

        self.orders[order.data] = None

    def notify_trade(self, trade):
        if trade.isclosed:
            dname = trade.data._name or trade.data._dataname
            self.log(
                f'[{dname}] TRADE CLOSED  gross={trade.pnl:.2f}  '
                f'net={trade.pnlcomm:.2f}')

    # ------------------------------------------------------------------ core

    def next(self):
        for d in self.datas:
            if self.orders[d]:
                continue

            dname  = d._name or d._dataname
            pos    = self.getposition(d)
            regime = self._get_regime(d)
            hmm_st = self._get_hmm_state(d)
            hmm_sc = self._get_hmm_score(d)
            sig    = self._signal(d)

            # Determine position size based on regime
            if self.p.regime_mode in ('score', 'linear'):
                # Continuous sizing: regime is a float in [0, 1]
                if regime < 1e-6:
                    size = 0   # score ≈ 0 → block trades entirely
                else:
                    size = max(1, int(self.p.stake * regime))
            elif regime >= 1.0:
                size = self.p.stake
            elif self.p.regime_mode == 'size':
                size = max(1, int(self.p.stake * self.p.unfav_fraction))
            else:
                size = 0   # strict mode: no new entries

            if not pos.size:
                if sig > 0 and size > 0:
                    self.log(f'[{dname}] BUY  signal  close={d.close[0]:.2f}  '
                             f'regime={regime:.2f}  size={size}')
                    self.orders[d] = self.buy(data=d, size=size)
                    self._regime_entries.append((regime, size, dname, hmm_st, hmm_sc))
            else:
                if self.p.regime_mode in ('score', 'linear') and sig > 0:
                    # Score mode: resize position to match current regime score
                    if regime < 1e-6:
                        # Score = 0 → close position entirely
                        self.log(f'[{dname}] SELL signal (regime≈0)  close={d.close[0]:.2f}')
                        self.orders[d] = self.sell(data=d, size=pos.size)
                        continue
                    target = max(1, int(self.p.stake * regime))
                    if pos.size > target:
                        trim = pos.size - target
                        self.log(f'[{dname}] TRIM position ({pos.size}→{target})  '
                                 f'regime={regime:.2f}  close={d.close[0]:.2f}')
                        self.orders[d] = self.sell(data=d, size=trim)
                        continue
                elif self.p.regime_mode == 'size' and regime == 0 and sig > 0:
                    # Regime turned unfavourable while in position – resize
                    target = max(1, int(self.p.stake * self.p.unfav_fraction))
                    if pos.size > target:
                        trim = pos.size - target
                        self.log(f'[{dname}] TRIM position ({pos.size}→{target})  '
                                 f'regime={regime:.2f}  close={d.close[0]:.2f}')
                        self.orders[d] = self.sell(data=d, size=trim)
                        continue
                if sig < 0 or (self.p.regime_mode == 'strict' and regime == 0) \
                        or (self.p.regime_mode in ('score', 'linear') and regime < 1e-6):
                    reason = 'signal' if sig < 0 else 'regime=0'
                    self.log(f'[{dname}] SELL signal ({reason})  close={d.close[0]:.2f}')
                    self.orders[d] = self.sell(data=d, size=pos.size)

    def stop(self):
        # Print regime trade analysis
        if self._regime_entries and self.p.printlog:
            self._print_regime_analysis()

    def _print_regime_analysis(self):
        """Print a breakdown of trade entries by HMM component (state)."""
        entries = self._regime_entries
        n = len(entries)
        print(f'\n{"=" * 75}')
        print(f'  REGIME TRADE ANALYSIS  ({n} entries)')
        print(f'{"=" * 75}')

        # Group by HMM state number
        from collections import defaultdict
        by_state = defaultdict(lambda: {'count': 0, 'total_size': 0,
                                        'regimes': [], 'scores': []})
        for regime, size, dname, hmm_st, hmm_sc in entries:
            by_state[hmm_st]['count'] += 1
            by_state[hmm_st]['total_size'] += size
            by_state[hmm_st]['regimes'].append(regime)
            by_state[hmm_st]['scores'].append(hmm_sc)

        # Sort states by score (highest first)
        all_states = sorted(by_state.keys(),
                            key=lambda s: sum(by_state[s]['scores']) /
                                          max(by_state[s]['count'], 1),
                            reverse=True)

        print(f'\n  {"State":>7}  {"Score":>7}  {"Entries":>8}  {"% of Total":>10}  '
              f'{"Avg Size":>9}  {"Avg Regime":>11}  {"Bar":>20}')
        print(f'  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*10}  {"-"*9}  {"-"*11}  {"-"*20}')

        for st in all_states:
            info = by_state[st]
            cnt = info['count']
            pct = 100 * cnt / n
            avg_size = info['total_size'] / cnt
            avg_regime = sum(info['regimes']) / cnt
            avg_score = sum(info['scores']) / cnt
            bar_len = int(pct / 2)
            bar = '█' * bar_len
            label = f'S{st}' if st >= 0 else 'N/A'
            print(f'  {label:>7}  {avg_score:>7.3f}  {cnt:>8}  {pct:>9.1f}%  '
                  f'{avg_size:>9.1f}  {avg_regime:>11.3f}  {bar}')

        print(f'  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*10}  {"-"*9}  {"-"*11}')
        avg_all = sum(r for r, _, _, _, _ in entries) / n
        avg_sz  = sum(s for _, s, _, _, _ in entries) / n
        avg_sc  = sum(sc for _, _, _, _, sc in entries) / n
        print(f'  {"TOTAL":>7}  {avg_sc:>7.3f}  {n:>8}  {100:>9.1f}%  '
              f'{avg_sz:>9.1f}  {avg_all:>11.3f}')
        print(f'{"=" * 75}')
