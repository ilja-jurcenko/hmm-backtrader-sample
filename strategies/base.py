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
        self._all_state_scores = {} # state_id → list of scores (from every bar)
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

            # Track all states seen (for regime analysis of 0-trade states)
            if hmm_st >= 0 and hmm_st not in self._all_state_scores:
                self._all_state_scores[hmm_st] = hmm_sc

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

        # Group by HMM state number
        from collections import defaultdict
        by_state = defaultdict(lambda: {'count': 0, 'total_size': 0,
                                        'regimes': [], 'scores': []})
        for regime, size, dname, hmm_st, hmm_sc in entries:
            by_state[hmm_st]['count'] += 1
            by_state[hmm_st]['total_size'] += size
            by_state[hmm_st]['regimes'].append(regime)
            by_state[hmm_st]['scores'].append(hmm_sc)

        # Include states that were seen but had 0 trades
        for st, sc in self._all_state_scores.items():
            if st not in by_state:
                by_state[st] = {'count': 0, 'total_size': 0,
                                'regimes': [], 'scores': [sc]}

        # Sort states by score (highest first)
        all_states = sorted(by_state.keys(),
                            key=lambda s: (sum(by_state[s]['scores']) /
                                           max(len(by_state[s]['scores']), 1)),
                            reverse=True)

        n_active = sum(1 for s in all_states if by_state[s]['count'] > 0)

        # Build rows
        rows = []
        for rank, st in enumerate(all_states, 1):
            info = by_state[st]
            cnt = info['count']
            pct = (100 * cnt / n) if n > 0 else 0
            avg_size = info['total_size'] / cnt if cnt > 0 else 0
            avg_regime = sum(info['regimes']) / cnt if cnt > 0 else 0
            avg_score = sum(info['scores']) / len(info['scores']) if info['scores'] else 0
            bar_len = max(1, int(pct / 2)) if cnt > 0 else 0
            bar = '█' * bar_len
            label = f'S{st}' if st >= 0 else 'N/A'
            rows.append((f'#{rank}', label, avg_score, cnt, pct, avg_size, avg_regime, bar))

        avg_all = sum(r for r, _, _, _, _ in entries) / n if n > 0 else 0
        avg_sz  = sum(s for _, s, _, _, _ in entries) / n if n > 0 else 0
        avg_sc  = sum(sc for _, _, _, _, sc in entries) / n if n > 0 else 0

        W = 80
        print(f'\n┌{"─" * (W-2)}┐')
        print(f'│{"REGIME TRADE ANALYSIS":^{W-2}}│')
        print(f'│{f"{n} entries · {len(all_states)} states ({n_active} active)":^{W-2}}│')
        print(f'├{"─" * (W-2)}┤')
        print(f'│  {"Rank":>4}  {"State":>5}  {"Score":>6}  '
              f'{"Entries":>6}  {"%":>6}  {"AvgSz":>6}  {"PosSz":>6}  '
              f'{"Distribution":<22} │')
        print(f'│  {"────":>4}  {"─────":>5}  {"──────":>6}  '
              f'{"──────":>6}  {"──────":>6}  {"──────":>6}  {"──────":>6}  '
              f'{"─" * 22} │')

        for rank_s, label, avg_score, cnt, pct, avg_size, avg_regime, bar in rows:
            entries_s = str(cnt) if cnt > 0 else '—'
            pct_s = f'{pct:5.1f}%' if cnt > 0 else '    —'
            sz_s = f'{avg_size:6.0f}' if cnt > 0 else '     —'
            ps_s = f'{avg_regime:6.3f}' if cnt > 0 else '     —'
            print(f'│  {rank_s:>4}  {label:>5}  {avg_score:>6.2f}  '
                  f'{entries_s:>6}  {pct_s:>6}  {sz_s:>6}  {ps_s:>6}  '
                  f'{bar:<22} │')

        print(f'├{"─" * (W-2)}┤')
        print(f'│  {"":>4}  {"Total":>5}  {avg_sc:>6.2f}  '
              f'{n:>6}  {"100%":>6}  {avg_sz:>6.0f}  {avg_all:>6.3f}  '
              f'{"":22} │')
        print(f'└{"─" * (W-2)}┘')
