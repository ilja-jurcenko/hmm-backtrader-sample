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
        stake           = 100,        # shares per trade per instrument
        printlog        = True,       # print trade log to stdout
        use_hmm         = False,      # enable HMM regime gate
        regime_mode     = 'strict',   # 'strict' = block trades in unfav regime
                                      # 'size'   = reduce position size instead
        unfav_fraction  = 0.25,       # fraction of stake used in unfavourable regime
                                      # (only used when regime_mode='size')
        stop_loss_perc  = 0.02,       # stop-loss  as fraction (0 = disabled)
        take_profit_perc = 0.10,      # take-profit as fraction (0 = disabled)
        invert_regime   = False,      # invert the HMM gate: trade in 'unfavourable'
                                      # (high-vol/choppy) states instead of trending ones.
                                      # Set True for mean-reversion strategies.
    )

    # ------------------------------------------------------------------ helpers

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    def _get_regime(self, d) -> float:
        """Return regime value: 1.0 (fully allowed), 0.0 (suppressed),
        or a float in [0,1] for score-based sizing.

        When invert_regime=True the signal is flipped so that states the HMM
        labels 'unfavourable' (high-vol / choppy) become the entry window.
        This is appropriate for mean-reversion strategies: their signals
        (e.g. RSI oversold) fire almost exclusively in volatile regimes, not
        in the low-vol trending regimes the HMM normally favours.

        Inversion formula: max(0.0, 1.0 - raw_regime)
          strict/size mode (raw ∈ {0,1}): 0→1, 1→0  — clean binary flip.
          score/linear mode (raw ∈ [0, max_pos_size]): low-score unfav states
            become high-score entry windows; scores above 1.0 collapse to 0.
        """
        if self.p.use_hmm and hasattr(d, 'regime'):
            val = float(d.regime[0])
            if self.p.invert_regime:
                val = max(0.0, 1.0 - val)
            return val
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
        # Per-component profitability tracking
        self._entry_state = {}      # dname → hmm_state at entry
        self._entry_price = {}      # dname → close price at entry
        self._entry_size  = {}      # dname → position size at entry
        self._regime_pnl  = {}      # hmm_state → list of (pnlcomm, return_pct)
        # Collect state_pos_sizes from data feeds (set by ma-quantstats.py)
        self._state_pos_sizes = {}  # state_id → pos_size fraction
        self._bracket_children = {}   # d → [stop_order, limit_order] or []
        self._bracket_roles    = {}   # order_ref → 'STOP-LOSS' | 'TAKE-PROFIT'
        for d in self.datas:
            self.orders[d] = None
            self._bracket_children[d] = []
            self.inds[d]   = {}
            self._init_indicators(d)
            if hasattr(d, 'state_pos_sizes'):
                self._state_pos_sizes.update(d.state_pos_sizes)

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

        # ── bracket child (stop-loss / take-profit) ──
        for d, children in self._bracket_children.items():
            if order in children:
                children.remove(order)
                if order.status == order.Completed:
                    tag = self._bracket_roles.get(order.ref, 'EXIT')
                    self.log(
                        f'[{dname}] {tag} hit  '
                        f'price={order.executed.price:.2f}  '
                        f'comm={order.executed.comm:.2f}')
                # Canceled means its sibling already closed the position — ignore
                self._bracket_roles.pop(order.ref, None)
                self.orders[d] = None
                return

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
            # Record PnL by HMM component
            st = self._entry_state.pop(dname, -1)
            entry_px = self._entry_price.pop(dname, None)
            entry_sz = self._entry_size.pop(dname, 0)
            ret_pct = (trade.pnl / (entry_px * entry_sz) * 100) if entry_px and entry_sz else 0.0
            self._regime_pnl.setdefault(st, []).append((trade.pnlcomm, ret_pct))

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
                    use_bracket = (self.p.stop_loss_perc > 0
                                   or self.p.take_profit_perc > 0)
                    if use_bracket:
                        kw = dict(data=d, size=size, exectype=bt.Order.Market)
                        if self.p.stop_loss_perc > 0:
                            kw['stopprice'] = d.close[0] * (1.0 - self.p.stop_loss_perc)
                        if self.p.take_profit_perc > 0:
                            kw['limitprice'] = d.close[0] * (1.0 + self.p.take_profit_perc)
                        bracket = self.buy_bracket(**kw)
                        self.orders[d] = bracket[0]
                        self._bracket_children[d] = [bracket[1], bracket[2]]
                        self._bracket_roles[bracket[1].ref] = 'STOP-LOSS'
                        self._bracket_roles[bracket[2].ref] = 'TAKE-PROFIT'
                        sl_s = (f'  SL={kw["stopprice"]:.2f}'
                                if self.p.stop_loss_perc > 0 else '')
                        tp_s = (f'  TP={kw["limitprice"]:.2f}'
                                if self.p.take_profit_perc > 0 else '')
                        self.log(f'[{dname}] BRACKET placed{sl_s}{tp_s}')
                    else:
                        self.orders[d] = self.buy(data=d, size=size)
                    self._regime_entries.append((regime, size, dname, hmm_st, hmm_sc))
                    self._entry_state[dname] = hmm_st
                    self._entry_price[dname] = d.close[0]
                    self._entry_size[dname]  = size
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
                    elif pos.size < target:
                        add = target - pos.size
                        self.log(f'[{dname}] EXPAND position ({pos.size}→{target})  '
                                 f'regime={regime:.2f}  close={d.close[0]:.2f}')
                        self.orders[d] = self.buy(data=d, size=add)
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
                    # Cancel any live bracket children before issuing manual exit
                    for child in list(self._bracket_children.get(d, [])):
                        self.cancel(child)
                    self._bracket_children[d] = []
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

        # Compute profitability per state
        state_profit = {}  # st → (net_pnl, avg_pnl, win_rate, avg_ret)
        total_net_pnl = 0.0
        total_trades_closed = 0
        total_wins = 0
        total_ret_sum = 0.0
        for st in all_states:
            pnl_list = self._regime_pnl.get(st, [])
            if pnl_list:
                net_pnl  = sum(p for p, _ in pnl_list)
                avg_pnl  = net_pnl / len(pnl_list)
                wins     = sum(1 for p, _ in pnl_list if p > 0)
                win_rate = 100 * wins / len(pnl_list)
                avg_ret  = sum(r for _, r in pnl_list) / len(pnl_list)
                total_net_pnl += net_pnl
                total_trades_closed += len(pnl_list)
                total_wins += wins
                total_ret_sum += sum(r for _, r in pnl_list)
            else:
                net_pnl = avg_pnl = win_rate = avg_ret = 0.0
            state_profit[st] = (net_pnl, avg_pnl, win_rate, avg_ret)

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
            net_pnl, avg_pnl, win_rate, avg_ret = state_profit[st]
            rows.append((f'#{rank}', label, avg_score, cnt, pct, avg_size, avg_regime,
                         net_pnl, avg_pnl, win_rate, avg_ret, bar))

        avg_all = sum(r for r, _, _, _, _ in entries) / n if n > 0 else 0
        avg_sz  = sum(s for _, s, _, _, _ in entries) / n if n > 0 else 0
        avg_sc  = sum(sc for _, _, _, _, sc in entries) / n if n > 0 else 0
        tot_avg_pnl = total_net_pnl / total_trades_closed if total_trades_closed else 0
        tot_win_rate = 100 * total_wins / total_trades_closed if total_trades_closed else 0
        tot_avg_ret = total_ret_sum / total_trades_closed if total_trades_closed else 0

        # ── table formatting helpers ──
        import unicodedata

        def _dw(s):
            """Display width of string (accounts for wide/fullwidth chars)."""
            return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1
                       for c in s)

        def _rpad(s, w):
            """Left-align s in w display-columns."""
            return s + ' ' * (w - _dw(s))

        def _lpad(s, w):
            """Right-align s in w display-columns."""
            return ' ' * (w - _dw(s)) + s

        COL = [
            # (header,        width, align)
            ('Rank',            4, '>'),
            ('Comp',            4, '>'),
            ('Score',           5, '>'),
            ('CompSz',          6, '>'),
            ('Entries',         7, '>'),
            ('%',               6, '>'),
            ('PosSz',           5, '>'),
            ('Net PnL',        10, '>'),
            ('Avg PnL',         9, '>'),
            ('Win%',            5, '>'),
            ('AvgRet%',         7, '>'),
            ('Distribution',   20, '<'),
        ]
        GAP = '  '

        def _row(cells):
            parts = []
            for (_, w, align), val in zip(COL, cells):
                s = str(val)
                parts.append(_lpad(s, w) if align == '>' else _rpad(s, w))
            return GAP.join(parts)

        hdr_line = _row([h for h, _, _ in COL])
        sep_line = _row(['-' * w for _, w, _ in COL])
        table_w = _dw(hdr_line) + 6   # "│  " + content + "  │"

        def _hline(l, r):
            return l + '─' * (table_w - 2) + r

        def _print_row(content):
            pad = table_w - 2 - _dw(content) - 4  # 4 = leading + trailing spaces
            print(f'│  {content}{" " * max(pad, 0)}  │')

        subtitle = (f"{n} entries | {len(all_states)} components "
                    f"({n_active} active) | {total_trades_closed} closed trades")

        print(f'\n{_hline("┌", "┐")}')
        title_pad = table_w - 2 - len('REGIME TRADE ANALYSIS')
        lp = title_pad // 2
        rp = title_pad - lp
        print(f'│{" " * lp}REGIME TRADE ANALYSIS{" " * rp}│')
        sub_pad = table_w - 2 - len(subtitle)
        lp2 = sub_pad // 2
        rp2 = sub_pad - lp2
        print(f'│{" " * lp2}{subtitle}{" " * rp2}│')
        print(f'{_hline("├", "┤")}')
        _print_row(hdr_line)
        _print_row(sep_line)

        for (rank_s, label, avg_score, cnt, pct, avg_size, avg_regime,
             net_pnl, avg_pnl, win_rate, avg_ret, bar) in rows:
            st_key = int(label[1:]) if label != 'N/A' else -1
            has_pnl = bool(self._regime_pnl.get(st_key, []))
            has_trades = cnt > 0

            cells = [
                rank_s,
                label,
                f'{avg_score:.2f}',
                f'{self._state_pos_sizes.get(st_key, 0):.3f}' if st_key in self._state_pos_sizes else '-',
                str(cnt) if has_trades else '-',
                f'{pct:.1f}%' if has_trades else '-',
                f'{avg_regime:.3f}' if has_trades else '-',
                f'{net_pnl:+,.1f}' if has_pnl else '-',
                f'{avg_pnl:+,.1f}' if has_pnl else '-',
                f'{win_rate:.0f}%' if has_pnl else '-',
                f'{avg_ret:+.2f}%' if has_pnl else '-',
                bar,
            ]
            _print_row(_row(cells))

        print(f'{_hline("├", "┤")}')
        tot_cells = [
            '',
            'ALL',
            f'{avg_sc:.2f}',
            '',
            str(n),
            '',
            f'{avg_all:.3f}',
            f'{total_net_pnl:+,.1f}',
            f'{tot_avg_pnl:+,.1f}',
            f'{tot_win_rate:.0f}%',
            f'{tot_avg_ret:+.2f}%',
            '',
        ]
        _print_row(_row(tot_cells))
        print(f'{_hline("└", "┘")}')
