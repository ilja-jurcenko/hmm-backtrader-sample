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
        stake    = 100,        # shares per trade per instrument
        printlog = True,       # print trade log to stdout
        use_hmm  = False,      # enable HMM regime gate
    )

    # ------------------------------------------------------------------ helpers

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    def _get_regime(self, d) -> int:
        """Return 1 (trade allowed) or 0 (suppressed by HMM)."""
        if self.p.use_hmm and hasattr(d, 'regime'):
            return int(d.regime[0])
        return 1

    # ------------------------------------------------------------------ lifecycle

    def __init__(self):
        self.inds   = {}
        self.orders = {}
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
            sig    = self._signal(d)

            if not pos.size:
                if sig > 0 and regime == 1:
                    self.log(f'[{dname}] BUY  signal  close={d.close[0]:.2f}  regime={regime}')
                    self.orders[d] = self.buy(data=d, size=self.p.stake)
            else:
                if sig < 0 or regime == 0:
                    reason = 'signal' if sig < 0 else 'regime=0'
                    self.log(f'[{dname}] SELL signal ({reason})  close={d.close[0]:.2f}')
                    self.orders[d] = self.sell(data=d, size=self.p.stake)

    def stop(self):
        self.log(
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
