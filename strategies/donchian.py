#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Donchian Channel Strategy
==========================
Trades the Donchian Channel breakout using *separate* entry and exit periods,
giving finer control over the exit than the simpler ChannelBreakout strategy.

Entry rule (long):
    Close > Highest(close, donchian_entry) of the previous N bars

Exit rule:
    Close < Lowest(close, donchian_exit) of the previous M bars
    (M < N  → tighter stop that exits positions sooner)

Optionally gated by the HMM regime signal.

CLI key: ``donchian``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class DonchianStrategy(BaseRegimeStrategy):
    """
    Donchian channel with independent entry / exit lookbacks, long-only.

    Buy  : close > prior donchian_entry-bar high
    Sell : close < prior donchian_exit-bar low
    """

    params = dict(
        donchian_entry = 20,   # breakout entry channel
        donchian_exit  = 10,   # exit channel (shorter = tighter stop)
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        self.inds[d]['entry_high'] = bt.ind.Highest(d.close, period=self.p.donchian_entry)
        self.inds[d]['exit_low']   = bt.ind.Lowest(d.close,  period=self.p.donchian_exit)

    def _signal(self, d) -> int:
        entry_high = self.inds[d]['entry_high']
        exit_low   = self.inds[d]['exit_low']
        pos        = self.getposition(d)

        if not pos.size:
            if d.close[0] > entry_high[-1]:
                return 1
        else:
            if d.close[0] < exit_low[-1]:
                return -1
        return 0

    def stop(self):
        self.log(
            f'Donchian(entry={self.p.donchian_entry} exit={self.p.donchian_exit})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
