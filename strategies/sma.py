#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
SMA Crossover Strategy
======================
Enters long when the fast Simple Moving Average crosses *above* the slow
SMA, exits when it crosses *below*.  Optionally gated by an HMM regime
signal.

CLI key: ``sma``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class SmaCrossOver(BaseRegimeStrategy):
    """
    Dual Simple Moving Average crossover, long-only.

    Buy  : fast SMA crosses above slow SMA (and regime == 1)
    Sell : fast SMA crosses below slow SMA (or regime flips to 0)
    """

    params = dict(
        fast           = 10,
        slow           = 30,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        fast = bt.ind.SMA(d.close, period=self.p.fast)
        slow = bt.ind.SMA(d.close, period=self.p.slow)
        self.inds[d]['crossover'] = bt.ind.CrossOver(fast, slow)

    def _signal(self, d) -> int:
        cross = self.inds[d]['crossover']
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def stop(self):
        self.log(
            f'SMA({self.p.fast}/{self.p.slow})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
