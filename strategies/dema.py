#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
DEMA Crossover Strategy
=======================
Double Exponential Moving Average crossover.

DEMA = 2·EMA(n) − EMA(EMA(n))

Because DEMA responds to price changes faster than a plain EMA of the same
period, crossover signals tend to be earlier and cleaner in trending markets.

CLI key: ``dema``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class DemaCrossOver(BaseRegimeStrategy):
    """
    Dual DEMA crossover, long-only.

    Buy  : fast DEMA crosses above slow DEMA (and regime == 1)
    Sell : fast DEMA crosses below slow DEMA (or regime flips to 0)
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
        fast = bt.ind.DEMA(d.close, period=self.p.fast)
        slow = bt.ind.DEMA(d.close, period=self.p.slow)
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
            f'DEMA({self.p.fast}/{self.p.slow})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
