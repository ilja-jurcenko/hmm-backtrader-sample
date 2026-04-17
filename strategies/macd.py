#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
MACD Signal-Line Crossover Strategy
=====================================
Trades the classic MACD histogram / signal-line crossover.

    MACD line   = EMA(fast) − EMA(slow)
    Signal line = EMA(MACD, period=signal_period)
    Histogram   = MACD line − Signal line

Entry rule (long):
    MACD line crosses *above* the signal line (histogram turns positive).

Exit rule:
    MACD line crosses *below* the signal line (histogram turns negative),
    or the HMM regime flips to 0.

Optionally gated by the HMM regime signal.

CLI key: ``macd``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class MacdStrategy(BaseRegimeStrategy):
    """
    MACD line × signal line crossover, long-only.

    Buy  : MACD line crosses above signal line (and regime == 1)
    Sell : MACD line crosses below signal line (or regime == 0)
    """

    params = dict(
        macd_fast   = 12,     # fast EMA period for MACD
        macd_slow   = 26,     # slow EMA period for MACD
        macd_signal = 9,      # signal (smoothing) EMA period
        stake       = 100,
        printlog    = True,
        use_hmm     = False,
    )

    def _init_indicators(self, d):
        macd_ind = bt.ind.MACD(
            d.close,
            period_me1     = self.p.macd_fast,
            period_me2     = self.p.macd_slow,
            period_signal  = self.p.macd_signal,
        )
        self.inds[d]['macd']      = macd_ind.macd
        self.inds[d]['signal']    = macd_ind.signal
        # CrossOver > 0 when macd line ticks above signal line
        self.inds[d]['crossover'] = bt.ind.CrossOver(
            macd_ind.macd, macd_ind.signal)

    def _signal(self, d) -> int:
        cross = self.inds[d]['crossover']
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def stop(self):
        self.log(
            f'MACD({self.p.macd_fast}/{self.p.macd_slow}/{self.p.macd_signal})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
