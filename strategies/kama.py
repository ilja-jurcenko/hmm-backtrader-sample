#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Kaufman Adaptive Moving Average (KAMA) Strategy
================================================
Trend-following strategy using Perry Kaufman's Adaptive Moving Average.

KAMA adapts its smoothing speed to the ratio of directional price movement
to total path length (the Efficiency Ratio).  During trending markets it
tracks price closely; during choppy/sideways markets it smooths heavily,
reducing whipsaws compared to a standard EMA.

Entry rule (long):
    Price crosses *above* KAMA (and KAMA is rising).

Exit rule:
    Price crosses *below* KAMA.

Optionally gated by the HMM regime signal.

CLI key: ``kama``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class KamaStrategy(BaseRegimeStrategy):
    """
    Kaufman Adaptive Moving Average crossover, long-only.

    Buy  : close crosses above KAMA  (close[-1] <= KAMA[-1], close[0] > KAMA[0])
    Sell : close crosses below KAMA  (close[-1] >= KAMA[-1], close[0] < KAMA[0])
    """

    params = dict(
        kama_period    = 10,    # Efficiency Ratio look-back window
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
        invert_regime  = False,  # trend-following: trade in low-vol, trending regimes
    )

    def _init_indicators(self, d):
        self.inds[d]['kama'] = bt.ind.KAMA(d.close, period=self.p.kama_period)

    def _signal(self, d) -> int:
        kama       = self.inds[d]['kama']
        pos        = self.getposition(d)

        close_now  = d.close[0]
        close_prev = d.close[-1]
        kama_now   = kama[0]
        kama_prev  = kama[-1]

        if not pos.size:
            # Price crossed above KAMA
            if close_prev <= kama_prev and close_now > kama_now:
                return 1
        else:
            # Price crossed below KAMA
            if close_prev >= kama_prev and close_now < kama_now:
                return -1
        return 0

    def stop(self):
        self.log(
            f'KAMA({self.p.kama_period})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
