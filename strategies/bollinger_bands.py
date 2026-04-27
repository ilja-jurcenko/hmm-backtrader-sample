#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Bollinger Bands Mean-Reversion Strategy
========================================
Trades mean-reversion signals based on Bollinger Bands.

Entry rule (long):
    Price closes below the lower band on the previous bar and then closes
    back above it — signals a bounce off the oversold extreme.

Exit rules:
    Price closes above the middle band (SMA) → partial reversion, take profit.
    Price closes above the upper band        → full reversion, close position.
    Whichever comes first triggers the sell.

Because Bollinger Band extremes coincide with high-volatility regimes the HMM
labels "unfavourable", the HMM gate is inverted by default so those regimes
become the entry window.

CLI key: ``bollinger``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class BollingerBandsStrategy(BaseRegimeStrategy):
    """
    Bollinger Bands oversold-bounce entry, mid/upper-band exit.

    Buy  : close was below lower band last bar, closes above it this bar
           (bounce from oversold extreme)
    Sell : close crosses above the middle band (SMA)  OR
           close crosses above the upper band
    """

    params = dict(
        bb_period      = 20,
        bb_devfactor   = 2.0,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
        invert_regime  = True,   # mean-reversion fires in high-vol regimes
    )

    def _init_indicators(self, d):
        self.inds[d]['bb'] = bt.ind.BollingerBands(
            d.close,
            period    = self.p.bb_period,
            devfactor = self.p.bb_devfactor,
        )

    def _signal(self, d) -> int:
        bb  = self.inds[d]['bb']
        pos = self.getposition(d)

        close_now  = d.close[0]
        close_prev = d.close[-1]
        lower_now  = bb.lines.bot[0]
        lower_prev = bb.lines.bot[-1]
        mid_now    = bb.lines.mid[0]
        upper_now  = bb.lines.top[0]

        if not pos.size:
            # Bounce: was below lower band, now back above it
            if close_prev <= lower_prev and close_now > lower_now:
                return 1
        else:
            # Reversion to mean: close crossed above middle band
            if close_now >= mid_now:
                return -1
            # Full reversion: close reached upper band
            if close_now >= upper_now:
                return -1
        return 0

    def stop(self):
        self.log(
            f'BollingerBands({self.p.bb_period}, {self.p.bb_devfactor})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
