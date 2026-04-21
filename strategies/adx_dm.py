#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
ADX + Directional Movement Strategy
=====================================
Uses the Average Directional Index (ADX) together with the +DI / -DI
directional-movement lines to trade trending markets.

Entry rule (long):
    +DI crosses above -DI  AND  ADX > adx_threshold
    (trend is bullish and strong enough to be worth trading)

Exit rule:
    -DI crosses above +DI  (trend has flipped bearish)
    OR  ADX drops below adx_threshold after entry (trend has faded)

Optionally gated by the HMM regime signal.

CLI key: ``adx_dm``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class AdxDmStrategy(BaseRegimeStrategy):
    """
    ADX + DI-crossover, long-only trend-following.

    Buy  : +DI crosses above -DI while ADX > threshold
    Sell : -DI crosses above +DI  OR  ADX drops below threshold
    """

    params = dict(
        adx_period     = 14,
        adx_threshold  = 25,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        dm = bt.ind.DirectionalMovement(d, period=self.p.adx_period)
        self.inds[d]['adx']      = dm.adx
        self.inds[d]['diplus']   = dm.plusDI
        self.inds[d]['diminus']  = dm.minusDI
        self.inds[d]['di_cross'] = bt.ind.CrossOver(dm.plusDI, dm.minusDI)

    def _signal(self, d) -> int:
        adx      = self.inds[d]['adx']
        di_cross = self.inds[d]['di_cross']
        pos      = self.getposition(d)

        if not pos.size:
            # +DI just crossed above -DI and trend is strong
            if di_cross[0] > 0 and adx[0] > self.p.adx_threshold:
                return 1
        else:
            # -DI crossed above +DI (bearish flip)
            if di_cross[0] < 0:
                return -1
            # Trend has weakened – exit early
            if adx[0] < self.p.adx_threshold:
                return -1
        return 0

    def stop(self):
        self.log(
            f'ADX+DM(period={self.p.adx_period} threshold={self.p.adx_threshold})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
