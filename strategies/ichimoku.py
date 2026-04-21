#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Ichimoku Cloud Strategy
========================
Trades the classic Ichimoku Kinko Hyo system using the Tenkan-sen /
Kijun-sen crossover as the trigger, confirmed by price position relative
to the Kumo (Cloud).

Entry rule (long):
    Tenkan-sen crosses above Kijun-sen  AND
    Close is above both Senkou Span A and Senkou Span B  (above the cloud)

Exit rule:
    Tenkan-sen crosses below Kijun-sen  OR
    Close drops below both Senkou Span A and Senkou Span B  (inside/below cloud)

Optionally gated by the HMM regime signal.

CLI key: ``ichimoku``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class IchimokuStrategy(BaseRegimeStrategy):
    """
    Ichimoku Cloud (Tenkan/Kijun cross + cloud filter), long-only.

    Buy  : Tenkan × above Kijun  AND  close above cloud
    Sell : Tenkan × below Kijun  OR   close falls below cloud
    """

    params = dict(
        tenkan         = 9,
        kijun          = 26,
        senkou         = 52,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        ichi = bt.ind.Ichimoku(
            d,
            tenkan   = self.p.tenkan,
            kijun    = self.p.kijun,
            senkou   = self.p.senkou,
        )
        self.inds[d]['tenkan']    = ichi.tenkan_sen
        self.inds[d]['kijun']     = ichi.kijun_sen
        self.inds[d]['span_a']    = ichi.senkou_span_a
        self.inds[d]['span_b']    = ichi.senkou_span_b
        self.inds[d]['tk_cross']  = bt.ind.CrossOver(ichi.tenkan_sen, ichi.kijun_sen)

    def _signal(self, d) -> int:
        tk_cross = self.inds[d]['tk_cross']
        span_a   = self.inds[d]['span_a']
        span_b   = self.inds[d]['span_b']
        close    = d.close[0]
        pos      = self.getposition(d)

        cloud_top    = max(span_a[0], span_b[0])
        cloud_bottom = min(span_a[0], span_b[0])

        if not pos.size:
            # Bullish TK cross AND price is above the cloud
            if tk_cross[0] > 0 and close > cloud_top:
                return 1
        else:
            # Bearish TK cross
            if tk_cross[0] < 0:
                return -1
            # Price fell inside or below the cloud
            if close < cloud_bottom:
                return -1
        return 0

    def stop(self):
        self.log(
            f'Ichimoku(tenkan={self.p.tenkan} kijun={self.p.kijun} senkou={self.p.senkou})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
