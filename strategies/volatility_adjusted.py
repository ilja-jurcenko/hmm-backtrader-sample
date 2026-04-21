#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Volatility-Adjusted Strategy  (Keltner Channel Breakout)
==========================================================
Uses a Keltner Channel – an EMA centre-line flanked by ±N×ATR bands –
as a volatility-normalised breakout signal.  Unlike fixed-percentage or
standard-deviation bands, ATR-based bands automatically widen in choppy
markets and contract in quiet ones, providing adaptive sensitivity.

Entry rule (long):
    Close breaks above the upper Keltner band
    (Upper = EMA(close, vol_period) + vol_atr_mult × ATR(vol_atr_period))

Exit rule:
    Close falls back below the EMA centre-line
    (momentum has evaporated)

Optionally gated by the HMM regime signal.

CLI key: ``vol_adj``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class VolatilityAdjustedStrategy(BaseRegimeStrategy):
    """
    Keltner Channel breakout with ATR-based dynamic bands, long-only.

    Buy  : close > EMA + vol_atr_mult × ATR  (breakout above upper band)
    Sell : close < EMA                         (price returns to midline)
    """

    params = dict(
        vol_period     = 20,    # EMA/ATR period
        vol_atr_period = 14,    # ATR period for band width
        vol_atr_mult   = 1.5,   # multiplier for the upper/lower band
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        self.inds[d]['ema']   = bt.ind.EMA(d.close, period=self.p.vol_period)
        self.inds[d]['atr']   = bt.ind.ATR(d,       period=self.p.vol_atr_period)

    def _signal(self, d) -> int:
        ema   = self.inds[d]['ema']
        atr   = self.inds[d]['atr']
        close = d.close[0]
        pos   = self.getposition(d)

        upper_band = ema[0] + self.p.vol_atr_mult * atr[0]

        if not pos.size:
            # Upside volatility breakout
            if close > upper_band:
                return 1
        else:
            # Price returned to the EMA midline – exit
            if close < ema[0]:
                return -1
        return 0

    def stop(self):
        self.log(
            f'VolAdj/Keltner(period={self.p.vol_period} '
            f'atr={self.p.vol_atr_period} mult={self.p.vol_atr_mult})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
