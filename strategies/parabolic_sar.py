#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Parabolic SAR Strategy
=======================
Trades the Parabolic Stop-and-Reverse (PSAR) indicator.
The SAR trails the price and flips side when touched.

Entry rule (long):
    Close > SAR  (price has risen above the trailing stop → bullish)

Exit rule:
    Close < SAR  (price has fallen below the trailing stop → SAR flipped)

Optionally gated by the HMM regime signal.

CLI key: ``parabolic_sar``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


# ---------------------------------------------------------------------------
# Minimal Parabolic SAR custom indicator
# ---------------------------------------------------------------------------

class _ParabolicSAR(bt.Indicator):
    """
    Standard Parabolic SAR.

    Lines:
        sar  – the stop-and-reverse value
    """
    lines = ('sar',)
    params = dict(af=0.02, max_af=0.20)

    def __init__(self):
        # Seed computation in next(); nothing to chain-build here.
        self._bull   = True     # current trend direction
        self._af     = self.p.af
        self._ep     = None     # extreme point
        self._sar    = None     # current SAR

    def nextstart(self):
        # Initialise on first bar
        self._bull  = True
        self._ep    = self.data.high[0]
        self._sar   = self.data.low[0]
        self._af    = self.p.af
        self.lines.sar[0] = self._sar

    def next(self):
        high  = self.data.high[0]
        low   = self.data.low[0]
        ph    = self.data.high[-1]
        pl    = self.data.low[-1]

        sar = self._sar
        ep  = self._ep
        af  = self._af

        if self._bull:
            # Advance SAR
            new_sar = sar + af * (ep - sar)
            # SAR must be ≤ prior two lows
            new_sar = min(new_sar, pl, self.data.low[-2] if len(self) > 2 else pl)
            if low < new_sar:
                # Trend reversal → bearish
                self._bull = False
                new_sar    = ep
                self._ep   = low
                self._af   = self.p.af
            else:
                if high > ep:
                    self._ep = high
                    self._af = min(af + self.p.af, self.p.max_af)
        else:
            # Bearish
            new_sar = sar + af * (ep - sar)
            # SAR must be ≥ prior two highs
            new_sar = max(new_sar, ph, self.data.high[-2] if len(self) > 2 else ph)
            if high > new_sar:
                # Trend reversal → bullish
                self._bull = True
                new_sar    = ep
                self._ep   = high
                self._af   = self.p.af
            else:
                if low < ep:
                    self._ep = low
                    self._af = min(af + self.p.af, self.p.max_af)

        self._sar           = new_sar
        self.lines.sar[0]   = new_sar


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ParabolicSarStrategy(BaseRegimeStrategy):
    """
    Parabolic SAR trend-following, long-only.

    Buy  : close crosses above SAR (trend flipped bullish)
    Sell : close crosses below SAR (trend flipped bearish)
    """

    params = dict(
        psar_af        = 0.02,
        psar_max_af    = 0.20,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        self.inds[d]['sar'] = _ParabolicSAR(d, af=self.p.psar_af, max_af=self.p.psar_max_af)

    def _signal(self, d) -> int:
        sar   = self.inds[d]['sar'].sar
        close = d.close
        pos   = self.getposition(d)

        if not pos.size:
            # Price just crossed above the SAR (bullish flip)
            if close[-1] <= sar[-1] and close[0] > sar[0]:
                return 1
        else:
            # Price just crossed below the SAR (bearish flip)
            if close[-1] >= sar[-1] and close[0] < sar[0]:
                return -1
        return 0

    def stop(self):
        self.log(
            f'ParabolicSAR(af={self.p.psar_af} max_af={self.p.psar_max_af})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
