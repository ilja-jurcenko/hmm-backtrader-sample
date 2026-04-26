#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
RSI Mean-Reversion Strategy
============================
Trades mean-reversion signals based on the Relative Strength Index.

Entry rule  (long):
    RSI rises back above the *oversold* level after being below it.
    This captures the start of a bullish bounce.

Exit rule:
    RSI rises above the *overbought* level  → take profit / reduce risk.
    RSI drops back below 50               → momentum has faded, exit.
    Whichever comes first triggers the sell.

Optionally gated by the HMM regime signal.

CLI key: ``rsi``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class RsiStrategy(BaseRegimeStrategy):
    """
    RSI oversold-bounce entry, overbought or mid-line exit.

    Buy  : RSI crosses *above* the oversold threshold from below
           (i.e. was ≤ oversold last bar, now > oversold)
    Sell : RSI crosses *above* the overbought threshold  OR
           drops *below* 50 while in a position
    """

    params = dict(
        rsi_period     = 14,
        oversold       = 30,
        overbought     = 70,
        midline        = 50,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
        invert_regime  = True,   # RSI signals fire in high-vol regimes;
                                  # invert HMM gate so unfav = entry window.
    )

    def _init_indicators(self, d):
        # Only keep the raw RSI line; level-crossovers are computed inline
        # to avoid the backtrader _periodset cycle that arises when CrossOver
        # is given a Python int as the second argument.
        self.inds[d]['rsi'] = bt.ind.RSI(d.close, period=self.p.rsi_period)

    def _signal(self, d) -> int:
        rsi      = self.inds[d]['rsi']
        rsi_now  = rsi[0]
        rsi_prev = rsi[-1]
        pos      = self.getposition(d)

        if not pos.size:
            # RSI recovered from oversold: was ≤ oversold last bar, now above it
            if rsi_prev <= self.p.oversold < rsi_now:
                return 1
        else:
            # Take profit: RSI crossed above overbought
            if rsi_prev <= self.p.overbought < rsi_now:
                return -1
            # Momentum faded: RSI dropped back below mid-line
            if rsi_prev >= self.p.midline > rsi_now:
                return -1
        return 0

    def stop(self):
        self.log(
            f'RSI({self.p.rsi_period})  oversold={self.p.oversold}  '
            f'overbought={self.p.overbought}  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
