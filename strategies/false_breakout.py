#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
False Breakout (Fakeout) Strategy
==================================
A false breakout, or fakeout, is a move (and subsequent close) beyond a
support or resistance level that immediately reverses and fails to hold the
broken level as a new support/resistance.

This strategy trades the long side of a *false breakdown*:

1. Breakdown bar: previous bar closes **below** the N-period support level
   (the lowest close of the N bars that preceded it).  This looks like a
   bearish breakdown.

2. Reversal bar: the very next bar (today) closes **back above** that same
   support level.  The bears could not sustain the move — the "breakdown"
   was a fakeout.

Entry rule (long):
    close[-1] < support[-2]   (yesterday closed below prior N-bar low)
    close[ 0] > support[-2]   (today closed back above that level)

Exit rules:
    close[0] >= resistance[-1]  → price reached the N-bar resistance; take profit.
    close[0] <  support[-1]     → breakdown resumed; stop out.

Because fakeouts occur predominantly in high-volatility/choppy regimes that the
HMM labels "unfavourable" for trend strategies, the HMM gate is inverted by
default so those regimes become the *entry* window.

CLI key: ``false_breakout``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class FalseBreakoutStrategy(BaseRegimeStrategy):
    """
    False-breakdown fakeout, long-only.

    Buy  : yesterday closed below the prior N-bar support AND
           today closes back above that support (failed breakdown)
    Sell : today closes at or above the N-bar resistance (target reached) OR
           today closes below the current N-bar support (breakdown resumed)
    """

    params = dict(
        fb_period      = 20,    # channel look-back for support / resistance
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
        invert_regime  = True,  # fakeouts fire in high-vol regimes HMM dislikes
    )

    def _init_indicators(self, d):
        # Lowest/Highest over fb_period bars (includes the current bar).
        # indicator[-1] = value computed at the *previous* bar (no look-ahead).
        # indicator[-2] = value computed two bars ago (prior-to-breakdown level).
        self.inds[d]['support']    = bt.ind.Lowest(d.close,  period=self.p.fb_period)
        self.inds[d]['resistance'] = bt.ind.Highest(d.close, period=self.p.fb_period)

    def _signal(self, d) -> int:
        support    = self.inds[d]['support']
        resistance = self.inds[d]['resistance']
        pos        = self.getposition(d)

        # The support level *before* yesterday's bar (does not include yesterday).
        # Using this avoids a circular condition: if yesterday was the lowest
        # close, support[-1] == close[-1] and close[-1] < support[-1] is
        # never True.  support[-2] is the floor that was broken.
        prior_support = support[-2]

        if not pos.size:
            breakdown  = d.close[-1] < prior_support  # yesterday faked-out below
            recovered  = d.close[0]  > prior_support  # today reclaimed support
            if breakdown and recovered:
                return 1
        else:
            # Target hit: price reached the N-bar resistance
            if d.close[0] >= resistance[-1]:
                return -1
            # Breakdown resumed: support gave way again
            if d.close[0] < support[-1]:
                return -1
        return 0

    def stop(self):
        self.log(
            f'FalseBreakout(period={self.p.fb_period})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
