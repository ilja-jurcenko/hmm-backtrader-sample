#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Channel Breakout Strategy
==========================
Buys when price closes above the N-period highest high (upside breakout)
and sells when price closes below the N-period lowest low (downside break).
A single ``channel_period`` governs both the entry and exit channel.

Entry rule (long):
    Close > Highest(close, channel_period) of the *previous* N bars
    (i.e. close exceeds the prior channel top, signalling breakout)

Exit rule:
    Close < Lowest(close, channel_period) of the *previous* N bars

Optionally gated by the HMM regime signal.

CLI key: ``channel_breakout``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class ChannelBreakoutStrategy(BaseRegimeStrategy):
    """
    N-period price-channel breakout, long-only.

    Buy  : close > rolling N-bar high  (new breakout high)
    Sell : close < rolling N-bar low
    """

    params = dict(
        channel_period = 20,
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        # Highest / Lowest over channel_period bars (includes current bar,
        # so compare close[0] > highest[-1] to detect a true new high)
        self.inds[d]['highest'] = bt.ind.Highest(d.close, period=self.p.channel_period)
        self.inds[d]['lowest']  = bt.ind.Lowest(d.close,  period=self.p.channel_period)

    def _signal(self, d) -> int:
        highest = self.inds[d]['highest']
        lowest  = self.inds[d]['lowest']
        pos     = self.getposition(d)

        if not pos.size:
            # Close breaks above prior channel top
            if d.close[0] > highest[-1]:
                return 1
        else:
            # Close drops below prior channel bottom
            if d.close[0] < lowest[-1]:
                return -1
        return 0

    def stop(self):
        self.log(
            f'ChannelBreakout(period={self.p.channel_period})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
