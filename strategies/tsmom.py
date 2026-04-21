#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Time-Series Momentum Strategy (TSMOM)
=======================================
Trades the sign of the instrument's own past return.  The classic version
(Mosowitz, Ooi & Pedersen, 2012) uses the 12-month return *excluding* the
most recent month (12-1 momentum) to avoid short-term reversal noise.

Entry rule (long):
    12-1 month return is positive  (last-12-months return > 0, skipping last 1 month)
    i.e.  close[-(skip)] / close[-(lookback+skip)] > 1

Exit rule:
    12-1 month return turns negative

Optionally gated by the HMM regime signal.

CLI key: ``tsmom``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class TsmomStrategy(BaseRegimeStrategy):
    """
    Time-series momentum: long when own past-return is positive.

    Buy  : close[−skip] / close[−(lookback+skip)] > 1  (return turned positive)
    Sell : same ratio < 1  (return turned negative)
    """

    params = dict(
        tsmom_lookback = 252,   # approx 12 months (trading days)
        tsmom_skip     = 21,    # approx 1 month  (skip recent reversal)
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def _init_indicators(self, d):
        # No declarative indicators needed – signal computed directly in _signal
        pass

    def _signal(self, d) -> int:
        skip     = self.p.tsmom_skip
        lookback = self.p.tsmom_lookback
        needed   = lookback + skip

        if len(d) < needed + 1:
            return 0

        close     = d.close
        recent    = close[-skip]            # price ~1 month ago
        past      = close[-(lookback+skip)] # price ~13 months ago
        ret_sign  = 1 if recent > past else -1

        pos = self.getposition(d)
        if not pos.size:
            if ret_sign > 0:
                return 1
        else:
            if ret_sign < 0:
                return -1
        return 0

    def stop(self):
        self.log(
            f'TSMOM(lookback={self.p.tsmom_lookback} skip={self.p.tsmom_skip})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
