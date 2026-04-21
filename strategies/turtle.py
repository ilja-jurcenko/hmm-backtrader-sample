#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Turtle Trading Strategy
========================
Implements the core of Richard Dennis's Turtle Trading rules (System 1).

    Entry  : Close breaks above the N-day highest high  (default 20)
    Exit   : Close drops below the M-day lowest low     (default 10)
    Filter : Skip entry if the *last* signal of the same direction was
             a winning trade (classic Turtle rule 1 filter).  This filter
             is optional and disabled by default for simplicity in this
             framework.

The ATR-based position size ("N") from the original system is handled by
backtrader's base stake mechanism; the turtle_atr_mult parameter controls
an optional ATR trailing stop which is *separate* from the fixed
stop_loss_perc on the base class.

Optionally gated by the HMM regime signal.

CLI key: ``turtle``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class TurtleStrategy(BaseRegimeStrategy):
    """
    Turtle System 1: 20-day entry breakout, 10-day exit channel, long-only.

    Buy  : close > prior 20-bar high
    Sell : close < prior 10-bar low
           OR 2×ATR trailing stop breach (if turtle_atr_mult > 0)
    """

    params = dict(
        turtle_entry   = 20,    # entry channel look-back
        turtle_exit    = 10,    # exit channel look-back
        turtle_atr     = 20,    # ATR period for trailing stop
        turtle_atr_mult= 2.0,   # ATR multiplier for stop (0 = disabled)
        stake          = 100,
        printlog       = True,
        use_hmm        = False,
        regime_mode    = 'strict',
        unfav_fraction = 0.25,
    )

    def __init__(self):
        super().__init__()
        # Track entry price for ATR trailing stop
        self._entry_prices = {}
        for d in self.datas:
            self._entry_prices[d] = None

    def _init_indicators(self, d):
        self.inds[d]['entry_high'] = bt.ind.Highest(d.close, period=self.p.turtle_entry)
        self.inds[d]['exit_low']   = bt.ind.Lowest(d.close,  period=self.p.turtle_exit)
        self.inds[d]['atr']        = bt.ind.ATR(d,            period=self.p.turtle_atr)

    def _signal(self, d) -> int:
        entry_high  = self.inds[d]['entry_high']
        exit_low    = self.inds[d]['exit_low']
        atr         = self.inds[d]['atr']
        pos         = self.getposition(d)

        if not pos.size:
            if d.close[0] > entry_high[-1]:
                self._entry_prices[d] = d.close[0]
                return 1
        else:
            # Channel exit
            if d.close[0] < exit_low[-1]:
                self._entry_prices[d] = None
                return -1
            # ATR trailing stop
            if self.p.turtle_atr_mult > 0 and self._entry_prices[d] is not None:
                stop = self._entry_prices[d] - self.p.turtle_atr_mult * atr[0]
                if d.close[0] < stop:
                    self._entry_prices[d] = None
                    return -1
        return 0

    def stop(self):
        self.log(
            f'Turtle(entry={self.p.turtle_entry} exit={self.p.turtle_exit} '
            f'atr_mult={self.p.turtle_atr_mult})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
