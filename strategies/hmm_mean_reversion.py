#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
HMM Mean-Reversion Strategy
=============================
The HMM model is used as the *signal generator*, not a regime filter.

The model is trained on closing prices, so each hidden state has a
Gaussian distribution whose mean (μ) and std (σ) represent the typical
price level for that regime.

Core intuition
--------------
Within a regime, prices tend to oscillate around the state mean.
If the current price is below the state mean the market is "cheap" relative
to the regime centroid → long.  As soon as price reverts back to (or above)
the mean we close the position.

Mechanics
---------
Entry  : close < state_mean − z_threshold × state_std   (price dipped below mean)
Exit   : close ≥ state_mean                              (mean-reversion achieved)
         OR state flips and new state_mean < close        (no longer undervalued)

The data feed must expose two custom lines pre-computed by ma-quantstats.py:
    d.state_mean[0]  – predicted state's mean price (from model.means_)
    d.state_std[0]   – predicted state's std  price (from model.covars_)

HMM training and line injection are handled by ma-quantstats.py before
cerebro.run() is called.  The strategy itself carries no ML code.

CLI key: ``hmm_mr``
"""
from __future__ import annotations

import backtrader as bt


class HmmMeanReversionStrategy(bt.Strategy):
    """
    HMM-native mean-reversion: buy the dip below the state mean,
    exit when price reverts.

    Parameters
    ----------
    z_threshold : float
        Minimum number of standard deviations the price must be below the
        state mean before an entry is triggered.
        0.0  → enter whenever  close < state_mean  (any dip)
        0.5  → enter when      close < state_mean − 0.5 × state_std
        1.0  → enter on deeper dips only (more conservative)
    stake : int
        Shares per trade per instrument.
    printlog : bool
        Print trade-log entries to stdout.
    """

    params = dict(
        z_threshold = 0.0,   # std-dev threshold for entry
        stake       = 100,
        printlog    = True,
    )

    # ------------------------------------------------------------------ helpers

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()}  {txt}')

    # ------------------------------------------------------------------ lifecycle

    def __init__(self):
        self.orders = {d: None for d in self.datas}
        # Verify feeds have the required custom lines
        for d in self.datas:
            if not hasattr(d, 'state_mean'):
                raise RuntimeError(
                    f'Feed "{d._name}" is missing the state_mean line.  '
                    'Ensure ma-quantstats.py is using HMMStateFeed for this strategy.')
            if not hasattr(d, 'state_std'):
                raise RuntimeError(
                    f'Feed "{d._name}" is missing the state_std line.  '
                    'Ensure ma-quantstats.py is using HMMStateFeed for this strategy.')

    # ------------------------------------------------------------------ events

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        dname = order.data._name or order.data._dataname
        if order.status == order.Completed:
            verb = 'BUY ' if order.isbuy() else 'SELL'
            self.log(
                f'[{dname}] {verb} executed  price={order.executed.price:.2f}'
                f'  cost={order.executed.value:.2f}'
                f'  comm={order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'[{dname}] Order canceled / margin / rejected')
        self.orders[order.data] = None

    def notify_trade(self, trade):
        if trade.isclosed:
            dname = trade.data._name or trade.data._dataname
            self.log(
                f'[{dname}] TRADE CLOSED  '
                f'gross={trade.pnl:.2f}  net={trade.pnlcomm:.2f}')

    # ------------------------------------------------------------------ core logic

    def next(self):
        for d in self.datas:
            if self.orders[d]:
                continue   # pending order

            close      = d.close[0]
            mu         = d.state_mean[0]
            sigma      = d.state_std[0]
            entry_lvl  = mu - self.p.z_threshold * sigma
            pos        = self.getposition(d)
            dname      = d._name or d._dataname

            if not pos.size:
                # Enter if price is below the regime mean (minus the threshold)
                if close < entry_lvl:
                    self.log(
                        f'[{dname}] BUY   close={close:.2f}  '
                        f'state_mean={mu:.2f}  '
                        f'entry_level={entry_lvl:.2f}  '
                        f'z={(mu - close) / max(sigma, 1e-9):.2f}σ')
                    self.orders[d] = self.buy(data=d, size=self.p.stake)
            else:
                # Exit when price has reverted to (or above) the state mean
                if close >= mu:
                    self.log(
                        f'[{dname}] SELL  close={close:.2f}  '
                        f'state_mean={mu:.2f}  (mean reached)')
                    self.orders[d] = self.sell(data=d, size=self.p.stake)

    # ------------------------------------------------------------------ summary

    def stop(self):
        self.log(
            f'HMM-MR(z={self.p.z_threshold})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
