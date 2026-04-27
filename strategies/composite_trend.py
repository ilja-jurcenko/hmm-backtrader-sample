#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Composite Trend-Pullback Strategy
===================================
Combines four confirming filters so that trades are only placed when a
strong trend is in place AND price has pulled back into a well-defined zone
AND momentum is turning in the direction of the trend.

Periods are calibrated for a ~1-year testing window (252 trading days):
    EMA fast  =  10  (≈ 2 weeks momentum)
    EMA slow  =  30  (≈ 6 weeks trend anchor)
    RSI       =   7  (responsive to short-term oscillations)
    ADX       =   7  (faster DI computation)
    MACD      = 5 / 13 / 4

🟢 BUY Setup — all four must fire on the same bar:
    1. EMA(10)  >  EMA(30)                  — up-trend
    2. ADX(7)   >  25                       — trend is strong
    3. RSI(7) was in [30, 40] last bar AND
       RSI(7) is rising this bar            — healthy pullback reversing
    4. MACD(5,13,4) line crosses above
       signal line on this bar              — momentum turning bullish

🔴 SELL / EXIT Setup:
    Hard exit : EMA(10) crosses below EMA(30)   — trend flipped
    Soft exit : MACD(5,13,4) bearish crossover   — momentum stalled
                   AND RSI was in [60, 70] last bar
                   AND RSI is falling this bar   — overbought fading

Optionally gated by the HMM regime signal.

CLI key: ``composite_trend``
"""
from __future__ import annotations

import backtrader as bt

from .base import BaseRegimeStrategy


class CompositeTrendStrategy(BaseRegimeStrategy):
    """
    EMA-trend + ADX strength + RSI pullback + MACD crossover entry.

    Buy  : EMA fast > slow  AND  ADX > threshold  AND
           RSI bounces from [buy_zone_low, buy_zone_high]  AND
           MACD bullish crossover
    Sell : EMA fast crosses below slow (hard exit)  OR
           MACD bearish crossover while RSI fading from overbought zone
    """

    params = dict(
        ema_fast          = 10,
        ema_slow          = 30,
        rsi_period        = 7,
        rsi_buy_low       = 25,    # bottom of pullback buy zone
        rsi_buy_high      = 45,    # top of pullback buy zone
        rsi_sell_low      = 55,    # bottom of overbought sell zone
        rsi_sell_high     = 75,    # top of overbought sell zone
        signal_window     = 10,    # bars to look back for the RSI pullback condition;
                                   # RSI need not be in the zone on the exact entry bar
        adx_period        = 7,
        adx_threshold     = 20.0,
        macd_fast         = 5,
        macd_slow         = 13,
        macd_signal       = 4,
        stake             = 100,
        printlog          = True,
        use_hmm           = False,
        regime_mode       = 'strict',
        unfav_fraction    = 0.25,
    )

    def _init_indicators(self, d):
        # ── EMA trend ─────────────────────────────────────────────────────
        self.inds[d]['ema_fast']  = bt.ind.EMA(d.close, period=self.p.ema_fast)
        self.inds[d]['ema_slow']  = bt.ind.EMA(d.close, period=self.p.ema_slow)
        self.inds[d]['ema_cross'] = bt.ind.CrossOver(
            self.inds[d]['ema_fast'], self.inds[d]['ema_slow'])

        # ── ADX ───────────────────────────────────────────────────────────
        dm = bt.ind.DirectionalMovement(d, period=self.p.adx_period)
        self.inds[d]['adx'] = dm.adx

        # ── RSI ───────────────────────────────────────────────────────────
        self.inds[d]['rsi'] = bt.ind.RSI(d.close, period=self.p.rsi_period)

        # ── MACD ──────────────────────────────────────────────────────────
        macd_ind = bt.ind.MACD(
            d.close,
            period_me1    = self.p.macd_fast,
            period_me2    = self.p.macd_slow,
            period_signal = self.p.macd_signal,
        )
        self.inds[d]['macd_cross'] = bt.ind.CrossOver(
            macd_ind.macd, macd_ind.signal)

    def _signal(self, d) -> int:
        ema_fast   = self.inds[d]['ema_fast']
        ema_slow   = self.inds[d]['ema_slow']
        ema_cross  = self.inds[d]['ema_cross']
        adx        = self.inds[d]['adx']
        rsi        = self.inds[d]['rsi']
        macd_cross = self.inds[d]['macd_cross']
        pos        = self.getposition(d)

        rsi_now  = rsi[0]

        # RSI was in the buy pullback zone within the last signal_window bars.
        # Using a lookback window reflects how traders actually use this setup:
        # the RSI dips into 30-40, then MACD crosses as the entry trigger a
        # bar or two later.  The condition is: min RSI over the window was in
        # the buy zone AND current RSI is higher than the window minimum
        # (i.e. the bounce from the pullback has started).
        rsi_window_min = min(rsi[-i] for i in range(1, self.p.signal_window + 1))
        rsi_bounced_from_buy = (
            self.p.rsi_buy_low <= rsi_window_min <= self.p.rsi_buy_high
            and rsi_now > rsi_window_min
        )

        # RSI was in the overbought zone within the last signal_window bars
        # AND is now falling from that peak.
        rsi_window_max = max(rsi[-i] for i in range(1, self.p.signal_window + 1))
        rsi_fading_from_sell = (
            self.p.rsi_sell_low <= rsi_window_max <= self.p.rsi_sell_high
            and rsi_now < rsi_window_max
        )

        if not pos.size:
            # ── BUY: all four confirmations required ──────────────────────
            trend_up     = ema_fast[0] > ema_slow[0]
            adx_strong   = adx[0] > self.p.adx_threshold
            macd_bullish = macd_cross[0] > 0

            if trend_up and adx_strong and rsi_bounced_from_buy and macd_bullish:
                return 1
        else:
            # ── HARD EXIT: EMA trend flipped ──────────────────────────────
            if ema_cross[0] < 0:
                return -1

            # ── SOFT EXIT: MACD bearish while RSI fading from overbought ──
            if macd_cross[0] < 0 and rsi_fading_from_sell:
                return -1

        return 0

    def stop(self):
        self.log(
            f'CompositeTrend  EMA({self.p.ema_fast}/{self.p.ema_slow})  '
            f'RSI({self.p.rsi_period})  ADX({self.p.adx_period}>{self.p.adx_threshold})  '
            f'MACD({self.p.macd_fast}/{self.p.macd_slow}/{self.p.macd_signal})  '
            f'Ending value: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.datetime(0))
        super().stop()
