#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
strategies package
==================
Exports all concrete strategy classes and a REGISTRY dict that maps
the CLI ``--strategy`` key to the class and its parameter defaults.

Usage in ma-quantstats.py::

    from strategies import REGISTRY
    entry = REGISTRY['sma']
    cls    = entry['cls']
    kwargs = entry['build_kwargs'](args)
    cerebro.addstrategy(cls, **kwargs)
"""
from .base               import BaseRegimeStrategy
from .sma                import SmaCrossOver
from .dema               import DemaCrossOver
from .rsi                import RsiStrategy
from .macd               import MacdStrategy
from .hmm_mean_reversion import HmmMeanReversionStrategy
from .adx_dm             import AdxDmStrategy
from .bollinger_bands    import BollingerBandsStrategy
from .channel_breakout   import ChannelBreakoutStrategy
from .composite_trend    import CompositeTrendStrategy
from .donchian           import DonchianStrategy
from .false_breakout     import FalseBreakoutStrategy
from .ichimoku           import IchimokuStrategy
from .kama               import KamaStrategy
from .parabolic_sar      import ParabolicSarStrategy
from .tsmom              import TsmomStrategy
from .turtle             import TurtleStrategy
from .volatility_adjusted import VolatilityAdjustedStrategy

__all__ = [
    'BaseRegimeStrategy',
    'SmaCrossOver',
    'DemaCrossOver',
    'RsiStrategy',
    'MacdStrategy',
    'HmmMeanReversionStrategy',
    'AdxDmStrategy',
    'BollingerBandsStrategy',
    'ChannelBreakoutStrategy',
    'CompositeTrendStrategy',
    'DonchianStrategy',
    'FalseBreakoutStrategy',
    'IchimokuStrategy',
    'KamaStrategy',
    'ParabolicSarStrategy',
    'TsmomStrategy',
    'TurtleStrategy',
    'VolatilityAdjustedStrategy',
    'REGISTRY',
]


def _sma_kwargs(args) -> dict:
    return dict(
        fast             = args.fast,
        slow             = args.slow,
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _dema_kwargs(args) -> dict:
    return dict(
        fast             = args.fast,
        slow             = args.slow,
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _rsi_kwargs(args) -> dict:
    return dict(
        rsi_period       = args.rsi_period,
        oversold         = args.rsi_oversold,
        overbought       = args.rsi_overbought,
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
        # invert_regime defaults to True inside RsiStrategy.params; allow
        # explicit override from args (e.g. ablation experiments).
        invert_regime    = getattr(args, 'rsi_invert_regime', True),
    )


def _macd_kwargs(args) -> dict:
    return dict(
        macd_fast        = args.macd_fast,
        macd_slow        = args.macd_slow,
        macd_signal      = args.macd_signal,
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _hmm_mr_kwargs(args) -> dict:
    return dict(
        z_threshold      = getattr(args, 'hmm_mr_z_threshold', 0.0),
        stake            = args.stake,
        printlog         = args.printlog,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _bollinger_kwargs(args) -> dict:
    return dict(
        bb_period        = getattr(args, 'bb_period', 20),
        bb_devfactor     = getattr(args, 'bb_devfactor', 2.0),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _kama_kwargs(args) -> dict:
    return dict(
        kama_period      = getattr(args, 'kama_period', 10),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _false_breakout_kwargs(args) -> dict:
    return dict(
        fb_period        = getattr(args, 'fb_period', 20),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _composite_trend_kwargs(args) -> dict:
    return dict(
        ema_fast         = getattr(args, 'ct_ema_fast', 10),
        ema_slow         = getattr(args, 'ct_ema_slow', 30),
        rsi_period       = getattr(args, 'ct_rsi_period',   7),
        rsi_buy_low      = getattr(args, 'ct_rsi_buy_low',  25),
        rsi_buy_high     = getattr(args, 'ct_rsi_buy_high', 45),
        rsi_sell_low     = getattr(args, 'ct_rsi_sell_low', 55),
        rsi_sell_high    = getattr(args, 'ct_rsi_sell_high',75),
        signal_window    = getattr(args, 'ct_signal_window',10),
        adx_period       = getattr(args, 'ct_adx_period',   7),
        adx_threshold    = getattr(args, 'ct_adx_threshold',20.0),
        macd_fast        = getattr(args, 'ct_macd_fast', 5),
        macd_slow        = getattr(args, 'ct_macd_slow', 13),
        macd_signal      = getattr(args, 'ct_macd_signal', 4),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _adx_dm_kwargs(args) -> dict:
    return dict(
        adx_period       = getattr(args, 'adx_period', 14),
        adx_threshold    = getattr(args, 'adx_threshold', 25),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _channel_breakout_kwargs(args) -> dict:
    return dict(
        channel_period   = getattr(args, 'channel_period', 20),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _donchian_kwargs(args) -> dict:
    return dict(
        donchian_entry   = getattr(args, 'donchian_entry', 20),
        donchian_exit    = getattr(args, 'donchian_exit', 10),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _ichimoku_kwargs(args) -> dict:
    return dict(
        tenkan           = getattr(args, 'ichimoku_tenkan', 9),
        kijun            = getattr(args, 'ichimoku_kijun', 26),
        senkou           = getattr(args, 'ichimoku_senkou', 52),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _parabolic_sar_kwargs(args) -> dict:
    return dict(
        psar_af          = getattr(args, 'psar_af', 0.02),
        psar_max_af      = getattr(args, 'psar_max_af', 0.20),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _tsmom_kwargs(args) -> dict:
    return dict(
        tsmom_lookback   = getattr(args, 'tsmom_lookback', 252),
        tsmom_skip       = getattr(args, 'tsmom_skip', 21),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _tsmom_fast_kwargs(args) -> dict:
    """Short-lookback TSMOM variant (3-month / 1-week skip) designed to trade
    in volatile HMM regimes where rapid momentum reversals are common."""
    return dict(
        tsmom_lookback   = 63,   # ~3 months instead of 12
        tsmom_skip       = 5,    # ~1 week instead of 1 month
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _turtle_kwargs(args) -> dict:
    return dict(
        turtle_entry     = getattr(args, 'turtle_entry', 20),
        turtle_exit      = getattr(args, 'turtle_exit', 10),
        turtle_atr       = getattr(args, 'turtle_atr', 20),
        turtle_atr_mult  = getattr(args, 'turtle_atr_mult', 2.0),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


def _vol_adj_kwargs(args) -> dict:
    return dict(
        vol_period       = getattr(args, 'vol_period', 20),
        vol_atr_period   = getattr(args, 'vol_atr_period', 14),
        vol_atr_mult     = getattr(args, 'vol_atr_mult', 1.5),
        stake            = args.stake,
        printlog         = args.printlog,
        use_hmm          = args.hmm,
        regime_mode      = getattr(args, 'regime_mode', 'strict'),
        unfav_fraction   = getattr(args, 'unfav_fraction', 0.25) or 0.25,
        stop_loss_perc   = getattr(args, 'stop_loss_perc', 0.0) or 0.0,
        take_profit_perc = getattr(args, 'take_profit_perc', 0.0) or 0.0,
    )


# ---------------------------------------------------------------------------
# REGISTRY
#   key          : CLI value for --strategy
#   cls          : strategy class
#   label        : human-readable name used in reports / HTML titles
#   build_kwargs : callable(args) → dict of strategy params
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    'sma': {
        'cls':          SmaCrossOver,
        'label':        'SMA Crossover',
        'build_kwargs': _sma_kwargs,
    },
    'dema': {
        'cls':          DemaCrossOver,
        'label':        'DEMA Crossover',
        'build_kwargs': _dema_kwargs,
    },
    'rsi': {
        'cls':           RsiStrategy,
        'label':         'RSI Mean-Reversion',
        'build_kwargs':  _rsi_kwargs,
        'mean_reversion': True,
    },
    'macd': {
        'cls':          MacdStrategy,
        'label':        'MACD Crossover',
        'build_kwargs': _macd_kwargs,
    },
    'adx_dm': {
        'cls':          AdxDmStrategy,
        'label':        'ADX + Directional Movement',
        'build_kwargs': _adx_dm_kwargs,
    },
    'channel_breakout': {
        'cls':          ChannelBreakoutStrategy,
        'label':        'Channel Breakout',
        'build_kwargs': _channel_breakout_kwargs,
    },
    'donchian': {
        'cls':          DonchianStrategy,
        'label':        'Donchian Channel',
        'build_kwargs': _donchian_kwargs,
    },
    'ichimoku': {
        'cls':          IchimokuStrategy,
        'label':        'Ichimoku Cloud',
        'build_kwargs': _ichimoku_kwargs,
    },
    'parabolic_sar': {
        'cls':          ParabolicSarStrategy,
        'label':        'Parabolic SAR',
        'build_kwargs': _parabolic_sar_kwargs,
    },
    'tsmom': {
        'cls':          TsmomStrategy,
        'label':        'Time-Series Momentum',
        'build_kwargs': _tsmom_kwargs,
    },
    'tsmom_fast': {
        'cls':           TsmomStrategy,
        'label':         'Time-Series Momentum (Fast 3-month)',
        'build_kwargs':  _tsmom_fast_kwargs,
        'mean_reversion': True,
    },
    'turtle': {
        'cls':          TurtleStrategy,
        'label':        'Turtle System 1',
        'build_kwargs': _turtle_kwargs,
    },
    'vol_adj': {
        'cls':          VolatilityAdjustedStrategy,
        'label':        'Volatility-Adjusted (Keltner)',
        'build_kwargs': _vol_adj_kwargs,
    },
    'bollinger': {
        'cls':           BollingerBandsStrategy,
        'label':         'Bollinger Bands Mean-Reversion',
        'build_kwargs':  _bollinger_kwargs,
        'mean_reversion': True,
    },
    'kama': {
        'cls':          KamaStrategy,
        'label':        'KAMA Crossover',
        'build_kwargs': _kama_kwargs,
    },
    'false_breakout': {
        'cls':           FalseBreakoutStrategy,
        'label':         'False Breakout (Fakeout)',
        'build_kwargs':  _false_breakout_kwargs,
        'mean_reversion': True,
    },
    'composite_trend': {
        'cls':          CompositeTrendStrategy,
        'label':        'Composite Trend-Pullback (EMA+ADX+RSI+MACD)',
        'build_kwargs': _composite_trend_kwargs,
    },
}
