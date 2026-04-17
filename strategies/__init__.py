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
from .base import BaseRegimeStrategy
from .sma  import SmaCrossOver
from .dema import DemaCrossOver
from .rsi  import RsiStrategy
from .macd import MacdStrategy
from .hmm_mean_reversion import HmmMeanReversionStrategy

__all__ = [
    'BaseRegimeStrategy',
    'SmaCrossOver',
    'DemaCrossOver',
    'RsiStrategy',
    'MacdStrategy',
    'HmmMeanReversionStrategy',
    'REGISTRY',
]


def _sma_kwargs(args) -> dict:
    return dict(
        fast     = args.fast,
        slow     = args.slow,
        stake    = args.stake,
        printlog = args.printlog,
        use_hmm  = args.hmm,
    )


def _dema_kwargs(args) -> dict:
    return dict(
        fast     = args.fast,
        slow     = args.slow,
        stake    = args.stake,
        printlog = args.printlog,
        use_hmm  = args.hmm,
    )


def _rsi_kwargs(args) -> dict:
    return dict(
        rsi_period  = args.rsi_period,
        oversold    = args.rsi_oversold,
        overbought  = args.rsi_overbought,
        stake       = args.stake,
        printlog    = args.printlog,
        use_hmm     = args.hmm,
    )


def _macd_kwargs(args) -> dict:
    return dict(
        macd_fast   = args.macd_fast,
        macd_slow   = args.macd_slow,
        macd_signal = args.macd_signal,
        stake       = args.stake,
        printlog    = args.printlog,
        use_hmm     = args.hmm,
    )


def _hmm_mr_kwargs(args) -> dict:
    return dict(
        z_threshold = getattr(args, 'hmm_mr_z_threshold', 0.0),
        stake       = args.stake,
        printlog    = args.printlog,
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
        'cls':          RsiStrategy,
        'label':        'RSI Mean-Reversion',
        'build_kwargs': _rsi_kwargs,
    },
    'macd': {
        'cls':          MacdStrategy,
        'label':        'MACD Crossover',
        'build_kwargs': _macd_kwargs,
    },
    'hmm_mr': {
        'cls':          HmmMeanReversionStrategy,
        'label':        'HMM Mean-Reversion',
        'build_kwargs': _hmm_mr_kwargs,
    },
}
