"""
Technical indicator functions for stock trading feature engineering.

Provides MACD, RSI, CCI, and ADX indicators that can be included in the
MDP state representation to give the RL agent additional market signals.
"""

import numpy as np
import pandas as pd


def calculate_state_dim(args, n_stock) -> int:
    # balance + (n_stock * stock_owned) + (n_stock * stock_price)
    state_dim = 1 + n_stock * 2

    if args.use_macd:
        state_dim += n_stock
    if args.use_rsi:
        state_dim += n_stock
    if args.use_cci:
        state_dim += n_stock
    if args.use_adx:
        state_dim += n_stock

    return state_dim


def macd(data: pd.DataFrame):
    ema_12 = data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def rsi(data: pd.DataFrame, window=14):
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def cci(data: pd.DataFrame, window=20):
    tp = (data["high"] + data["low"] + data["close"]) / 3
    sma = tp.rolling(window=window).mean()
    mad = (tp - sma).abs().rolling(window=window).mean()
    return (tp - sma) / (0.015 * mad)


def adx(data: pd.DataFrame, window=14):
    high = data["high"]
    low = data["low"]
    close = data["close"]

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # atr = tr.rolling(window=window).mean()

    up = high.diff()
    down = -low.diff()

    pos = (up > down) & (up > 0)
    neg = (down > up) & (down > 0)

    dip = pd.Series(np.zeros(len(data)))
    din = pd.Series(np.zeros(len(data)))

    dip[pos] = up[pos]
    din[neg] = down[neg]

    dip = dip.ewm(span=window).mean()
    din = din.ewm(span=window).mean()

    dx = (dip - din).abs() / (dip + din)
    adx = dx.ewm(span=window).mean()

    return adx
