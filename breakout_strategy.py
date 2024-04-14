import yfinance as yf
import pandas as pd
import numpy as np
import math
import pandas_ta as ta


def get_breakout_events(bars, ema_length, candle_percent):
    """
    samples events (strategy triggers) from the given bars
    """
    tEvents = []
    ema = bars["Close"].ewm(span=ema_length, adjust=False).mean()

    for i in range(0, len(bars) - 1):
        high = bars["High"].iloc[i]
        low = bars["Low"].iloc[i]
        close = bars["Close"].iloc[i]
        open = bars["Open"].iloc[i]
        ema_value = ema.iloc[i]

        if shouldBuy(high, low, close, open, ema_value, candle_percent):
            tEvents.append(bars["EndDate"].iloc[i])
        # elif shouldSell(high, low, close, open, ema_value, candle_percent):
        #     tEvents.append(bars["EndDate"].iloc[i])
    return pd.DatetimeIndex(tEvents)


def getCusumEvents(gRaw, h):
    """
    implementation of Lopez de Prado's 'Cusum Filter' for sampling events (not used)
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos = max(0, sPos + diff.loc[i])
        sNeg = min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def breakout_strategy(df, ema_length, candle_percent):
    """
    applys the base strategy on the entire set of given bars
    """
    ema = df["Close"].ewm(span=ema_length, adjust=False).mean()
    signals = np.zeros(len(df))

    for i in range(0, len(df) - 1):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]
        open = df["Open"].iloc[i]
        ema_value = ema[i]

        if shouldBuy(high, low, close, open, ema_value, candle_percent):
            signals[i] = 1
        elif shouldSell(high, low, close, open, ema_value, candle_percent):
            signals[i] = -1
        elif i == (len(df) - 2):
            signals[i] = -1
        else:
            signals[i] = -1

    return ema, signals


def apply_features(dollar_bars, ema_length, holding_period):
    """
    from pandas_ta, calculates all indicators used for model predictons
    """
    dollar_bars["EMA"] = ta.ema(close=dollar_bars.Close, length=ema_length)
    dollar_bars["rsi"] = ta.rsi(close=dollar_bars.EMA, length=ema_length)
    dollar_bars["adx"] = ta.adx(
        dollar_bars.High, dollar_bars.Low, dollar_bars.Close, ema_length
    )["ADX_" + str(ema_length)]
    dollar_bars[["bband_lower", "bband_middle", "bband_upper"]] = ta.bbands(
        close=dollar_bars.EMA,
        length=7,
        std=3,
    )[["BBL_7_3.0", "BBM_7_3.0", "BBU_7_3.0"]]
    dollar_bars["cmf"] = ta.cmf(
        dollar_bars.High,
        dollar_bars.Low,
        dollar_bars.Close,
        dollar_bars.Volume,
        dollar_bars.Open,
        holding_period,
    )
    dollar_bars["obv"] = ta.obv(dollar_bars.EMA, dollar_bars.Volume)
    dollar_bars[["stoch_k", "stoch_d"]] = ta.stoch(
        dollar_bars.High, dollar_bars.Low, dollar_bars.Close, holding_period, 3, 3
    )[
        [
            "STOCHk_" + str(holding_period) + "_3_3",
            "STOCHd_" + str(holding_period) + "_3_3",
        ]
    ]
    return dollar_bars


def shouldBuy(high, low, close, open, ema_value, candle_percent):
    candlePercent = (high - ema_value) / (high - low)
    return (close > open) and (close > ema_value) and (candlePercent > candle_percent)


def shouldSell(high, low, close, open, ema_value, candle_percent):
    candlePercent = (ema_value - low) / (high - low)
    return (close < open) and (close < ema_value) and (candlePercent > candle_percent)
