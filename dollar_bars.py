import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
import mplfinance as mpf
from breakout_strategy import *
from triple_barrier_labeling import *
from walkforward import *


def plot_trade(dollar_bars: pd.DataFrame, trades: pd.DataFrame, trade_date: int):
    trade = trades.loc[trade_date]
    entry_date = trade.name
    vertical_barrier = trade["vertical_barrier"]
    earliest_barrier = trade["barrier_earliest"]
    candles = np.log(
        dollar_bars.iloc[
            dollar_bars.index.get_loc(entry_date)
            - 5 : dollar_bars.index.get_loc(vertical_barrier)
            + 5
        ]
    )
    tp = [
        (candles.loc[entry_date].name, trade["take_profit"]),
        (candles.loc[vertical_barrier].name, trade["take_profit"]),
    ]
    sl = [
        (candles.loc[entry_date].name, trade["stop_loss"]),
        (candles.loc[vertical_barrier].name, trade["stop_loss"]),
    ]

    mco = [None] * len(candles)
    mco[candles.index.get_loc(earliest_barrier)] = "orange"
    mco[candles.index.get_loc(entry_date)] = "orange"
    mco[candles.index.get_loc(vertical_barrier)] = "blue"

    mpf.plot(
        candles,
        alines=dict(alines=[tp, sl], colors=["g", "b"]),
        type="candle",
        style="charles",
        marketcolor_overrides=mco,
    )
    plt.show()


def get_performance_report(no_filter_returns, all_returns, dollar_bars):
    print("All Trades PF", prof_factor(no_filter_returns))
    print("All Trades Avg", all_returns.mean())
    print("All Trades Win Rate", len(all_returns[all_returns >= 0]) / len(all_returns))
    print(
        "All Trades Time In Market",
        len(dollar_bars[dollar_bars["dumb_sig"] > 0]) / len(dollar_bars),
    )

    print("Meta-Label Trades PF", prof_factor(filter_returns))
    print("Meta-Label Trades Avg", meta_label_returns.mean())
    print(
        "Meta-Label Trades Win Rate",
        len(meta_label_returns[meta_label_returns >= 0]) / len(meta_label_returns),
    )
    print(
        "Meta-Label Time In Market",
        len(dollar_bars[dollar_bars["sig"] > 0]) / len(dollar_bars),
    )


def prof_factor(rets):
    """
    calcualtes profit factor (ratio of postive to negative returns)
    """
    return rets[rets > 0].sum() / rets[rets < 0].abs().sum()


def calculate_dollar_threshold(daily_bars):
    """
    calculates the average weekly dollar volume for the entire daily_bars set of daily bars
    """
    total_volume = daily_bars["Dollar_Volume"].sum()
    num_weeks = len(daily_bars) / 5
    # Calculate average weekly dollar volume
    average_weekly_volume = (total_volume / num_weeks) / 2
    return average_weekly_volume


def extract_weekly_dollar_bars(daily_bars):
    """
    converts daily bars to dollar bars based on the provided threshold.
    """
    daily_bars["Dollar_Volume"] = daily_bars["Close"] * daily_bars["Volume"]
    # Calculate average weekly dollar volume to use as threshold
    threshold = calculate_dollar_threshold(daily_bars)

    daily_bars["Cumulative_Dollar_Volume"] = daily_bars["Dollar_Volume"].cumsum()
    daily_bars["bar_id"] = daily_bars["Cumulative_Dollar_Volume"] // threshold
    daily_bars = daily_bars.reset_index()

    # Group by bar_id and define the new dollar bar for that aggregate
    dollar_bars = daily_bars.groupby("bar_id").agg(
        Open=("Open", "first"),
        Close=("Close", "last"),
        Low=("Low", "min"),
        High=("High", "max"),
        Volume=("Volume", "sum"),
        StartDate=("Date", "first"),
        EndDate=("Date", "last"),
        Average=("Dollar_Volume", "sum"),
    )
    dollar_bars["Average"] = dollar_bars["Average"] / dollar_bars["Volume"]
    dollar_bars.index = dollar_bars.EndDate
    return pd.DataFrame(dollar_bars)


pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
if __name__ == "__main__":
    # Get daily bars
    daily_bars = yf.download("ABX.TO", start="2014-01-01")
    # Define strategy params
    ema_length = 12
    holding_period = 14
    # Get dollar bars
    dollar_bars = extract_weekly_dollar_bars(daily_bars)
    # add technical analysis features to dollar bars
    dollar_bars = apply_features(
        dollar_bars=dollar_bars, ema_length=ema_length, holding_period=holding_period
    )
    print(dollar_bars)

    # get the time of events sampled using our breakout strategy
    events = get_breakout_events(dollar_bars, ema_length, 0.5)
    """Apply the Triple Barrier Method"""
    # get vertical bar (hit holding period max)
    events = get_vertical_barriers(events, dollar_bars, holding_period)
    # get stop loss and take profits based of KC channels
    events = get_stop_loss_take_profit(events, dollar_bars, ema_length)
    events = events.dropna(how="any")
    # get the labels for each event/triple barrier instance (-1 or 1 depending on returns)
    labels = get_classification_labels(daily_bars.Close, events)
    # get sample weights for each event/triple barrier instance as a fn of overlap and returns
    labels = get_sample_weights(daily_bars, labels)
    # seperate out the features and labels from our events dataset (for simplicity sake)
    features_cols = [
        "EMA",
        "rsi",
        "adx",
        "bband_lower",
        "bband_middle",
        "bband_upper",
        "cmf",
        "obv",
        "stoch_k",
        "stoch_d",
    ]
    trades = pd.merge(
        labels,
        dollar_bars[features_cols],
        how="left",
        left_index=True,
        right_index=True,
    ).dropna(subset=features_cols)
    print(trades)
    features = trades[features_cols]
    labels = trades["label"].replace(-1, 0)
    dollar_bars = dollar_bars[dollar_bars.index >= "2014-03-01"]
    # perform a walkforward optimization for a rolling window of training (train_size) then evaluating (step_size) the model
    signal, prob = walkforward_model(
        dollar_bars["Close"], trades, features, labels, 500, 75
    )

    dollar_bars = dollar_bars[
        ["Open", "High", "Low", "Close", "Volume", "StartDate", "Average"]
    ]
    dollar_bars["sig"] = signal
    # dumb_sig takes every trade, no ML filter
    dollar_bars["dumb_sig"] = prob
    dollar_bars.loc[dollar_bars["dumb_sig"] > 0, "dumb_sig"] = 1
    dollar_bars = dollar_bars[dollar_bars.index >= "2020-06-01"]
    dollar_bars["r"] = np.log(dollar_bars["Close"]).diff().shift(-1)

    all_returns = trades["returns"]
    meta_label_returns = trades[trades["model_prob"] > 0.6]["returns"]

    filter_returns = dollar_bars["r"] * dollar_bars["sig"]
    no_filter_returns = dollar_bars["r"] * dollar_bars["dumb_sig"]

    filter_returns.cumsum().plot(label="Meta-Labeled")
    no_filter_returns.cumsum().plot(label="All Trades")
    (dollar_bars["r"]).cumsum().plot(label="Buy Hold")
    plt.legend()
    plt.show()

    plot_trade(
        dollar_bars[["Open", "High", "Low", "Close"]], trades, trade_date="2022-09-13"
    )
    get_classification_report(labels, trades["model_prob"])
