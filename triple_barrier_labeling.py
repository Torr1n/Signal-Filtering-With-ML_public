import pandas as pd
import numpy as np
import pandas_ta as ta


def get_vertical_barriers(events, bars, num_bars_to_vertical_barrier=20):
    """
    calculates and returns the vertical threshold for all events
    """
    # get EndDate of the bar that is num_bars_to_vertical_barrier bars later than the event's date
    vertical_barriers = pd.DataFrame(
        events.to_series().map(
            lambda event: (
                bars.iloc[bars.index.get_loc(event) + num_bars_to_vertical_barrier][
                    "EndDate"
                ]
                if bars.index.get_loc(event) + num_bars_to_vertical_barrier < len(bars)
                else None
            )
        )
    )
    return vertical_barriers.rename(
        columns={vertical_barriers.columns[0]: "vertical_barrier"}
    )


def get_volatility(bars, num_bars_lag, ewm_span=100):
    """
    calculates a volatility target which could be used calculate stop loss and take profit barriers (not currently used)
    """
    # get start and end of trade events
    tEvents = bars.EndDate[num_bars_lag:].to_frame("past_point")
    tEvents["past_point"] = bars.EndDate[:-num_bars_lag].values
    # calculate log returns of events
    df0 = np.log(
        bars.loc[tEvents.index]["Close"].values
        / bars.loc[tEvents.past_point]["Close"].values
    )
    # calculate volatility
    df0 = pd.DataFrame(df0)
    df0 = df0.ewm(span=ewm_span).std()
    return df0


def get_stop_loss_take_profit(events, bars, ema_length):
    """
    using keltner channels, calculates the stop loss and take profit thresholds for each event
    """
    # get kc channels from pandas_ta
    keltner_channels = ta.kc(
        np.log(bars.High), np.log(bars.Low), np.log(bars.Close), ema_length
    )
    # profit taking and stop loss based off kc channels (as defined by trader)
    channel_width = (
        keltner_channels["KCUe_" + str(ema_length) + "_2"]
        - keltner_channels["KCBe_" + str(ema_length) + "_2"]
    )
    keltner_channels["stop_loss"] = np.log(bars.Low) - 0.5 * channel_width
    keltner_channels["take_profit"] = np.log(bars.High) + 2 * channel_width
    # left join
    events = pd.merge(
        events,
        keltner_channels[["stop_loss", "take_profit"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    return events


def get_time_of_each_barrier_touch(close, events_df):
    """
    credit to Lopez de Prado, given events and the path of prices, determines which barrier was hit first
    """
    upper_barrier = events_df["take_profit"]
    lower_barrier = events_df["stop_loss"]

    events = events_df[["vertical_barrier", "take_profit", "stop_loss"]].copy()
    for event_start_time, vertical_barrier in (
        events_df["vertical_barrier"].fillna(close.index[-1]).items()
    ):
        # find the path of prices for this event excluding first price (price at time of event)
        path_prices = np.log(close.loc[event_start_time:vertical_barrier].iloc[1:])
        events.loc[event_start_time, "lower_barrier_earliest"] = path_prices[
            path_prices <= lower_barrier[event_start_time]
        ].index.min()  # earliest stop loss
        events.loc[event_start_time, "upper_barrier_earliest"] = path_prices[
            path_prices >= upper_barrier[event_start_time]
        ].index.min()  # earliest profit taking
        events.loc[event_start_time, "first_bar_start_date"] = path_prices.index[0]

    events["barrier_earliest"] = events[
        ["lower_barrier_earliest", "upper_barrier_earliest", "vertical_barrier"]
    ].min(axis=1)
    return events


def get_log_returns(close, events_df):
    """
    calcualtes the log returns between the event's start time and first barrier hit
    """
    earliest_touch_df = get_time_of_each_barrier_touch(close, events_df)
    earliest_touch_df["returns"] = np.log(
        close.loc[earliest_touch_df["barrier_earliest"].values].values
        / close.loc[earliest_touch_df.index].values
    )
    return earliest_touch_df


def get_classification_labels(close, events_df):
    """
    binary classifaction as described by Lopez de Prado (-1 or 1)
    """
    returns_df = get_log_returns(close, events_df)
    returns_df["label"] = np.sign(returns_df["returns"])
    return returns_df


def get_sample_weights(bars, labels):
    """
    for each event, calculates the sample weight of the event as a function of its uniqueness (overlap) and returns
    """
    # initialize df to track concurrency for each bar
    uniqueness_df = pd.DataFrame(index=bars.index)
    uniqueness_df.index.name = "StartDate"

    uniqueness_df["EndDateTime"] = uniqueness_df.index + pd.Timedelta(days=1)
    uniqueness_df["concurrent_labels"] = 0
    # for each event, if a bar falls within the event's range increment its concurrency by 1
    for index, (first_bar, last_bar) in labels[
        ["first_bar_start_date", "barrier_earliest"]
    ].iterrows():
        uniqueness_df.loc[first_bar:last_bar, "concurrent_labels"] += 1
    # calculate sample weight for each BAR in the dataset
    uniqueness_df["Log_Close"] = np.log(bars.Close)
    uniqueness_df["Log_Return"] = uniqueness_df["Log_Close"].diff()
    uniqueness_df["Sample_Weight"] = (
        uniqueness_df["Log_Return"] / uniqueness_df["concurrent_labels"]
    )
    # calculate sample weight for each EVENT as a sum of the weights of each bar in the event
    for index, (first_bar, last_bar) in labels[
        ["first_bar_start_date", "barrier_earliest"]
    ].iterrows():
        labels.loc[index, "Sample_Weight"] = np.sqrt(
            np.abs(np.sum(uniqueness_df.loc[first_bar:last_bar, "Sample_Weight"]))
        )
    return labels
