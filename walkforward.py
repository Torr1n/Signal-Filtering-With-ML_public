import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.metrics import roc_auc_score, log_loss


def walkforward_model(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    train_size: int,
    step_size: int,
):
    """
    executes a walkforward optimization. Trains, then evaluates a random forest on an expanding window of trades
    """
    close = np.log(prices).to_numpy()
    signal = np.zeros(len(close))
    prob_signal = np.zeros(len(close))

    next_train = train_size
    trade_i = 0

    in_trade = False
    take_profit = None
    stop_loss = None
    vertical_barrier = None

    model = None
    # iterate through all bars
    for i in range(len(close)):
        # if we have walked through enough bars to be able to train, then train a model on the past train_size bars
        if i == next_train:
            start_date = prices.index[i - train_size]
            end_date = prices.index[i]
            # get the trade events that are within the training range
            train_indices = trades[
                (trades["first_bar_start_date"] > start_date)
                & (trades["barrier_earliest"] < end_date)
            ].index
            print(train_indices)
            # get features and labels
            x_train = features.loc[train_indices]
            y_train = labels.loc[train_indices]
            sample_weights = trades.loc[train_indices]["Sample_Weight"]
            print("training", i, "N cases", len(train_indices))
            model = RandomForestClassifier(
                n_estimators=1000,
                max_depth=3,
                random_state=69420,
            )
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            # increment the rolling window by the step size (the length of the testing range)
            next_train += step_size
        # handle exiting trades if any barriers are hit
        if in_trade:
            if (
                close[i] >= take_profit
                or close[i] <= stop_loss
                or prices.index[i] >= vertical_barrier
            ):
                signal[i] = 0
                prob_signal[i] = 0
                in_trade = False
            else:
                signal[i] = signal[i - 1]
                prob_signal[i] = prob_signal[i - 1]
        # print(prices.index[i], trades.index[trade_i])
        # if in the daily bars we have stepped up to the first trade event, and if the model has trained, predict the outcome of the trade
        if trade_i < len(trades) and prices.index[i] == trades.index[trade_i]:
            if model is not None:
                print("predicting trade ", prices.index[i])
                prob = model.predict_proba(
                    features.iloc[trade_i].to_numpy().reshape(1, -1)
                )[0][1]
                prob_signal[i] = prob
                print(
                    "based off these features: ",
                    features.iloc[trade_i].to_numpy().reshape(1, -1),
                )
                print("model predicted probability of a 1: ", prob)
                trades.loc[trades.index[trade_i], "model_prob"] = prob

                if prob > 0.6:  # greater than 60% (arbitrary), take trade
                    signal[i] = 1

                in_trade = True
                trade = trades.iloc[trade_i]
                take_profit = trade["take_profit"]
                stop_loss = trade["stop_loss"]
                vertical_barrier = trade["vertical_barrier"]

            trade_i += 1

    return signal, prob_signal


def get_classification_report(labels, probs):
    probs.dropna(how="any")
    labels.dropna(how="any")
    probs = pd.DataFrame({"prob": probs})
    labels = pd.DataFrame({"label": labels})
    df = labels.merge(probs, how="left", left_index=True, right_index=True)
    df = df.dropna(axis=0)
    probs = df.prob
    labels = df.label

    # Get predictions from probs
    predictions = (probs >= 0.6).astype(int)

    # 2. Generate the confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 3. Calculate Accuracy
    accuracy = accuracy_score(labels, predictions)
    print("\nAccuracy:", accuracy)

    # 4. Calculate Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    print("\nBalanced Accuracy:", balanced_accuracy)

    # 5. Generate a classification report for other metrics
    class_report = classification_report(
        labels, predictions, target_names=["SELL", "BUY"]
    )
    print("\nClassification Report:")
    print(class_report)
    # 5. Calculate AUC
    auc = roc_auc_score(y_true=labels, y_score=probs)
    print(f"AUC: {auc}")

    # 6. Calculate log loss
    loss = log_loss(labels, probs)
    print(f"Log loss: {loss}")
