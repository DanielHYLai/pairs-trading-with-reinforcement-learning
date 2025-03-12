# Import the necessary packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create the initial state space
def state_space_creator(data: pd.DataFrame, ticker_list: list, trade_window: int = 1):
    """
    Create the initial state space.

    Parameters:
    ----------
    data : pd.DataFrame
        Input the training data or testing data

    ticker_list : list
        Input the list of pairs

    trade_window : int
        Enter the number of equal parts into which want to divide the period

    Returns:
    -------
    state_space : dict
        Return the inital state space

    coint_coef : dict
        Return the cointegration coefficients of each input pairs
    """

    state_space = {}
    coint_coef  = {}
    idx = 1

    # Obtain all unique years and months
    unique_years  = data["Year"].unique().tolist()
    unique_months = data["Month"].unique()

    # Check if the trading window is a factor of 12
    if 12 % trade_window == 0:
        trade_interval = np.split(unique_months, len(unique_months) // trade_window)

    else:
        raise ValueError("The number passed in trade_window is not a factor of 12.")

    for year in unique_years:
        for month in trade_interval:
            state_key = f"state_{idx}"
            state_space[state_key] = {}
            coint_coef[state_key] = []

            print(f"Processing state {idx} ... ", end="")

            for ticker_1, ticker_2 in ticker_list:
                ticker_key = f"{ticker_1}-{ticker_2}"
                date_mask = (data["Year"] == year) & (data["Month"].isin(month))
                mask_table = data[date_mask]

                # Start to do cointegration test for selected pair
                close_1 = mask_table[mask_table["Ticker"] == ticker_1][
                    "Close"
                ].to_numpy()
                close_2 = mask_table[mask_table["Ticker"] == ticker_2][
                    "Close"
                ].to_numpy()
                month_inner = mask_table[mask_table["Ticker"] == ticker_1][
                    "Month"
                ].to_numpy()

                model_LR = LinearRegression(fit_intercept=False, n_jobs=-1)
                model_LR.fit(X=close_1.reshape(-1, 1), y=close_2.reshape(-1, 1))
                coef = model_LR.coef_[0][0]

                # Only record data of pairs with positive coefficients, otherwise record zero
                if coef > 0:
                    spread = close_2 - coef * close_1
                    state_info = pd.DataFrame(
                        {
                            "spread": spread,
                            "close_1": close_1,
                            "close_2": close_2,
                            "month": month_inner,
                        },
                        index=[ticker_key] * len(spread),
                    )

                    state_space[state_key][ticker_key] = state_info
                    coint_coef[state_key].append(np.round(coef, 2))

                else:
                    state_space[state_key][ticker_key] = pd.DataFrame(
                        np.zeros(shape=(23, 4)),
                        index=[ticker_key] * 23,
                        columns=["spread", "close_1", "close_2", "month"],
                    )
                    coint_coef[state_key].append(0)

            idx += 1
            print("Done")

    return (state_space, coint_coef)

# Calculate the trading threshold for traidng strategy
def trading_threshold_creator(
    obj: dict, ticker_list: list, window_size: int = 5, ub_lb_size: int = 1
):
    """
    Calculate the trading threshold for traidng strategy.

    Parameters:
    ----------
    obj : dict
        Input the state space of DQN

    ticker_list : list
        Input the list of tickers

    window_size : int, default 5
        Input the window size of Bollinger Bands

    ub_lb_size : int, default 1
        Input the size of upper and lower bands

    Returns:
    -------
    result : dict
        Return the trading threshold of DQN.
    """

    BB_dict = {}
    result = obj.copy()

    # Calculate the Bollinger Bands for each pair
    print("Processing: calculate Bollinger bands ... ", end="")
    for ticker_1, ticker_2 in ticker_list:
        ticker_idx = f"{ticker_1}-{ticker_2}"
        combined_data = pd.concat(
            [result[state][ticker_idx] for state in result.keys()], axis=0
        )
        combined_data["ma_line"] = (
            combined_data["spread"].rolling(window=window_size).mean()
        )
        combined_std = combined_data["spread"].rolling(window=window_size).std()
        combined_data["upper_bound"] = (
            combined_data["ma_line"] + ub_lb_size * combined_std
        )
        combined_data["lower_bound"] = (
            combined_data["ma_line"] - ub_lb_size * combined_std
        )
        BB_dict[ticker_idx] = combined_data.reset_index(drop=True)
    print("Done")

    # Paste the calculated results to the original dictionary
    print("Processing: paste the calculated results ... ", end="")
    for pair in list(BB_dict.keys()):
        for state in list(result.keys()):
            result[state][pair]["ma_line"] = (
                BB_dict[pair]["ma_line"].iloc[: len(result[state][pair])].tolist()
            )
            result[state][pair]["upper_bound"] = (
                BB_dict[pair]["upper_bound"].iloc[: len(result[state][pair])].tolist()
            )
            result[state][pair]["lower_bound"] = (
                BB_dict[pair]["lower_bound"].iloc[: len(result[state][pair])].tolist()
            )
            BB_dict[pair] = (
                BB_dict[pair].iloc[len(result[state][pair]) :].reset_index(drop=True)
            )
    print("Done")

    # Remove missing values ​​generated by calculating BB bands
    for ticker_1, ticker_2 in ticker_list:
        ticker_idx = f"{ticker_1}-{ticker_2}"
        result["state_1"][ticker_idx] = result["state_1"][ticker_idx].dropna()

    # Check the trading days
    print("Processing: check the trading days ... ", end="")
    for state in list(result.keys()):
        for pair in list(result[state].keys()):
            target = result[state][pair]
            unique_month = target["month"].unique()
            target_split = []

            for month in unique_month:
                target_inner = target[target["month"] == month]
                if target_inner.shape[0] != 23:
                    insert_data = pd.DataFrame(
                        np.zeros(
                            shape=(23 - target_inner.shape[0], target_inner.shape[1])
                        ),
                        columns=target_inner.columns,
                    )
                    target_split.append(
                        pd.concat([insert_data, target_inner], ignore_index=True)
                    )

                else:
                    target_split.append(target_inner)

            result[state][pair] = pd.concat(target_split, ignore_index=True)
    print("Done")

    return result

# Merge and flatten the state of data
def state_merge(obj: dict, ticker_list: list):
    """
    Merge and flatten the state of data.

    Parameters
    ----------
    obj : dict
        Input the state space of DQN

    ticker_list : list
        Input the list of tickers

    Returns
    -------
    result : dict
        Return the merged and flattened state space of DQN.
    """

    result = {}
    for state in list(obj.keys()):
        table_merge = []
        for ticker_1, ticker_2 in ticker_list:
            ticker_idx = f"{ticker_1}-{ticker_2}"
            # Only take the spread feature
            temp = obj[state][ticker_idx].iloc[:, 0].T
            table_merge.append(temp)
        result[state] = pd.concat(table_merge, axis=0).to_numpy().flatten()

    return result
