import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils_file.tools import load_cache_file, write_cache_file

warnings.filterwarnings("ignore")
test = pd.read_csv("data_file/test_PT.csv")

ticker = test["Ticker"].unique()

# 將 S&P500 成分股的股價資料新增 Bollinger bands
BB_value_result = []
for element in ticker:
    temp = test[test["Ticker"] == element]
    temp["ma_line"] = temp["Close"].rolling(window=5).mean()
    std = temp["Close"].rolling(window=5).std()
    temp["upper_bound"] = temp["ma_line"] + std
    temp["lower_bound"] = temp["ma_line"] - std
    temp = temp.dropna()
    BB_value_result.append(temp)

test_BB = pd.concat(BB_value_result, ignore_index=True)

# 將 S&P500 成分股的股價資料按照每個月切割出來，並存於字典中
split_test = {}

unique_years = test["Year"].unique()
unique_months = test["Month"].unique()

idx = 1
for year in unique_years:
    for month in unique_months:

        split_test[f"state_{idx}"] = {}

        date_mask = (test_BB["Year"] == year) & (test_BB["Month"] == month)
        mask_table = test_BB[date_mask]

        for element in ticker:
            split_test[f"state_{idx}"][element] = mask_table[
                mask_table["Ticker"] == element
            ]

        idx += 1

# 每一支成分股都執行一次自定義交易策略
trading_result = {}
for state in list(split_test.keys()):
    all_ticker_reward = []
    for element in ticker:
        temp = split_test[state][element]

        ## 判斷是否建倉
        ub_open_flag = False
        lb_open_flag = False
        status_list = []

        for i, row in temp.iterrows():
            if row["Close"] > row["upper_bound"] and ub_open_flag is False:
                status_list.append("UB_open")
                ub_open_flag = True

            elif row["Close"] < row["ma_line"] and ub_open_flag is True:
                status_list.append("UB_close")
                ub_open_flag = False

            elif row["Close"] < row["lower_bound"] and lb_open_flag is False:
                status_list.append("LB_open")
                lb_open_flag = True

            elif row["Close"] > row["ma_line"] and lb_open_flag is True:
                status_list.append("LB_close")
                lb_open_flag = False

            else:
                status_list.append(0)

        status_list[-1] = "close"
        temp["status"] = status_list

        ## 實際執行交易策略結果
        reward_record = []
        UB_price = None
        LB_price = None

        for i, row in temp.iterrows():
            if row["status"] == "UB_open" and UB_price is None:
                UB_price = row["Close"]

            elif row["status"] == "LB_open" and LB_price is None:
                LB_price = row["Close"]

            elif row["status"] == "UB_close" and UB_price is not None:
                simple_return = (row["Close"] - UB_price) / UB_price
                reward_record.append(simple_return)
                UB_price = None

            elif row["status"] == "LB_close" and LB_price is not None:
                simple_return = -(row["Close"] - LB_price) / LB_price
                reward_record.append(simple_return)
                LB_price = None

            elif row["status"] == "close" and UB_price is not None:
                simple_return = (row["Close"] - UB_price) / UB_price
                reward_record.append(simple_return)
                UB_price = None

            elif row["status"] == "close" and LB_price is not None:
                simple_return = -(row["Close"] - LB_price) / LB_price
                reward_record.append(simple_return)
                LB_price = None

        all_ticker_reward.append(sum(reward_record))

    trading_result[state] = all_ticker_reward

# write_cache_file(trading_result, "data_file/SP500_component_trading_result.pkl")

# temp = load_cache_file("data_file/SP500_component_trading_result.pkl")

path = "figure_file"

dqn_result = load_cache_file("pre-train-fixWindow/test_result_119.pkl")

trading_table = pd.DataFrame(trading_result)

above_median_x, above_median_y = [], []
below_median_x, below_median_y = [], []

for x, y in enumerate(trading_table.median()):
    value = list(dqn_result["reward_result"].values())[x]
    if value > y:
        above_median_x.append(x + 1)
        above_median_y.append(value)

    else:
        below_median_x.append(x + 1)
        below_median_y.append(value)

plt.figure(figsize=(12, 6))
plt.boxplot(trading_table)
plt.scatter(
    above_median_x,
    above_median_y,
    s=90,
    color="red",
    marker="o",
    label=f"{len(above_median_x)}",
)
plt.scatter(
    below_median_x,
    below_median_y,
    s=90,
    color="blue",
    marker="x",
    label=f"{len(below_median_x)}",
)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend()
plt.savefig(f"{path}/5.png", dpi=200, bbox_inches="tight")
plt.show()

rank_result = []

for i in range(len(dqn_result["reward_result"])):
    target = list(dqn_result["reward_result"].values())[i]
    value = trading_table[f"state_{i + 1}"].sort_values(ascending=False)
    rank = 1

    for j in value:
        if target < j:
            rank += 1
    rank_result.append(rank)

trading_result_mean = trading_table.mean().tolist()
trading_result_mean_rank = []

for i in range(len(trading_result_mean)):
    target = trading_result_mean[i]
    value = trading_table[f"state_{i + 1}"].sort_values(ascending=False)
    rank = 1

    for j in value:
        if target < j:
            rank += 1
    trading_result_mean_rank.append(rank)

WRS = wilcoxon(x=rank_result, y=trading_result_mean_rank, alternative="less")
print(f"p.value of WRS: {WRS[1]}")

plt.figure(figsize=(10, 8))
plt.scatter(dqn_result["reward_result"].values(), trading_result_mean)
plt.plot(
    np.linspace(-0.06, 0.06, 200),
    np.linspace(-0.06, 0.06, 200),
    color="green",
    alpha=0.6,
)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.xlabel("DQN", fontsize=14)
plt.ylabel("Mean", fontsize=14)
plt.title("Reward", fontsize=16)
plt.savefig(f"{path}/8.png", dpi=200, bbox_inches="tight")
plt.plot()
