import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from utils_file.tools import load_cache_file

warnings.filterwarnings("ignore")

path = "figure_file"

train_result = load_cache_file("pre-train-extendWindow/train_39120.pkl")

train_result = {key: train_result[key]["episode_3136"] for key in train_result.keys()}

# comparison target: reduced S&P500 components
train_data = pd.read_csv("data_file/train_PT.csv")

tickers = train_data["Ticker"].unique()

# 將個股的股價資料新增 Bollinger bands
BB_value = []
for ticker in tickers:
    temp = train_data[train_data["Ticker"] == ticker]
    temp["ma_line"] = temp["Close"].rolling(window=5).mean()
    std = temp["Close"].rolling(window=5).std()
    temp["upper_bound"] = temp["ma_line"] + std
    temp["lower_bound"] = temp["ma_line"] - std
    temp["log_rtn"] = np.log(temp["Close"] / temp["Close"].shift(1))
    temp = temp.dropna()
    BB_value.append(temp)

train_append = pd.concat(BB_value, ignore_index=True)

# 將個股的股價資料按照每個月切割出來，並存於字典中
split_train = {}

unique_years = train_append["Year"].unique()
unique_months = train_append["Month"].unique()

idx = 1
for year in unique_years:
    for month in unique_months:
        split_train[f"state_{idx}"] = {}
        
        date_mask  = (train_append["Year"] == year) & (train_append["Month"] == month)
        mask_table = train_append[date_mask]
        
        for ticker in tickers:
            split_train[f"state_{idx}"][ticker] = mask_table[mask_table["Ticker"] == ticker]
        
        idx += 1

# 每一支成分股都執行一次自定義交易策略
trading_BB_result = {}

for state in list(split_train.keys()):
    all_ticker_reward = []
    
    for ticker in tickers:
        temp = split_train[state][ticker]
        
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

    trading_BB_result[state] = all_ticker_reward

# 每一支成分股都執行一次 B&H 交易策略
trading_BandH_result = {}

for state in list(split_train.keys()):
    all_ticker_reward = []
    
    for ticker in tickers:
        all_ticker_reward.append(np.exp(split_train[state][ticker]["log_rtn"].sum()) - 1)
    
    trading_BandH_result[state] = all_ticker_reward

trading_BB_table = pd.DataFrame(trading_BB_result)

# ----- Result: Bollinger Bands ----- #
above_median_x, above_median_y = [], []
below_median_x, below_median_y = [], []

for x, y in enumerate(trading_BB_table.median()):
    value = train_result["reward_result"][x]
    
    if value > y:
        above_median_x.append(x + 1)
        above_median_y.append(value)
    
    else:
        below_median_x.append(x + 1)
        below_median_y.append(value)

# boxplot for reward
plt.figure(figsize=(10, 8))
plt.boxplot(trading_BB_table)
plt.scatter(above_median_x, above_median_y, s=90, color="red", marker="o", 
            label=f"{len(above_median_x)}")
plt.scatter(below_median_x, below_median_y, s=90, color="blue", marker="x", 
            label=f"{len(below_median_x)}")
plt.xticks([])
plt.xlabel("Month", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend()
plt.title("Target: Reduced Components  Strategy: Bollinger Bands", fontsize=16)
plt.show()

# WRS test
def rank_finder(target_list):
    result = []
    
    for i in range(len(target_list)):
        target = target_list[i]
        value = trading_BB_table[f"state_{i + 1}"].sort_values(ascending=False)
        rank = 1
        
        for j in value:
            if target <= j:
                rank += 1
        
        result.append(rank)
    
    return result

train_rank_result = rank_finder(train_result["reward_result"])
trading_BB_mean = trading_BB_table.mean()
trading_BB_mean_rank = rank_finder(trading_BB_mean.tolist())

WRS = wilcoxon(train_rank_result, trading_BB_mean_rank, alternative="less")
# print(f"p.value of WRS: {WRS[1]}")

plt.figure(figsize=(10, 8))
plt.scatter(trading_BB_mean, train_result["reward_result"])
plt.plot(
    np.linspace(-0.06, 0.15, 200),
    np.linspace(-0.06, 0.15, 200),
    color="green",
    alpha=0.6,
)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.xlabel("Mean of Target", fontsize=14)
plt.ylabel("DQN", fontsize=14)
plt.title("Reward Comparison: Mean of Target vs. DQN", fontsize=16)
plt.text(0.05, -0.05, f"p.value of WRS: {WRS[1]: .1e}")
plt.plot()

trading_BB_q3 = trading_BB_table.quantile(0.75)
trading_BB_q3_rank = rank_finder(trading_BB_q3.tolist())

WRS = wilcoxon(train_rank_result, trading_BB_q3_rank, alternative="less")
# print(f"p.value of WRS: {WRS[1]}")

plt.figure(figsize=(10, 8))
plt.scatter(trading_BB_q3, train_result["reward_result"])
plt.plot(
    np.linspace(-0.06, 0.15, 200),
    np.linspace(-0.06, 0.15, 200),
    color="green",
    alpha=0.6,
)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.xlabel("Q3 of Target", fontsize=14)
plt.ylabel("DQN", fontsize=14)
plt.title("Reward Comparison: Q3 of Target vs. DQN", fontsize=16)
plt.text(0.05, -0.05, f"p.value of WRS: {WRS[1]: .1e}")
plt.plot()

# ----- Result: Bollinger Bands ----- #
above_median_x, above_median_y = [], []
below_median_x, below_median_y = [], []

for x, y in enumerate(trading_BB_table.median()):
    value = train_result["reward_result"][x]
    
    if value > y:
        above_median_x.append(x + 1)
        above_median_y.append(value)
    
    else:
        below_median_x.append(x + 1)
        below_median_y.append(value)

# boxplot for reward
plt.figure(figsize=(10, 8))
plt.boxplot(trading_BB_table)
plt.scatter(above_median_x, above_median_y, s=90, color="red", marker="o", 
            label=f"{len(above_median_x)}")
plt.scatter(below_median_x, below_median_y, s=90, color="blue", marker="x", 
            label=f"{len(below_median_x)}")
plt.xticks([])
plt.xlabel("Month", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend()
plt.title("Target: Reduced Components  Strategy: Bollinger Bands", fontsize=16)
plt.show()

# WRS test
def rank_finder(target_list):
    result = []
    
    for i in range(len(target_list)):
        target = target_list[i]
        value = trading_BB_table[f"state_{i + 1}"].sort_values(ascending=False)
        rank = 1
        
        for j in value:
            if target <= j:
                rank += 1
        
        result.append(rank)
    
    return result

train_rank_result = rank_finder(train_result["reward_result"])
trading_BB_mean = trading_BB_table.mean()
trading_BB_mean_rank = rank_finder(trading_BB_mean.tolist())

WRS = wilcoxon(train_rank_result, trading_BB_mean_rank, alternative="less")
# print(f"p.value of WRS: {WRS[1]}")

plt.figure(figsize=(10, 8))
plt.scatter(trading_BB_mean, train_result["reward_result"])
plt.plot(
    np.linspace(-0.06, 0.15, 200),
    np.linspace(-0.06, 0.15, 200),
    color="green",
    alpha=0.6,
)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.xlabel("Mean of Target", fontsize=14)
plt.ylabel("DQN", fontsize=14)
plt.title("Reward Comparison: Mean of Target vs. DQN", fontsize=16)
plt.text(0.05, -0.05, f"p.value of WRS: {WRS[1]: .1e}")
plt.plot()

trading_BB_q3 = trading_BB_table.quantile(0.75)
trading_BB_q3_rank = rank_finder(trading_BB_q3.tolist())

WRS = wilcoxon(train_rank_result, trading_BB_q3_rank, alternative="less")
# print(f"p.value of WRS: {WRS[1]}")

plt.figure(figsize=(10, 8))
plt.scatter(trading_BB_q3, train_result["reward_result"])
plt.plot(
    np.linspace(-0.06, 0.15, 200),
    np.linspace(-0.06, 0.15, 200),
    color="green",
    alpha=0.6,
)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.1)
plt.xlabel("Q3 of Target", fontsize=14)
plt.ylabel("DQN", fontsize=14)
plt.title("Reward Comparison: Q3 of Target vs. DQN", fontsize=16)
plt.text(0.05, -0.05, f"p.value of WRS: {WRS[1]: .1e}")
plt.plot()