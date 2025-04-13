import os

import matplotlib.pyplot as plt
import pandas as pd

from utils_file.tools import load_cache_file

path = "figure_file"
if not os.path.exists(path):
    os.makedirs(path)

ap_reward = load_cache_file("data_file/static_strategy/ap_reward_result_test.pkl")
ap_cum_reward = load_cache_file(
    "data_file/static_strategy/ap_cum_reward_result_test.pkl"
)
test_result = load_cache_file("pre-train/test_result_119.pkl")

cum_reward = []
temp = 0
for v in test_result["reward_result"].values():
    temp += v
    cum_reward.append(temp)

ap_reward_table = pd.DataFrame(ap_reward).T

above_median_x, above_median_y = [], []
below_median_x, below_median_y = [], []

for x, y in enumerate(ap_reward_table.median()):
    value = list(test_result["reward_result"].values())[x]
    if value > y:
        above_median_x.append(x + 1)
        above_median_y.append(value)

    else:
        below_median_x.append(x + 1)
        below_median_y.append(value)

plt.figure(figsize=(12, 6))
plt.boxplot(ap_reward_table)
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
plt.savefig(f"{path}/1.png", dpi=200, bbox_inches="tight")
plt.show()

ap_cum_reward_table = pd.DataFrame(ap_cum_reward).T
plt.figure(figsize=(12, 6))
plt.boxplot(ap_cum_reward_table)
plt.scatter(
    range(1, len(cum_reward) + 1), cum_reward, s=90, color="red", zorder=3, label="DQN"
)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Cumulative Reward", fontsize=14)
plt.legend()
plt.savefig(f"{path}/2.png", dpi=200, bbox_inches="tight")
plt.show()

rank_result = []

for i in range(len(cum_reward)):
    target = cum_reward[i]
    value = ap_cum_reward_table[i].sort_values(ascending=False)
    rank = 0
    for j in value:
        if target < j:
            rank += 1
    rank_result.append(rank)

plt.figure(figsize=(20, 8))
plt.plot(range(1, len(rank_result) + 1), rank_result, color="green")
plt.ylim(0, 1830)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Rank", fontsize=14)
plt.title("Rank of Cumulative Reward for Each Month", fontsize=16)
for i in range(len(rank_result)):
    plt.text(
        i + 1,
        rank_result[i] + 2,
        f"{rank_result[i]}",
        fontsize=12,
        ha="left",
        color="black",
    )
plt.savefig(f"{path}/3.png", dpi=200, bbox_inches="tight")
plt.show()

rank_result = []

for i in range(len(test_result["reward_result"])):
    target = list(test_result["reward_result"].values())[i]
    value = ap_reward_table[i].sort_values(ascending=False)
    rank = 0
    for j in value:
        if target < j:
            rank += 1
    rank_result.append(rank)

plt.figure(figsize=(20, 8))
plt.plot(range(1, len(rank_result) + 1), rank_result, color="green")
plt.ylim(0, 1830)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Rank", fontsize=14)
plt.title("Rank of Reward for Each Month", fontsize=16)
for i in range(len(rank_result)):
    plt.text(
        i + 1,
        rank_result[i],
        f"{rank_result[i]}",
        fontsize=12,
        ha="left",
        color="black",
    )
    plt.text(
        i + 1,
        rank_result[i] + 40,
        f"{list(test_result['action_result'].values())[i]}",
        fontsize=12,
        ha="left",
        color="blue",
    )
plt.savefig(f"{path}/4.png", dpi=200, bbox_inches="tight")
plt.show()

import numpy as np

ap_reward_mean = []
ap_reward_mean_rank = []

for i in range(len(ap_reward["AMCR-BBY"])):
    temp = []
    for v in ap_reward.values():
        temp.append(v[i])
    ap_reward_mean.append(np.mean(temp))

for i in range(len(test_result["reward_result"])):
    target = ap_reward_mean[i]
    value = ap_reward_table[i].sort_values(ascending=False)
    rank = 0
    for j in value:
        if target < j:
            rank += 1
    ap_reward_mean_rank.append(rank)

from scipy.stats import wilcoxon

WRS = wilcoxon(x=rank_result, y=ap_reward_mean, alternative="greater")
print(f"p.value of WRS: {WRS[1]}")
