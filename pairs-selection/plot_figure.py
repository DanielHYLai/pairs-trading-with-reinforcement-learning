import matplotlib.pyplot as plt

from utils_file.tools import load_cache_file

test_result = load_cache_file("pre-train/test_result_120.pkl")

cum_reward = []
temp = 0
for v in test_result["reward_result"].values():
    temp += v
    cum_reward.append(temp)

plt.figure()
plt.plot(range(len(cum_reward)), cum_reward, color="black")
plt.show()