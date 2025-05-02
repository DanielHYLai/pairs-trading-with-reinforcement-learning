# Import the necessary packages
import os
from itertools import combinations
from time import time

import pandas as pd

from utils_file.tools import load_cache_file, show_elapsed_time, write_cache_file
from utils_file.utils_preprocessing import (
    state_merge,
    state_space_creator,
    trading_threshold_creator,
)

# Create a folder to store pickle files
path = "data_file"
if not os.path.exists(path):
    os.makedirs(path)

# Load stock price data
train = pd.read_csv("data_file/train_PT.csv", encoding="UTF-8")
test = pd.read_csv("data_file/test_PT.csv", encoding="UTF-8")

# Create the initial state space for the training data and testing data
ticker_combn = list(combinations(train["Ticker"].unique(), r=2))

path = "data_file/env_dqn_space"
if not os.path.exists(path):
    os.makedirs(path)

# start_time = time()
# temp_train = state_space_creator(data=train, ticker_list=ticker_combn, trade_window=1)
# state_space_train = temp_train[0]
# coint_coef_train  = temp_train[1]

# write_cache_file(obj=state_space_train, file_name="data_file/env_dqn_space/state_space_train.pkl")
# write_cache_file(obj=coint_coef_train, file_name="data_file/env_dqn_space/coint_coef_train.pkl")

# temp_test = state_space_creator(data=test, ticker_list=ticker_combn, trade_window=1)
# state_space_test = temp_test[0]
# coint_coef_test  = temp_test[1]

# write_cache_file(obj=state_space_test, file_name="data_file/env_dqn_space/state_space_test.pkl")
# write_cache_file(obj=coint_coef_test, file_name="data_file/env_dqn_space/coint_coef_test.pkl")
# show_elapsed_time(time() - start_time)

state_space_train = load_cache_file("data_file/env_dqn_space/state_space_train.pkl")
coint_coef_train = load_cache_file("data_file/env_dqn_space/coint_coef_train.pkl")

state_space_test = load_cache_file("data_file/env_dqn_space/state_space_test.pkl")
coint_coef_test = load_cache_file("data_file/env_dqn_space/coint_coef_test.pkl")

# Create the action space
action_space = [k for k in state_space_train.keys()]
# write_cache_file(obj=action_space, file_name="data_file/env_dqn_space/action_space.pkl")

# Calculate the trading threshold for traidng strategy
# start_time = time()
# state_space_trading_train = trading_threshold_creator(obj=state_space_train, ticker_list=ticker_combn)
# state_space_trading_test  = trading_threshold_creator(obj=state_space_test, ticker_list=ticker_combn)

# write_cache_file(obj=state_space_trading_train, file_name="data_file/env_dqn_space/state_space_trading_train.pkl")
# write_cache_file(obj=state_space_trading_test, file_name="data_file/env_dqn_space/state_space_trading_test.pkl")
# show_elapsed_time(time() - start_time)

state_space_trading_train = load_cache_file(
    "data_file/env_dqn_space/state_space_trading_train.pkl"
)
state_space_trading_test = load_cache_file(
    "data_file/env_dqn_space/state_space_trading_test.pkl"
)

# Merge and flatten the state of data
# start_time = time()
# state_space_merge_train = state_merge(obj=state_space_trading_train, ticker_list=ticker_combn)
# state_space_merge_test = state_merge(obj=state_space_trading_test, ticker_list=ticker_combn)

# write_cache_file(obj=state_space_merge_train, file_name="data_file/env_dqn_space/state_space_merge_train.pkl")
# write_cache_file(obj=state_space_merge_test, file_name="data_file/env_dqn_space/state_space_merge_test.pkl")
# show_elapsed_time(time() - start_time)

state_space_merge_train = load_cache_file(
    "data_file/env_dqn_space/state_space_merge_train.pkl"
)
state_space_merge_test = load_cache_file(
    "data_file/env_dqn_space/state_space_merge_test.pkl"
)
