# Import the necessary packages
import os
import random
import warnings
from copy import deepcopy
from time import time

import numpy as np

from model.DQN import DQN_Agent
from model.environment import envTrader
from utils_file.tools import load_cache_file, show_elapsed_time, write_cache_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# Load environment data
state_space_trading_train = load_cache_file(
    "data_file/env_dqn_space/state_space_trading_train.pkl"
)
state_space_trading_test = load_cache_file(
    "data_file/env_dqn_space/state_space_trading_test.pkl"
)

state_space_merge_train = load_cache_file(
    "data_file/env_dqn_space/state_space_merge_train.pkl"
)
state_space_merge_test = load_cache_file(
    "data_file/env_dqn_space/state_space_merge_test.pkl"
)

coint_coef_train = load_cache_file("data_file/env_dqn_space/coint_coef_train.pkl")
coint_coef_test = load_cache_file("data_file/env_dqn_space/coint_coef_test.pkl")

action_space = [k for k in state_space_trading_train["state_1"].keys()]

key_mapping = {f"state_{i}": f"state_{i + 84}" for i in range(1, 37)}
state_space_trading_test = {
    key_mapping.get(k, k): v for k, v in state_space_trading_test.items()
}
state_space_merge_test = {
    key_mapping.get(k, k): v for k, v in state_space_merge_test.items()
}
coint_coef_test = {key_mapping.get(k, k): v for k, v in coint_coef_test.items()}

# Set up parameters
state_size = state_space_merge_train["state_1"].shape[0]
action_size = len(action_space)

max_memory = 2000
gamma = 0.5
epsilon = 0.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 10

# Initialize agent
agent = DQN_Agent(
    state_size=state_size,
    action_size=action_size,
    max_memory=max_memory,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
    learning_rate=learning_rate,
    batch_size=batch_size,
    pre_train_model="pre-train/cum_pre-train-model_ 3.9925.pth",
    pre_train_mode="train",
)

result = {"action_result": {}, "reward_result": {}}

seed = 112225024
random.seed(seed)
np.random.seed(seed)

# 預測 OOS 的第一個月
key_state = list(state_space_trading_test.keys())[0]
state_space_trading = deepcopy(state_space_trading_train)
state_space_trading[key_state] = state_space_trading_test[key_state]

state_space_merge = deepcopy(state_space_merge_train)
state_space_merge[key_state] = state_space_merge_test[key_state]

coint_coef = deepcopy(coint_coef_train)
coint_coef[key_state] = coint_coef_test[key_state]

env = envTrader(
    trade_state_space=state_space_trading,
    agent_state_space=state_space_merge,
    coint_coef_space=coint_coef,
    action_space=action_space,
)

env.idx = int(key_state.split("_")[1])
state = env.agent_state_space[f"state_{env.idx}"]
state = np.reshape(state, [1, state_size])

action_idx = agent.sample_action(state)
action = action_space[action_idx]
next_state, reward, cum_reward, done = env.step(action)
env.render(action, reward)


result["action_result"][key_state] = action
result["reward_result"][key_state] = reward

# 預測 OOS 的第二個月到最後一個月
for idx in range(1, len(state_space_merge_test) + 1):
    
    print(f"Processing State: {idx + 85}")
    epsilon = 1.0

    if idx == 1:
        pre_train_model = "pre-train/cum_pre-train-model_ 3.9925.pth"

    else:
        pre_train_model = "pre-train/pre-train-model-test.pth"
    
    ## Initialize agent
    
    max_memory = 2000
    gamma = 0.5
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    learning_rate = 0.001
    batch_size = 10
    
    agent = DQN_Agent(
        state_size=state_size,
        action_size=action_size,
        max_memory=max_memory,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        learning_rate=learning_rate,
        batch_size=batch_size,
        pre_train_model=pre_train_model,
        pre_train_mode="train",
    )

    ## Set up training parameters
    episode = 0
    stop_remain = 500
    stop_flag = False
    max_cum_reward = 0
    
    env = envTrader(
        trade_state_space=state_space_trading,
        agent_state_space=state_space_merge,
        coint_coef_space=coint_coef,
        action_space=action_space,
    )
    
    ## 開始重新配適模型
    start_time = time()
    while not stop_flag:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        cum_reward_history = []

        while not done:
            action_idx = agent.sample_action(state)
            action = action_space[action_idx]
            next_state, reward, cum_reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay()

            cum_reward_history.append(cum_reward)

        agent.epsilon_decrease()

        if cum_reward > max_cum_reward:
            max_cum_reward = cum_reward
            agent.save_model("pre-train/pre-train-model-test.pth")

        episode += 1

        if episode == 1:
            record_max = np.trapz(
                cum_reward_history, np.arange(len(cum_reward_history))
            )
            record_num = 0

        else:
            record_value = np.trapz(
                cum_reward_history, np.arange(len(cum_reward_history))
            )
            if record_value > record_max:
                record_max = record_value
                record_num = 0

            else:
                record_num += 1

                if record_num == stop_remain:
                    stop_flag = True
                    print(f"Early stopping at episode {episode}.")

    show_elapsed_time(time() - start_time)
    
    ## 預測下一個月
    key_state = list(state_space_trading_test.keys())[idx]

    state_space_trading[key_state] = state_space_trading_test[key_state]
    state_space_merge[key_state] = state_space_merge_test[key_state]
    coint_coef[key_state] = coint_coef_test[key_state]

    env = envTrader(
        trade_state_space=state_space_trading,
        agent_state_space=state_space_merge,
        coint_coef_space=coint_coef,
        action_space=action_space,
    )
    
    env.idx = int(key_state.split("_")[1])
    state = env.agent_state_space[f"state_{env.idx}"]
    state = np.reshape(state, [1, state_size])

    action_idx = agent.sample_action(state)
    action = action_space[action_idx]
    next_state, reward, cum_reward, done = env.step(action)
    env.render(action, reward)
        
    result["action_result"][key_state] = action
    result["reward_result"][key_state] = reward
    write_cache_file(result, f"pre-train/test_result_{idx+85}.pkl")
    
    print("-" * 20)
