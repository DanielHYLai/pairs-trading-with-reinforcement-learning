# Import the necessary packages
import os
import random
import warnings
from time import time

import numpy as np

from model.environment import envTrader
from model.DQN import DQN_Agent
from utils_file.tools import load_cache_file, show_elapsed_time, write_cache_file

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path = "pre-train"
if not os.path.exists(path):
    os.makedirs(path)

# Load environment data
state_space_trading = load_cache_file("data_file/env_dqn_space/state_space_trading_train.pkl")
state_space_merge = load_cache_file("data_file/env_dqn_space/state_space_merge_train.pkl")
coint_coef = load_cache_file("data_file/env_dqn_space/coint_coef_train.pkl")
action_space = [k for k in state_space_trading["state_1"].keys()]

# Initalize environment
env = envTrader(
    trade_state_space=state_space_trading,
    agent_state_space=state_space_merge,
    coint_coef_space=coint_coef,
    action_space=action_space,
)

# Set up parameters
state_size = state_space_merge["state_1"].shape[0]
action_size = len(action_space)

# Initialize agent
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
    batch_size=batch_size
)

# Set up training parameters
episode = 0
stop_remain = 500
stop_flag = False
max_cum_reward = 0

result = {"action_result": {}, "reward_result": {}, "cum_reward_result": {}}

seed = 415
random.seed(seed)
np.random.seed(seed)

# Start training
start_time = time()

while not stop_flag:
    print("-" * 20)
    print(f"episode {episode + 1}")
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    action_history = []
    reward_history = []
    cum_reward_history = []

    while not done:
        action_idx = agent.sample_action(state)
        action = action_space[action_idx]
        next_state, reward, cum_reward, done = env.step(action)
        if env.idx % 12 == 0:
            env.render(action, reward, max_cum_reward)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action_idx, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay()

        action_history.append(action)
        reward_history.append(reward)
        cum_reward_history.append(cum_reward)

    agent.epsilon_decrease()
    result["action_result"][f"episode_{episode + 1}"] = action_history
    result["reward_result"][f"episode_{episode + 1}"] = reward_history
    result["cum_reward_result"][f"episode_{episode + 1}"] = cum_reward_history

    if cum_reward > max_cum_reward:
        max_cum_reward = cum_reward
        agent.save_model(f"pre-train/pre-train-model_{cum_reward: .4f}.pth")
        write_cache_file(result, "pre-train/train_result.pkl")

    episode += 1

    if episode == 1:
        record_max = np.trapz(cum_reward_history, np.arange(len(cum_reward_history)))
        record_number = 0

    else:
        record_value = np.trapz(cum_reward_history, np.arange(len(cum_reward_history)))
        if record_value > record_max:
            record_max = record_value
            record_number = 0

        else:
            record_number += 1

            if record_number == stop_remain:
                stop_flag = True
                print(f"Early stopping at episode {episode}")

end_time = time() - start_time
show_elapsed_time(end_time)
