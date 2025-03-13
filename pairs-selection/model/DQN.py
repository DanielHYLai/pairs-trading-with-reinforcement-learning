# Import the necessary packages
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN_Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        max_memory: int = 2000,
        gamma: float = 0.6,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
        learning_rate: float = 0.001,
        batch_size: int = 10,
        pre_train_model: str = None,
        pre_train_mode: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=max_memory)

        # hyperparamters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pre_train_model is not None:
            self.model = torch.load(pre_train_model).to(self.device)
            self.target = torch.load(pre_train_model).to(self.device)

            if pre_train_mode == "eval":
                self.model.eval()

            elif pre_train_mode == "train":
                self.model.train()

            else:
                raise ValueError(f"There is no mode called {pre_train_mode}.")

        else:
            self.model = self.policy_net().to(self.device)
            self.target = self.policy_net().to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def policy_net(self):
        """
        Process the model.

        Returns:
        -------
        result : torch.nn.Sequential
            The processed model
        """

        result = nn.Sequential(
            nn.Linear(self.state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size),
        )

        return result
    
    def remember(self, state, action, reward, next_state, done):
        """
        Remember the experience.

        Parameters:
        ----------
        state : np.array
            The current state of the agent

        action : int
            The action taken by the agent

        reward : float
            The reward received by the agent

        next_state : np.array
            The next state of the agent

        done : bool
            The flag indicating the end of the episode
        """

        self.memory.append((state, action, reward, next_state, done))
    
    def sample_action(self, state):
        """
        Sample an action.

        Parameters:
        ----------
        state : np.array
            The current state of the agent

        Returns:
        -------
        action : int
            The action taken by the agent
        """

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        else:
            state = torch.FloatTensor(state).to(self.device)

            with torch.no_grad():
                action_value = self.model(state)

            return torch.argmax(action_value).item()
    
    def replay(self):
        """
        Replay the experience.
        """

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward

            if not done:
                target += self.gamma * torch.max(self.model(next_state)).detach()

            target_net = self.model(state).clone()
            target_net[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_net.detach())
            loss.backward()
            self.optimizer.step()

    
    def epsilon_decrease(self):
        """
        Decay the epsilon value till reach setting minimal.
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, file_name):
        """
        Save the model.

        Parameters:
        ----------
        file_name : str
            The name of the file
        """

        try:
            torch.save(self.model, f"{file_name}")
            print("Save model successfully.")

        except:
            print("Save model failed.")
