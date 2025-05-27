class envTrader_fee:
    def __init__(
        self,
        trade_state_space: dict,
        agent_state_space: dict,
        coint_coef_space: dict,
        action_space: list,
    ):
        self.trade_state_space = trade_state_space
        self.agent_state_space = agent_state_space
        self.coint_coef_space = coint_coef_space
        self.action_space = action_space
        self.idx = int(list(self.trade_state_space.keys())[0].split("_")[1])

        self.trade_state = self.trade_state_space[f"state_{self.idx}"]
        self.agent_state = self.agent_state_space[f"state_{self.idx}"]
        self.coint_coef = self.coint_coef_space[f"state_{self.idx}"]

        self.cumulative_reward = 0
        self.done = False

    def reset(self):
        """
        Initialize the environment to the initial state.

        Returns:
        -------
        agent_state : np.array
            The initial state of the agent
        """

        self.idx = int(list(self.trade_state_space.keys())[0].split("_")[1])
        self.trade_state = self.trade_state_space[f"state_{self.idx}"]
        self.agent_state = self.agent_state_space[f"state_{self.idx}"]
        self.coint_coef = self.coint_coef_space[f"state_{self.idx}"]
        self.cumulative_reward = 0
        self.done = False

        return self.agent_state

    def return_calculator(self, close_1, close_2, return_1, return_2, coef):
        """
        Calculate the total return of a transaction.

        Parameters:
        ----------
        close_1 : float
            The closing price of the first stock

        close_2 : float
            The closing price of the second stock

        return_1 : float
            The simple return of the first stock

        return_2 : float
            The simple return of the second stock

        coef : float
            The cointegration coefficient of the first stock

        Returns:
        -------
        result : float
            The total return of the transaction

        """
        denominator = coef * close_1 + close_2
        result = (((coef * close_1) / denominator) * return_1) + (
            ((close_2) / denominator) * return_2
        )

        transaction_fee = 0.0001
        result -= transaction_fee

        return result

    def step(self, action):
        """
        All the processing from the previous state to the next state.

        Parameters:
        ----------
        action : int
           The action taken by the agent

        Returns:
        -------
        next_agent_state : np.array
            The next state of the agent

        reward : float
            The reward obtained from executing the trading strategy

        cumulative_return : float
            The cumulative return obtained from executing the trading strategy

        done : bool
            Terminal flag
        """
        current_idx = self.idx
        current_trade_state = self.trade_state_space[f"state_{current_idx}"][action]
        current_action_idx = self.action_space.index(action)
        current_coef = self.coint_coef_space[f"state_{current_idx}"][current_action_idx]

        # Check whether the spreads have touched the upper bound or lower bound
        ub_open_flag = False
        lb_open_flag = False

        status_list = []

        for i, row in current_trade_state.iterrows():
            if row["spread"] > row["upper_bound"] and ub_open_flag is False:
                status_list.append("UB_open")
                ub_open_flag = True

            elif row["spread"] < row["ma_line"] and ub_open_flag is True:
                status_list.append("UB_close")
                ub_open_flag = False

            elif row["spread"] < row["lower_bound"] and lb_open_flag is False:
                status_list.append("LB_open")
                lb_open_flag = True

            elif row["spread"] > row["ma_line"] and lb_open_flag is True:
                status_list.append("LB_close")
                lb_open_flag = False

            else:
                status_list.append(0)

        status_list[-1] = "close"
        current_trade_state["status"] = status_list

        # Calculate the reward; that is, execute the trading strategy
        reward_record = []
        UB_long_price = None
        UB_short_price = None
        LB_long_price = None
        LB_short_price = None

        for i, row in current_trade_state.iterrows():
            if row["status"] == "UB_open" and UB_long_price is None:
                UB_long_price = row["close_1"]
                UB_short_price = row["close_2"]

            elif row["status"] == "LB_open" and LB_long_price is None:
                LB_long_price = row["close_2"]
                LB_short_price = row["close_1"]

            elif row["status"] == "UB_close" and UB_long_price is not None:
                return_1 = (row["close_1"] - UB_long_price) / UB_long_price
                return_2 = -(row["close_2"] - UB_short_price) / UB_short_price

                total_return = self.return_calculator(
                    UB_long_price, UB_short_price, return_1, return_2, current_coef
                )
                reward_record.append(total_return)

                UB_long_price, UB_short_price = None, None

            elif row["status"] == "LB_close" and LB_long_price is not None:
                return_1 = -(row["close_1"] - LB_short_price) / LB_short_price
                return_2 = (row["close_2"] - LB_long_price) / LB_long_price

                total_return = self.return_calculator(
                    LB_short_price, LB_long_price, return_1, return_2, current_coef
                )
                reward_record.append(total_return)

                LB_long_price, LB_short_price = None, None

            elif row["status"] == "close" and UB_long_price is not None:
                return_1 = (row["close_1"] - UB_long_price) / UB_long_price
                return_2 = -(row["close_2"] - UB_short_price) / UB_short_price

                total_return = self.return_calculator(
                    UB_long_price, UB_short_price, return_1, return_2, current_coef
                )
                reward_record.append(total_return)

                UB_long_price, UB_short_price = None, None

            elif row["status"] == "close" and LB_long_price is not None:
                return_1 = -(row["close_1"] - LB_short_price) / LB_short_price
                return_2 = (row["close_2"] - LB_long_price) / LB_long_price

                total_return = self.return_calculator(
                    LB_short_price, LB_long_price, return_1, return_2, current_coef
                )
                reward_record.append(total_return)

                LB_long_price, LB_short_price = None, None

            else:
                total_return = 0
                reward_record.append(total_return)

        reward = sum(reward_record)
        self.cumulative_reward += reward

        current_idx += 1
        self.idx = current_idx

        if current_idx > len(self.trade_state_space):
            self.done = True
            next_trade_state = self.trade_state_space[f"state_{current_idx - 1}"]
            next_agent_state = self.agent_state_space[f"state_{current_idx - 1}"]

        else:
            self.done = False
            next_trade_state = self.trade_state_space[f"state_{current_idx}"]
            next_agent_state = self.agent_state_space[f"state_{current_idx}"]

        self.trade_state = next_trade_state
        self.agent_state = next_agent_state

        return next_agent_state, reward, self.cumulative_reward, self.done

    def render(self, action, reward, max_cum_reward=None):
        """
        Display the results of the trading strategy.

        Parameters:
        ----------
        action : int
            The action taken by the agent

        reward : float
            The reward obtained from executing the trading strategy

        max_cum_reward : float
            The maximum cumulative reward obtained from executing the trading strategy
        """

        print("=" * 25)
        print(f"state {self.idx - 1}")
        print(f"action: {action}")
        print(f"reward: {reward: .4f}")
        print(f"cumulative reward: {self.cumulative_reward: .4f}")

        if max_cum_reward is not None:
            print(f"max cumulative reward: {max_cum_reward: .4f}")
