import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from simulation_model import StockMarketModel
import logging

logging.basicConfig(filename="utils/outputs/observations.log", level=logging.INFO, format="%(asctime)s - %(message)s")


class Env(gym.Env):
    """
    Purpose: Implements the Env class, serving as a Gym-compatible wrapper for an AgentPy-based stock market simulation.

    Responsibilities:
        - Initialize and manage the simulation environment, including the setup of agents, stock data, and simulation parameters.
        - Facilitate Gym-style reinforcement learning interactions, providing action and observation spaces.
        - Track and manage environment states, including agent actions, rewards, and trading history across episodes.
        - Convert simulation data into structured formats for analysis and debugging.

    Main Components:
        - Env.__init__():
            Sets up the environment by initializing attributes, loading stock data,
                and configuring the simulation's initial state.

        - Env.reset():
            Resets the environment to start a new episode while logging and re-initializing states for reproducibility.

        - Env.step(action):
            Executes a single timestep by processing the agent's actions, updating the state, calculating rewards,
                and managing transitions.

        - Env.render():
            Provides visual or textual representations of the simulation's current state for monitoring or
                debugging purposes.

        - Env.save_trade_history():
            Stores the trading history and final state of the simulation for post-episode analysis.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_data, agent_counts, global_seed=42, max_steps_per_episode=61):
        super(Env, self).__init__()
        self.final_net_worth = None
        self.stock_data = stock_data
        self.max_steps_per_episode = max_steps_per_episode
        self.global_seed = global_seed
        self.rng = np.random.default_rng(global_seed)
        self.rl_model = None
        self.last_obs = None
        self.current_step = 0
        self.episode_step = 0
        self.episode_count = 0
        self.previous_start_days = []
        self.check = False

        max_start_index = len(self.stock_data) - self.max_steps_per_episode
        step_interval = self.max_steps_per_episode // 2  # Overlap allowed by 50%

        self.valid_start_days_pool = list(range(0, max_start_index, step_interval))
        self.used_start_days = []

        # Simulation parameters
        self.parameters = {
            'global_seed': global_seed,
            'stock_data': self.stock_data,
            'agent_counts': agent_counts,
        }

        # Initialize AgentPy stock market simulation
        self.model = StockMarketModel(self.parameters)

        # === Define action and observation space for Rl agent ===
        self.tickers = list(self.stock_data.columns.get_level_values(1).unique())
        self.num_stocks = len(self.tickers)

        # Define Gym action space (multi-stock trading: Hold, Buy, Sell per stock): 0: Hold, 1: Sell, 2: Buy
        self.action_space = spaces.Discrete(3 ** self.num_stocks)

        # === Observation Space ===
        self.num_features = (
                6 * self.num_stocks +  # technical indicators
                9 +  # RL summary
                5 * self.num_stocks +  # RL per-ticker state
                7 +  # reactive avg metrics
                3 * self.num_stocks   # agent action sentiment
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)

        print("Gym environment initialized!")

    def reset(self, seed=None, options=None):
        """
        Resets the simulation for a new episode with a random starting point.
        :param seed: for randomness in this episode.
        :param options: (dict, optional) Additional configuration options for future used in needed
        :return tuple: (initial_observation, info)
        """
        super().reset(seed=seed)
        self.model.setup()
        obs = self._get_observation()

        # === Adjust episode length based on epsilon (exploration rate) ===
        if self.rl_model.exploration_rate <= 0.15 and self.episode_count+1 > 1 and self.check == False:
            self.max_steps_per_episode = self.max_steps_per_episode//2  # Half year fine-tuning
            self.check = True

        if len(self.used_start_days) == len(self.valid_start_days_pool):
            self.used_start_days = []
            self.rng.shuffle(self.valid_start_days_pool)

        for day in self.valid_start_days_pool:
            if day not in self.used_start_days:
                self.current_step = day
                self.used_start_days.append(day)
                break

        # Sync the model's internal time step with the environment's current step
        self.episode_step = 0
        self.model.t = self.current_step
        print(f"\n ===== Episode {self.episode_count + 1} initialized starting from day {self.current_step} ===== ")
        return obs, {}

    def step(self, action):
        """Executes one step in the simulation."""
        self.current_step += 1
        self.episode_step += 1

        # Check data bounds and update model.t first
        if self.current_step >= len(self.stock_data):
            print(f" WARNING :::::: Reached end of stock data at step {self.current_step}!")
            return self._get_observation(), 0, True, False, {}
        self.model.t = self.current_step

        # Decode base-3 action into per-stock commands (0=Hold, 1=Sell, 2=Buy)
        decoded_actions = []
        original_action = int(action)
        for _ in range(self.num_stocks):
            decoded_actions.append(int(original_action % 3))
            original_action //= 3
        decoded_actions.reverse()  # Restore correct order

        print(
            f" === Step {self.episode_step} at Episode {self.episode_count + 1} === \n"
            f" === RL Action Decoded -> {decoded_actions}\n"
            f" === Exploration Rate = {self.rl_model.exploration_rate}\n"
            f" === RL Model Action -> {action}")

        # Execute the decoded actions in the simulation, allow simulation to run for the current step
        self.model.step(rl_action=decoded_actions)

        # Extract observation, reward, and done flag
        obs = self._get_observation()
        base_reward = self._calculate_reward()
        penalty = self.calculate_trade_move_penalty()
        reward = base_reward + penalty
        done = self.episode_step == self.max_steps_per_episode

        logging.info(f" Current Total Reward: {reward}\n")

        if done:
            print(f"\n✅ Episode {self.episode_count + 1} complete!")
            self.final_net_worth = self.model.rl_agent.get_net_worth()
            self.last_obs = self._get_observation()
            self.save_trade_history()
            self.model.end()
            self.episode_count += 1

        return obs, reward, done, False, {}

    def save_trade_history(self):
        """
        Saves trade history with episode-specific filename.
        """
        if not self.model.history:
            print(f" WARNING ::::::  No trade history recorded for Episode {self.episode_count + 1}!")
            return
        df = pd.DataFrame(self.model.history)
        filename = f"Trade_History/trade_history_ep{self.episode_count + 1}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Trade history for Episode {self.episode_count + 1} saved to {filename}!")

    def _get_observation(self):
        """
        Constructs the RL agent's observation vector by aggregating and normalising technical indicators,
            peer(Reactive Agents) behaviours, and the agent's own financial and behavioural state.
        """
        if self.model.rl_agent is None:
            raise ValueError(" WARNING ::::::  No RL agent assigned in StockMarketModel!")
        agent_info = self.model.rl_agent

        def normalize(value, min_val, max_val):
            """Scales a value into the range [0,1] using Min-Max Scaling."""
            if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
                return 0.5
            return (value - min_val) / (max_val - min_val)

        # === 1. Normalize Stock Prices === #
        all_stock_features = []

        for ticker in self.tickers:
            raw_price = float(self.stock_data.xs('Open', level=0, axis=1)[ticker].iloc[self.model.t])
            price_series = self.stock_data.xs('Open', level=0, axis=1)[ticker].iloc[:self.model.t + 1]
            min_price = price_series.min()
            max_price = price_series.max()
            normalized_raw_price = normalize(raw_price, min_price, max_price)

            # --- Short-term volatility (returns standard deviation over available days, max 5) ---
            returns_window = self.stock_data.xs('Open', level=0, axis=1)[ticker].iloc[
                             max(self.model.t - 4, 0):self.model.t + 1]
            returns = returns_window.pct_change().dropna()
            volatility_5d = returns.std() if len(returns) > 0 else 0
            normalized_volatility_5d = normalize(volatility_5d, 0, 0.1)

            # --- Average True Range (ATR), use longest available period (max 5 days) ---
            high_window = self.stock_data.xs('High', level=0, axis=1)[ticker].iloc[
                          max(self.model.t - 4, 0):self.model.t + 1]
            low_window = self.stock_data.xs('Low', level=0, axis=1)[ticker].iloc[
                         max(self.model.t - 4, 0):self.model.t + 1]
            close_window = self.stock_data.xs('Close', level=0, axis=1)[ticker].iloc[
                           max(self.model.t - 5, 0):self.model.t]
            tr_components = pd.concat([
                high_window - low_window,
                abs(high_window - close_window.shift(1)),
                abs(low_window - close_window.shift(1))
            ], axis=1).max(axis=1).dropna()

            atr_5d = tr_components.mean() if len(tr_components) > 0 else 0
            normalized_atr_5d = normalize(atr_5d, 0, 50)

            # --- Short-term moving average (up to 7-day MA) ---
            short_window_prices = self.stock_data.xs('Open', level=0, axis=1)[ticker].iloc[
                                  max(self.model.t - 6, 0):self.model.t + 1]
            short_ma = short_window_prices.mean() if len(short_window_prices) > 0 else raw_price
            normalized_short_ma = normalize(raw_price - short_ma, -100, 100)

            # --- Long-term moving average (up to 50-day MA) ---
            long_window_prices = self.stock_data.xs('Open', level=0, axis=1)[ticker].iloc[
                                 max(self.model.t - 49, 0):self.model.t + 1]
            long_ma = long_window_prices.mean() if len(long_window_prices) > 0 else raw_price
            normalized_long_ma = normalize(raw_price - long_ma, -100, 100)

            # --- MACD (use available periods up to 30 days) ---
            close_prices = self.stock_data.xs('Close', level=0, axis=1)[ticker].iloc[:self.model.t + 1]
            ema_span12 = close_prices.ewm(span=min(12, len(close_prices)), adjust=False).mean().iloc[-1]
            ema_span50 = close_prices.ewm(span=min(30, len(close_prices)), adjust=False).mean().iloc[
                -1]
            macd = ema_span12 - ema_span50
            normalized_macd = normalize(macd, -100, 100)

            # --- Append all indicators for this ticker ---
            stock_features = np.array([
                normalized_raw_price,
                normalized_volatility_5d,
                normalized_atr_5d,
                normalized_short_ma,
                normalized_long_ma,
                normalized_macd

            ],dtype=np.float32)
            all_stock_features.extend(stock_features)

        stock_data = np.array(all_stock_features, dtype=np.float32)

        # === 2. Normalize Overall Reactive Agent Portfolio and Performance Details === #
        # Filter reactive agents explicitly (excluding RL agent)
        reactive_agents = [agent for agent in self.model.agents if agent != self.model.rl_agent]
        reactive_portfolio_values = np.array([agent.get_portfolio_value() for agent in reactive_agents])
        reactive_cum_returns = np.array([agent.cumulative_return for agent in reactive_agents])
        reactive_profitability_scores = np.array([agent.profitability_score for agent in reactive_agents])
        reactive_win_rates = np.array([agent.win_rate for agent in reactive_agents])
        reactive_net_worths = np.array([agent.get_net_worth() for agent in reactive_agents])
        reactive_unrealized_pls = np.array([agent.unrealized_PL for agent in reactive_agents])
        reactive_realized_pls = np.array([agent.realized_PL for agent in reactive_agents])

        # Safe min/max
        min_portfolio = np.min(reactive_portfolio_values)
        max_portfolio = np.max(reactive_portfolio_values)
        min_cum_return = np.min(reactive_cum_returns)
        max_cum_return = np.max(reactive_cum_returns)
        min_profitability = np.min(reactive_profitability_scores)
        max_profitability = np.max(reactive_profitability_scores)
        min_win_rate = np.min(reactive_win_rates)
        max_win_rate = np.max(reactive_win_rates)
        min_net_worth = np.min(reactive_net_worths)
        max_net_worth = np.max(reactive_net_worths)
        min_unrealized_pl = np.min(reactive_unrealized_pls)
        max_unrealized_pl = np.max(reactive_unrealized_pls)
        min_realized_pl = np.min(reactive_realized_pls)
        max_realized_pl = np.max(reactive_realized_pls)

        # Normalized averages for the simulation reactive agents
        avg_portfolio_value = normalize(np.mean(reactive_portfolio_values), min_portfolio, max_portfolio)
        avg_cumulative_return = normalize(np.mean(reactive_cum_returns), min_cum_return, max_cum_return)
        avg_profitability_score = normalize(np.mean(reactive_profitability_scores), min_profitability,
                                            max_profitability)
        avg_win_rate = normalize(np.mean(reactive_win_rates), min_win_rate, max_win_rate)
        avg_net_worth = normalize(np.mean(reactive_net_worths), min_net_worth, max_net_worth)
        avg_unrealized_pl = normalize(np.mean(reactive_unrealized_pls), min_unrealized_pl, max_unrealized_pl)
        avg_realized_pl = normalize(np.mean(reactive_realized_pls), min_realized_pl, max_realized_pl)

        # === Agent Actions Weighted by Performance === #
        latest_day = self.stock_data.index[self.model.t]
        agent_action_data = []
        for ticker in self.tickers:
            buy_weight = 0.0
            sell_weight = 0.0
            hold_weight = 0.0

            for agent in reactive_agents:
                agent_id = agent.agent_id
                cum_return = agent.cumulative_return
                norm_return = normalize(cum_return, min_cum_return, max_cum_return)

                action_today = next(
                    (entry['Action'] for entry in reversed(self.model.history)
                     if entry['Agent'] == agent_id and entry['Ticker'] == ticker and entry['Date'] == latest_day),
                    "hold"
                )
                action_code = {"buy": 1, "sell": -1, "hold": 0}.get(action_today.lower(), 0)

                if action_code == 1:
                    buy_weight += norm_return
                elif action_code == -1:
                    sell_weight += norm_return
                else:
                    hold_weight += norm_return

            # Optional: Normalize the weights per ticker (so they sum to ~1.0)
            total_weight = buy_weight + sell_weight + hold_weight + 1e-6  # prevent div-by-zero
            agent_action_data.extend([
                buy_weight / total_weight,
                sell_weight / total_weight,
                hold_weight / total_weight
            ])

        reactive_agents_data = np.array([
                avg_portfolio_value,
                avg_cumulative_return,
                avg_profitability_score,
                avg_win_rate,
                avg_net_worth,
                avg_unrealized_pl,
                avg_realized_pl
            ], dtype=np.float32)


        # === 3. RL Agent's Own Metrics (Normalized) === #
        initial_net_worth = agent_info.net_worth_history[0] if len(
            agent_info.net_worth_history) > 0 else agent_info.budget

        rl_budget = normalize(agent_info.budget, 0, initial_net_worth)
        rl_portfolio_value = normalize(agent_info.portfolio_value, 0, initial_net_worth)
        rl_net_worth = normalize(agent_info.get_net_worth(), 0, initial_net_worth)
        rl_realized_pl = normalize(agent_info.realized_PL, -10000, 10000)
        rl_unrealized_pl = normalize(agent_info.unrealized_PL, -10000, 10000)
        rl_cumulative_return = normalize(agent_info.cumulative_return, -1, 1)
        rl_profitability_score = normalize(agent_info.profitability_score, -10, 10)
        rl_win_rate = normalize(agent_info.win_rate, 0, 100)
        rl_invalid_action = normalize(agent_info.invalid_action_count, 0, 10)

        rl_per_ticker_data = [] # RL Agent's Own Actions and data per ticker
        latest_day = self.stock_data.index[self.model.t]

        for ticker in self.tickers:
            rl_action_today = next(
                (entry['Action'] for entry in reversed(self.model.history)
                 if
                 entry['Agent'] == agent_info.agent_id and entry['Ticker'] == ticker and entry['Date'] == latest_day),
                "hold"
            )
            action_mapping = {"buy": 1, "sell": -1, "hold": 0}
            action = action_mapping.get(rl_action_today.lower(), 0)

            # RL Agent's Holdings per Ticker
            portfolio_entry = agent_info.portfolio.get(ticker, {'quantity': 0, 'average_price': 0})
            quantity = portfolio_entry['quantity']
            avg_price = portfolio_entry['average_price']

            # Market Price
            market_price = self.stock_data.loc[latest_day, ('Open', ticker)]
            holding_value = quantity * market_price
            affordable_quantity = agent_info.budget / market_price if market_price > 0 else 0
            roi = ((market_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

            net_worth = agent_info.get_net_worth()
            normalized_quantity = normalize(quantity, 0, 100)
            normalized_afford = normalize(affordable_quantity, 0, 100)
            normalized_value = normalize(holding_value, 0, net_worth if net_worth > 0 else 1)
            normalized_roi = normalize(roi, -100, 100)

            rl_per_ticker_data.extend([
                action,
                normalized_quantity,
                normalized_afford,
                normalized_value,
                normalized_roi
            ])

        rl_own_data = np.array([
            rl_budget,
            rl_portfolio_value,
            rl_net_worth,
            rl_realized_pl,
            rl_unrealized_pl,
            rl_cumulative_return,
            rl_profitability_score,
            rl_win_rate,
            rl_invalid_action
        ], dtype=np.float32)

        # === 4. Create Final Observation Vector with separator marking === #
        # SEP = np.array([-100000.0], dtype=np.float32)

        observation = np.concatenate([stock_data,
                                      rl_own_data,
                                      rl_per_ticker_data,
                                      reactive_agents_data,
                                      agent_action_data])

        logging.info(f"Observation: {observation}")
        # Safe check
        if len(observation) != self.num_features:
            raise ValueError(f" WARNING :::::: Shape Mismatch! Got {len(observation)}, expected {self.num_features}")
        return observation

    def _calculate_reward(self):
        """
        This reward function incentives both short-term profitability and long-term performance of the RL agent.

        It consists of:
        - A short-term reward for realised/unrealised P&L, daily return, and win rate.
        - A long-term reward based on cumulative return, yearly MWRR, stability (volatility), and performance score.
        - Additional bonuses for improving upon past performance in win rate and cumulative return.

        All inputs are normalised using tanh or linear scaling to maintain stability.
        The final reward is a weighted combination of both components, clipped to [0, 40] to prevent extreme Q-value
        jumps.
        """
        agent = self.model.rl_agent
        if agent is None:
            return 0

        # === Fetch core performance metrics ===
        PL = agent.realized_PL + agent.unrealized_PL
        daily_mwrr = agent.daily_mwrr/2 # reduce value
        win_rate = agent.win_rate
        cumulative_return = agent.cumulative_return
        yearly_mwrr = agent.yearly_mwrr
        performance_score = agent.performance_score
        trade_volatility_penalty = agent.trade_volatility

        # === Normalisation of inputs, using tan scale ===
        normalized_PL = np.tanh(PL/500)
        normalized_daily_mwrr = np.tanh(daily_mwrr/10)
        normalized_win_rate = (win_rate - 15) / 20  # smother normalization 0 to 1 for 15–40%
        normalized_cumulative_return = np.tanh(cumulative_return/5)
        normalized_yearly_mwrr = np.tanh(yearly_mwrr/20)
        normalized_perf_score = np.tanh(performance_score / 10)
        normalized_volatility_penalty = np.tanh(trade_volatility_penalty/5)

        # === Portfolio concentration (Gini penalty) ===
        quantities = np.array([pos['quantity'] for pos in agent.portfolio.values()])
        if quantities.sum() > 0:
            shares = quantities / quantities.sum()
            gini = 1 - np.sum(shares ** 2)
            norm_concentration_penalty = np.tanh((1 - gini) * 5)  # 0 = diverse, 1 = concentrated
        else:
            norm_concentration_penalty = 0

        # === Short-Term Reward (Immediate performance) 1 unit ===
        short_term_reward = (
                0.4 * normalized_win_rate +
                0.35 * normalized_PL +
                0.25 * normalized_daily_mwrr
        )
        # === Long-Term Reward (Stable long-run performance) 1 unit ===
        long_term_reward = (
                0.35 * normalized_cumulative_return +
                0.2 * normalized_yearly_mwrr +
                0.2 * normalized_perf_score -
                0.15 * norm_concentration_penalty -
                0.1 * normalized_volatility_penalty
        )
        # Net worth change reward component
        if len(agent.net_worth_history) > 1:
            net_worth_delta = agent.get_net_worth() - agent.net_worth_history[-2]
            reward_nw_delta = np.tanh(net_worth_delta / 200)
        else:
            reward_nw_delta = 0

        # Total Reward Composition
        total_base_reward = 0.3 * short_term_reward + 0.4 * long_term_reward + 0.3 * reward_nw_delta

        # === Extra Reward: self-improvement bonuses ===
        # === Improvement Bonuses ===
        bonus = 0.0
        if win_rate > 50 or win_rate > agent.prev_win_rate:
            bonus += (win_rate - agent.prev_win_rate) / 100
        agent.prev_win_rate = win_rate

        if cumulative_return > agent.prev_cumulative_return:
            bonus += (cumulative_return - agent.prev_cumulative_return) / 5
        agent.prev_cumulative_return = cumulative_return

        # Add milestone rewards
        initial_net_worth = agent.net_worth_history[0]
        current_net_worth = agent.get_net_worth()
        growth = current_net_worth - initial_net_worth

        if growth > 1000:
            bonus += 2.0
        elif growth > 5000:
            bonus += 6.0
        elif growth > 10000:
            bonus += 15.0

        total_reward = total_base_reward + bonus

        # Clip reward to prevent extreme spikes
        total_reward = np.clip(total_reward, -20, 20)

        # Logging for debugging
        logging.info(
            f"Step {self.episode_step} at Episode {self.episode_count} "
            f" Reward details: Short-term: {short_term_reward}"
            f" Long-term: {long_term_reward}"
            f" Total Reward (clipped): {total_reward}"
        )
        return total_reward

    def calculate_trade_move_penalty(self):
        """
        Calculates and returns the RL agent's total penalty for invalid trade actions.

        This function retrieves the per-step penalty accumulated by the RL agent
            (e.g., attempting to buy without budget, sell without holdings, or excessive holding),
            and applies a dynamic scaling factor that decreases over training time.

        This ensures stronger penalisation during early exploration phases, while gradually relaxing
            the penalty weight as the agent matures.

        The final penalty is clipped to the range [-5.0, 0.0] to prevent excessive impact on reward signals.
        After retrieval, the agent's internal penalty tracker is reset to zero.
        """
        agent = self.model.rl_agent
        # === Retrieve the internal updated penalty ===
        base_penalty = agent.today_invalid_action_penalty

        # Penalty for budget trap: stuck with low cash
        liquidity_penalty = -1.0 if agent.budget < (0.2 * agent.net_worth_history[0]) and  agent.get_portfolio_value() > (0.75 * agent.net_worth_history[0]) else 0.0

        # === Dynamic penalty scale (stronger early training) ===
        penalty_scaler = max(1.0, 5.0 - self.episode_count / 2000)

        # === Adjust and clip ===
        adjusted_penalty = base_penalty * penalty_scaler
        penalty = np.clip(adjusted_penalty, -5, 0.0)

        total_penalty = penalty + liquidity_penalty

        # Reset the penalty for next step
        agent.today_invalid_action_penalty = 0.0
        if self.episode_step < 2 and agent.total_trades == 0:
            total_penalty = 0.0
        # **Log penalty for debugging**
        logging.info(f"Step {self.model.t} | Invalid Action Penalty: {total_penalty}")
        return total_penalty

    def render(self, mode='human'):
        """Prints simulation state for debugging."""
        print(f"Step {self.current_step}: Net Worth = {self.model.agents[0].get_net_worth()}")
