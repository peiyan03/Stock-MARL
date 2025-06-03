import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import glob
"""
This file serves as a comprehensive support module for backtesting systems and reinforcement-learning-based trading environments.

- Tools:
    Methods for data loading, performance summarization, and generating insightful visualizations related to trading activities:
    - load_trade_history: Load historical trade data.
    - summarize_agents_final_performance: Evaluate and summarize the final performances of trading agents.
    - plot_cumulative_return_trend: Plot cumulative return trends from trading results.
    - plot_reward_trend_per_episode: Visualize reward trends across episodes.
    - plot_single_trade_history: Generate detailed visualizations for individual trade activities.

- EpisodeTracker:
    Class designed to monitor and assess performance during model training or trading episodes:
    - Tracks rewards and net worths for each episode.
    - Maintains historical records and current episode metrics.
    - Supports early stopping mechanisms based on predefined performance criteria.
    - Saves episode-related data and metrics to a specified directory.
"""

class Tools:
    @staticmethod
    def load_trade_history(input_files):
        """
        Loads trade history from a single file or a list of files.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        df_list = [pd.read_csv(file) for file in input_files]
        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def summarize_agents_final_performance(input_files):
        """
        Summarizes the final recorded data for each agent type and computes averages.

        :param input_files: List of trade history files.
        :return avg_final_performance: Averages of final recorded values for each agent type in dataframe.
        """
        trade_history = Tools.load_trade_history(input_files)

        # Ensure Date is in datetime format
        trade_history['Date'] = pd.to_datetime(trade_history['Date'])

        # Extract agent type ( like 'MomentumTraderAgent_1 to _3')
        trade_history['Agent Type'] = trade_history['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))

        # Get last recorded entry per agent in each file
        last_entries = trade_history.sort_values(by=['Agent', 'Date']).groupby(['Agent Type', 'Agent']).last()

        # Select relevant columns for averaging
        columns_to_average = [
            'Budget', 'Net Worth', 'Portfolio Value',
            'Realized Profit/Loss', 'Unrealized Profit/Loss',
            'Daily MW Return Rate(%)', 'Monthly MW Return Rate (%)',
            'Quarter MW Return Rate (%)', 'Yearly MW Return Rate (%)',
            'Cumulative Return (%)', 'Win Rate (%)', 'Total Trades',
            'Profitability Score', 'Trade Volatility', 'Performance Score'
        ]

        # Compute averages across agent types
        avg_final_performance = last_entries.groupby('Agent Type')[columns_to_average].mean().reset_index()

        return avg_final_performance

    @staticmethod
    def plot_cumulative_return_trend(input_files):
            """
            Plots cumulative return trends for all agent types in multiple subplots, one per input file.

            :param input_files: List of trade history files.
            :return: graph of cumulative return trends for all agent types in multiple subplots
            """
            num_files = len(input_files)
            cols = 4  # Number of columns in subplot grid
            rows = (num_files // cols) + (num_files % cols > 0)  # Calculate required rows

            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), sharex=True, sharey=True)
            axes = axes.flatten()  # Flatten axes for easy iteration

            # Determine global y-axis limits for all subplots
            global_y_min, global_y_max = float('inf'), float('-inf')

            # Store agent type colors for consistency
            agent_types = set()
            agent_color_map = {}

            for i, file in enumerate(input_files):
                df = pd.read_csv(file)

                if 'Date' in df.columns and 'Agent' in df.columns and 'Cumulative Return (%)' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values(by='Date')
                    df['Days Since Start'] = (df['Date'] - df['Date'].min()).dt.days
                    df['Agent Type'] = df['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))
                    avg_returns = df.groupby(['Days Since Start', 'Agent Type'])[
                        'Cumulative Return (%)'].mean().reset_index()

                    global_y_min = min(global_y_min, avg_returns['Cumulative Return (%)'].min())
                    global_y_max = max(global_y_max, avg_returns['Cumulative Return (%)'].max())

                    unique_agent_types = avg_returns['Agent Type'].unique()
                    agent_types.update(unique_agent_types)

                    for agent_type in unique_agent_types:
                        if agent_type not in agent_color_map:
                            agent_color_map[agent_type] = None

            # Assign distinct colors to each agent type, Reinforcement Learning agent gets strong black
            colors = plt.cm.get_cmap("tab10", len(agent_types)).colors
            for idx, agent in enumerate(sorted(agent_types)):
                if "Reinforcement" in agent:
                    agent_color_map[agent] = "black"
                else:
                    agent_color_map[agent] = colors[idx]

            for i, file in enumerate(input_files):
                df = pd.read_csv(file)
                episode_num = os.path.basename(file).split('_')[-1].split('.')[0]

                if 'Date' in df.columns and 'Agent' in df.columns and 'Cumulative Return (%)' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values(by='Date')
                    df['Days Since Start'] = (df['Date'] - df['Date'].min()).dt.days
                    df['Agent Type'] = df['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))
                    avg_returns = df.groupby(['Days Since Start', 'Agent Type'])[
                        'Cumulative Return (%)'].mean().reset_index()

                    for agent_type in avg_returns['Agent Type'].unique():
                        agent_data = avg_returns[avg_returns['Agent Type'] == agent_type]
                        axes[i].plot(
                            agent_data['Days Since Start'],
                            agent_data['Cumulative Return (%)'],
                            label=agent_type if i == 0 else "",  # Only label once
                            color=agent_color_map[agent_type],
                            alpha=0.7,
                            linestyle="solid"
                        )

                    axes[i].set_title(f'Episode {episode_num}')
                    axes[i].grid(True)
                    axes[i].set_xticks(np.linspace(0, df['Days Since Start'].max(), num=5))

            for ax in axes:
                ax.set_yticks(np.linspace(global_y_min, global_y_max, num=5))
                ax.set_ylim(global_y_min, global_y_max)
            legend_handles = [plt.Line2D([0], [0], color=agent_color_map[agent],
                                         lw=3 if "Reinforcement" in agent else 2, label=agent) for
                              agent in sorted(agent_types)]
            fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=15)

            plt.suptitle("Cumulative Return Trend Across Episodes", fontsize=25, y=0.95)
            plt.tight_layout(rect=[0, 0, 1, 0.895])
            plt.show()

    @staticmethod
    def plot_reward_trend_per_episode(input_files):
        """
        Plots the reward trend for each episode over steps.

        :param input_files: List of reward history files for the DQN model which has calculated the reward for each step.
        """
        plt.figure(figsize=(12, 6))

        # Load and plot reward trends for each episode
        for file in input_files:
            df = pd.read_csv(file)
            episode_num = os.path.basename(file).split('_')[1].split('.')[0]  # Extract episode number

            if 'Step' in df.columns and 'Reward' in df.columns:
                df = df.sort_values(by='Step')
                plt.plot(df['Step'], df['Reward'], label=f'Episode {episode_num}')

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward Trend Over Steps for Each Episode")
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8)
        plt.grid(True)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_single_trade_history(input_file):
        """
        Plots:
        1️. The cumulative return trend of a single trade history file.
        2️. The stock price trends over time, using "Days Since Start".

        :param input_file: Path to a single trade history CSV file.
        """
        df = pd.read_csv(input_file)

        required_columns = {'Date', 'Agent', 'Cumulative Return (%)', 'Ticker', 'Stock Price'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns in {input_file}")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df['Days Since Start'] = (df['Date'] - df['Date'].min()).dt.days

        # Extract agent types from their names
        df['Agent Type'] = df['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))
        # Compute average cumulative return per agent type
        avg_returns = df.groupby(['Days Since Start', 'Agent Type'])['Cumulative Return (%)'].mean().reset_index()

        # Assign colors for agent types
        agent_types = sorted(avg_returns['Agent Type'].unique())
        colors = plt.get_cmap("tab10", len(agent_types)).colors
        agent_color_map = {agent: colors[i] for i, agent in enumerate(agent_types)}

        # Ensure "Reinforcement" agent is always BLACK
        for agent in agent_types:
            if "Reinforcement" in agent:
                agent_color_map[agent] = "black"

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        for agent in agent_types:
            agent_data = avg_returns[avg_returns['Agent Type'] == agent]
            axes[0].plot(agent_data['Days Since Start'], agent_data['Cumulative Return (%)'],
                         label=agent, color=agent_color_map[agent], alpha=0.8, linestyle="solid")

        axes[0].set_ylabel("Cumulative Return (%)")
        axes[0].set_title(f"Trade History - {os.path.basename(input_file)}")
        axes[0].legend(loc="upper left", fontsize=10)
        axes[0].grid(True)

        stock_prices = df.groupby(['Days Since Start', 'Ticker'])['Stock Price'].mean().reset_index()

        for ticker in stock_prices['Ticker'].unique():
            ticker_data = stock_prices[stock_prices['Ticker'] == ticker]
            axes[1].plot(ticker_data['Days Since Start'], ticker_data['Stock Price'], label=ticker)

        axes[1].set_ylabel("Stock Price")
        axes[1].set_title("Stock Price Trends")
        axes[1].legend(loc="upper left", fontsize=10)
        axes[1].grid(True)
        axes[1].set_xlabel("Days Since Start")
        plt.tight_layout()
        plt.show()


class EpisodeTracker:
    """
    Tracks episode performance for RL agent training and saves episode-wise reward data.
    This class is used to track and save episode-wise reward data for RL agents trained with DQN.
    """
    def __init__(self, patience=20, min_delta=0.05, save_directory="Trade_History/Episode_Track"):
        self.episode_rewards = []  # Total reward per episode
        self.episode_net_worths = []  # Final net worth per episode

        self.episode_reward_history = []  # Stores per-step rewards for each episode
        self.episode_net_worth_history = []  # Stores per-step net worth per episode

        self.current_episode_reward = 0
        self.current_episode_rewards = []  # Stores per-step rewards for current episode
        self.current_episode_net_worths = []  # Stores per-step net worth for current episode

        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -np.inf
        self.wait = 0

        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

    def track_step(self, reward, done, env):
        """Track reward and net worth at each step."""
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[-1]) if isinstance(reward, (list, np.ndarray)) else float(reward) # Take the first element and ensure it’s a float
        else:
            reward = float(reward)
        self.current_episode_reward += reward
        self.current_episode_rewards.append(reward)  # Step-wise reward tracking
        if done and env.final_net_worth is not None:
            self.current_episode_net_worths.append(env.final_net_worth)
            self.track_episode_end(env)
        else:
            self.current_episode_net_worths.append(env.model.rl_agent.get_net_worth())

    def track_episode_end(self, env):
        """Track final episode values and reset trackers."""
        episode_number = len(self.episode_rewards) + 1  # Track current episode number

        # Save episode-level rewards
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_net_worths.append(env.model.rl_agent.get_net_worth())
        # Store per-step episode rewards
        self.episode_reward_history.append(self.current_episode_rewards)
        self.episode_net_worth_history.append(self.current_episode_net_worths)

        # === Save Episode Rewards & Net Worth to CSV === #
        episode_df = pd.DataFrame({
            "Step": range(1, len(self.current_episode_rewards) + 1),
            "Reward": self.current_episode_rewards,
            "Net Worth": self.current_episode_net_worths
        })

        save_path = os.path.join(self.save_directory, f"episode_{episode_number}.csv")
        episode_df.to_csv(save_path, index=False)
        print(f"✅ Episode {episode_number} rewards saved to {save_path}")
        # Reset for next episode
        self.current_episode_rewards = []
        self.current_episode_net_worths = []
        self.current_episode_reward = 0

    def should_stop_early(self):
        """Implements early stopping if performance stops improving."""
        if len(self.episode_rewards) < self.patience:
            return False  # Not enough data yet

        avg_recent_rewards = np.mean(self.episode_rewards[-self.patience:])
        if avg_recent_rewards - self.best_reward > self.min_delta:
            self.best_reward = avg_recent_rewards
            self.wait = 0  # Reset patience counter
        else:
            self.wait += 1

        return self.wait >= self.patience  # Stop training if patience exceeded