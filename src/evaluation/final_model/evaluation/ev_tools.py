import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


class EvaluationTools:

    @staticmethod
    def plot_stocks_trades_summary(stock_df, trade_df):
        """
        This contains 2 sub-plot for showing the simulation history with the stock database
        :param stock_df: Stock Dataframe
        :param trade_df: Trade history
        :return: graphs showing the stock open prices move AND agent cumulative return trends
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        # Plot stock prices on the first subplot
        stocks = ['AAPL', 'META', 'XOM', 'V']
        for stock in stocks:
            if ('Open', stock) in stock_df.columns:
                axes[0].plot(stock_df.index, stock_df[('Open', stock)], label=stock)
            else:
                print(f"Warning: No data found for {stock}")
        axes[0].set_title("Stock Price Movement")
        axes[0].set_ylabel("Open Price")
        axes[0].legend()
        axes[0].grid(True)

        # Ensure necessary columns exist in trade history
        if 'Date' in trade_df.columns and 'Agent' in trade_df.columns and 'Cumulative Return (%)' in trade_df.columns:
            trade_df['Date'] = pd.to_datetime(trade_df['Date'])
            trade_df = trade_df.sort_values(by='Date')
            trade_df['Agent Type'] = trade_df['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))

            # Compute average cumulative return per agent type over time
            avg_returns = trade_df.groupby(['Date', 'Agent Type'])['Cumulative Return (%)'].mean().reset_index()

            # Determine unique agent types
            agent_types = avg_returns['Agent Type'].unique()

            # Assign colors, RL agent in black
            colors = plt.get_cmap("tab10", len(agent_types)).colors
            agent_color_map = {agent: colors[idx] for idx, agent in enumerate(sorted(agent_types))}
            agent_color_map = {agent: "black" if "Reinforcement" in agent else agent_color_map[agent] for agent in
                               agent_types}

            # Plot cumulative return trends on the second subplot
            for agent_type in agent_types:
                agent_data = avg_returns[avg_returns['Agent Type'] == agent_type]
                axes[1].plot(
                    agent_data['Date'],
                    agent_data['Cumulative Return (%)'],
                    label=agent_type,
                    color=agent_color_map[agent_type],
                    alpha=0.8,
                    linestyle="solid",
                    linewidth=2 if "Reinforcement" in agent_type else 1.5
                )
            axes[1].set_title("Cumulative Return Trend by Agent Type")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Cumulative Return (%)")
            axes[1].legend()
            axes[1].grid(True)
        else:
            print("Error: Required columns not found in the trade history CSV file.")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_stock_price_trends(stock_df, tickers=None):
        """
        Plots the stock price trends over the simulation period.

        :param stock_df: Multi-index DataFrame with stock prices (columns like ('Open', 'AAPL')).
        :param tickers: List of stock tickers to plot (default: AAPL, META, XOM, V).
        :return: Line chart showing the Open prices over time for selected stocks.
        """
        if tickers is None:
            tickers = ['AAPL', 'META', 'XOM', 'V']

        fig, ax = plt.subplots(figsize=(14, 6))
        for ticker in tickers:
            if ('Open', ticker) in stock_df.columns:
                ax.plot(stock_df.index, stock_df[('Open', ticker)], label=ticker)
            else:
                print(f"[Warning] Ticker '{ticker}' not found in stock_df")

        ax.set_title("Stock Open Price Trends Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Open Price")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_trade_action_volume(trade_df):
        """
        :param trade_df: Trade history
        :return: Plots a stacked bar chart where the x-axis is the date, and the y-axis represents the volume
                 of each trade action (Buy, Hold, Sell, Invalid).
        """
        if 'Date' not in trade_df.columns or 'Action' not in trade_df.columns:
            raise ValueError("Error: Required columns (Date, Action) not found in the trade history file.")

        trade_df['Date'] = pd.to_datetime(trade_df['Date'])
        trade_df = trade_df.sort_values(by='Date')

        # Count occurrences of each action per day
        trade_counts = trade_df.groupby(['Date', 'Action']).size().unstack(fill_value=0)

        # Define action colors
        action_colors = {"buy": "green", "hold": "yellow", "invalid": "blue", "sell": "red"}

        # Plot stacked bar chart
        ax = trade_counts.plot(kind='bar', stacked=True, figsize=(12, 6),
                               color=[action_colors[action] for action in trade_counts.columns])

        plt.title("Trade Action Volume Per Day")
        plt.xlabel("Date")
        plt.ylabel("Number of Trades")
        plt.legend(title="Trade Action")

        # Adjust x-axis labels to show one label every 3 months
        date_labels = trade_counts.index.strftime('%Y-%m')
        ax.set_xticks(np.arange(0, len(date_labels), len(date_labels) // 12 * 3))  # Approx every 3 months
        ax.set_xticklabels(date_labels[::len(date_labels) // 12 * 3], rotation=45)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_cumulative_return_with_trade_actions(trade_df):
        """
        :param trade_df: Trade history
        :return: Plot cumulative return graph with trade action volumes as background for all agents.
                 if agent types has more than 1 then it is the average of that type
        """
        required_columns = ['Date', 'Agent', 'Cumulative Return (%)', 'Action']
        if not all(col in trade_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in trade_df.columns]
            print(f"Error: Required columns missing in DataFrame: {missing}")
            return

        # Prepare DataFrame
        trade_df['Date'] = pd.to_datetime(trade_df['Date'])
        trade_df = trade_df.sort_values('Date')
        trade_df['Agent Type'] = trade_df['Agent'].apply(lambda x: "_".join(x.split("_")[:-1]))
        avg_returns = trade_df.groupby(['Date', 'Agent Type'])['Cumulative Return (%)'].mean().reset_index()
        agent_types = avg_returns['Agent Type'].unique()

        # Assign colors
        colors = plt.get_cmap("tab10", len(agent_types)).colors
        agent_color_map = {
            agent: "black" if "Reinforcement" in agent else colors[idx]
            for idx, agent in enumerate(sorted(agent_types))
        }

        # Count trade actions per date
        trade_counts = trade_df.groupby(['Date', 'Action']).size().unstack(fill_value=0)
        action_colors = {"buy": "green", "hold": "yellow", "invalid": "blue", "sell": "red"}

        # Create figure and axes
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = ax1.twinx()

        # Plot trade action volumes (bar chart in background)
        bar_width = 0.7
        dates = trade_counts.index
        bottom = np.zeros(len(dates))
        for action in ["buy", "hold", "invalid", "sell"]:
            if action in trade_counts:
                ax2.bar(
                    dates,
                    trade_counts[action],
                    bottom=bottom,
                    width=bar_width,
                    color=action_colors[action],
                    label=f"{action.capitalize()} volume",
                    alpha=0.2,
                    zorder=1
                )
                bottom += trade_counts[action].values

        ax2.set_ylabel("Trade Action Volume")
        ax2.grid(False)
        ax2.set_zorder(1)  # Background layer

        # Plot cumulative returns (lines on top)
        for agent_type in agent_types:
            agent_data = avg_returns[avg_returns['Agent Type'] == agent_type]
            ax1.plot(
                agent_data['Date'],
                agent_data['Cumulative Return (%)'],
                label=agent_type,
                color=agent_color_map[agent_type],
                linewidth=2 if "Reinforcement" in agent_type else 1,
                zorder=2
            )

        # Set x-axis precisely from first to last date (no gaps)
        ax1.set_xlim([dates.min(), dates.max()])

        # Set x-axis ticks (approx. every quarter)
        date_labels = dates.strftime('%Y-%m')
        step = max(1, len(date_labels) // 12 * 3)
        ax1.set_xticks(dates[::step])
        ax1.set_xticklabels(date_labels[::step], rotation=45)

        # Axis labels and grid
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.grid(True)

        # Move legend outside the plot clearly
        ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 0.7), borderaxespad=0)

        # Title and layout adjustment
        ax1.set_title("Cumulative Return with Trade Action Volume Background")
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend space outside plot
        plt.show()

    @staticmethod
    def summarize_rl_trades_actions(trade_df):
        """
        :param trade_df: Trade history
        :return: Summarizes total trades and breakdown of Buy/Sell/Hold/Invalid for each stock by the RL agent.
        """
        df = trade_df

        if 'Agent' not in df.columns or 'Action' not in df.columns or 'Ticker' not in df.columns:
            raise ValueError("Error: Required columns (Agent, Action, Ticker) not found in the trade history file.")

        # Filter for RL agent only
        rl_df = df[df['Agent'].str.contains("Reinforcement", case=False, na=False)]

        # Count actions per stock, including 'invalid'
        action_counts = rl_df.groupby(['Ticker', 'Action']).size().unstack(fill_value=0)

        # Calculate total trades correctly as row sum
        action_counts['Total Trades'] = action_counts.sum(axis=1)

        # Ensure all action types exist as columns
        for action_type in ['buy', 'sell', 'hold', 'invalid buy', 'invalid sell']:
            if action_type not in action_counts.columns:
                action_counts[action_type] = 0

        # Reorder columns
        summary_df = action_counts[['buy', 'sell', 'hold', 'invalid buy', 'invalid sell', 'Total Trades']].astype(int)

        return summary_df

    @staticmethod
    def plot_rl_actions(trade_df):
        """
        :param trade_df: Trade history
        :return: Plot number of Buy/Sell/Hold/Invalid for each stock by the RL agent.
        """
        rl_df = trade_df[trade_df['Agent'].str.contains("Reinforcement", case=False, na=False)]

        action_counts = rl_df['Action'].value_counts()
        action_counts = action_counts.reindex(['buy', 'sell', 'hold', 'invalid buy', 'invalid sell'], fill_value=0)
        color_map = {
            'buy': 'green',
            'sell': 'red',
            'hold': 'gray',
            'invalid buy': 'orange',
            'invalid sell': 'orange'
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        action_counts.plot(kind='bar', color=[color_map[a] for a in action_counts.index], ax=ax)
        ax.set_title("RL Agent Action Breakdown by Category")
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        for i, v in enumerate(action_counts):
            ax.text(i, v + 1, str(v), ha='center', fontsize=10)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rl_price_actions_overlay(trade_df):
        """
        Plots stock price lines for each ticker with RL agent actions overlaid.
        Actions:
            - Buy: green circle
            - Sell: red cross
            - Hold: grey dot
            - Invalids: orange diamond
        """
        required_cols = {'Date', 'Agent', 'Action', 'Ticker', 'Stock Price'}
        if not required_cols.issubset(trade_df.columns):
            raise ValueError(f"Missing columns: {required_cols - set(trade_df.columns)}")

        rl_df = trade_df[trade_df['Agent'].str.contains("Reinforcement", case=False, na=False)].copy()
        rl_df['Date'] = pd.to_datetime(rl_df['Date'])

        tickers = rl_df['Ticker'].unique()
        n_tickers = len(tickers)

        fig, axes = plt.subplots(n_tickers, 1, figsize=(14, 4 * n_tickers), sharex=True)
        if n_tickers == 1:
            axes = [axes]

        for ax, ticker in zip(axes, tickers):
            df = rl_df[rl_df['Ticker'] == ticker].sort_values('Date')
            ax.plot(df['Date'], df['Stock Price'], label=f"{ticker} Price", color='blue', linewidth=1.5)

            action_style = {
                'buy': {'color': 'green', 'marker': 'o', 'label': 'Buy'},
                'sell': {'color': 'red', 'marker': 'x', 'label': 'Sell'},
                'hold': {'color': 'blue', 'marker': 's', 'label': 'Hold'},
                'invalid buy': {'color': '#90ee90', 'marker': 'd', 'label': 'Invalid Buy'},
                'invalid sell': {'color': '#ff9999', 'marker': 'd', 'label': 'Invalid Sell'},
            }

            for action, style in action_style.items():
                points = df[df['Action'] == action]
                if not points.empty:
                    ax.scatter(
                        points['Date'], points['Stock Price'],
                        color=style['color'], marker=style['marker'],
                        s=60, label=style['label'],
                    )

            ax.set_title(f"{ticker} - RL Agent Trades on Price Line")
            ax.set_ylabel("Stock Price")
            ax.legend(loc='upper left')
            ax.grid(True)

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def summarize_agent_stats(file_dict):
        """
        file_dict: dict where keys are suffixes (e.g., '33333') and values are file paths.
        Returns a transposed dataframe where columns are AgentType-Suffix and rows are metrics.
        """

        def extract_agent_class(agent_id):
            return "_".join(agent_id.split("_")[:-1]) if "_" in agent_id else agent_id

        columns_to_aggregate = [
            'Portfolio Value', 'Net Worth',
            'Daily MW Return Rate(%)', 'Monthly MW Return Rate (%)',
            'Quarter MW Return Rate (%)', 'Yearly MW Return Rate (%)',
            'Cumulative Return (%)', 'Realized Profit/Loss', 'Unrealized Profit/Loss',
            'Win Rate (%)', 'Total Trades', 'Profitability Score', 'Trade Volatility',
            'Performance Score'
        ]

        all_summaries = []

        for suffix, file_path in file_dict.items():
            df = pd.read_csv(file_path)
            if 'Agent' not in df.columns:
                continue

            df['Agent Type'] = df['Agent'].apply(extract_agent_class)

            for agent_type, group in df.groupby('Agent Type'):
                full_agent_name = f"{agent_type}-{suffix}"
                summary = {'Agent Type': full_agent_name}

                for col in columns_to_aggregate:
                    if col in group.columns:
                        values = group[col].dropna()
                        if not values.empty:
                            mean = values.mean()
                            std_dev = values.std()
                            summary[col] = f"{mean:.2f} +/- {std_dev:.2f}"
                        else:
                            summary[col] = "N/A"
                    else:
                        summary[col] = "N/A"

                all_summaries.append(summary)

        # Create wide-format summary
        wide_df = pd.DataFrame(all_summaries)

        # Sort: ReinforcementAgent first
        wide_df['SortOrder'] = wide_df['Agent Type'].apply(lambda x: 0 if 'ReinforcementAgent' in x else 1)
        wide_df = wide_df.sort_values(by=['SortOrder', 'Agent Type']).drop(columns=['SortOrder'])

        # Transpose for comparison
        transposed_df = wide_df.set_index('Agent Type').T

        return transposed_df

