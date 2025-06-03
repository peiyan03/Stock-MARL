import agentpy as ap
import pandas as pd
import numpy as np
import copy
import random
import numpy_financial as npf


class AgentBase(ap.Agent):
    """
    The StockAgentBase class serves as the foundational trading agent within the AgentPy simulation framework.
    It encapsulates essential financial operations, portfolio management, and historical performance tracking.
    Each agent maintains detailed records of trading activities, including realized and unrealized profit/loss,
    cash flows, portfolio valuations, and various performance metrics such as Money-Weighted Rate of Return (MWRR).
    """

    def setup(self, agent_seed=None, agent_id=None):
        """
        Initializes an agent with financial data, portfolio, and historical tracking.
            :param agent_seed: Agent's General Seed created from the simulation_model.py during initialization
            :param agent_id: Agent's Unique ID
        """
        self.budget = 100000
        self.current_day = None
        self.stock_data = self.model.stock_data

        # Ticker list to avoid redundant fetching
        self.ticker_list = list(self.stock_data.columns.get_level_values(1).unique())

        # Financial metrics tracking
        self.portfolio = {}
        self.net_worth_history = []
        self.cash_flow_history = []

        self.daily_mwrr = 0.0
        self.monthly_mwrr = 0.0
        self.quarter_mwrr = 0.0
        self.yearly_mwrr = 0.0
        self.cumulative_return = 0.0

        self.realized_PL = 0.0
        self.unrealized_PL = 0.0
        self.portfolio_value = 0.0

        # Trade history and performance metrics
        self.last_buy_price = {}
        self.trade_history = []  # Each Agent's Trade Log History
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        self.total_trades = 0
        self.invalid_action_count = 0

        self.profitability_score = 0.0
        self.trade_volatility = 0.0
        self.performance_score = 0.0

        # Seed and ID management for reproducibility
        self.agent_seed = int(agent_seed) if agent_seed else self.model.rng.integers(0, 10 ** 6)
        self.agent_id = agent_id if agent_id else f"{self.__class__.__name__}_{id(self)}"

        # Unique random generator per agent, one is numpy one is python default
        self.random_generator = np.random.default_rng(self.agent_seed)
        self.py_random_generator = random.Random(self.agent_seed)

        # Model-level logging
        if not hasattr(self.model, "history"):
            self.model.history = []

    def update_current_day(self):
        """Update the current simulation day from the model."""
        if self.model.t < len(self.stock_data.index):
            self.current_day = self.stock_data.index[self.model.t]
        else:
            raise ValueError("Reach the end of available stock data")

    # ------------------- Trading Actions ------------------------
    def get_stock_price(self, ticker, price_type='Open'):
        """Fetches the specified stock price safely. Now only look at the Open Price"""
        try:
            return self.stock_data.loc[self.current_day, (price_type, ticker)]
        except KeyError:
            return None

    def buy(self, ticker, price, quantity=1):
        """Executes a buy transaction and updates the portfolio."""
        total_cost = price * quantity
        if self.budget >= total_cost:
            self.budget -= total_cost
            self.total_trades += 1
            if ticker not in self.portfolio:
                self.portfolio[ticker] = {'quantity': 0, 'average_price': 0}

            # Compute new average price using weighted average formula
            prev_quantity = self.portfolio[ticker]['quantity']
            prev_avg_price = self.portfolio[ticker]['average_price']
            new_quantity = prev_quantity + quantity
            if new_quantity > 0:
                new_avg_price = ((prev_avg_price * prev_quantity) + (price * quantity)) / new_quantity
            else:
                new_avg_price = price

            self.portfolio[ticker]['quantity'] = new_quantity
            self.portfolio[ticker]['average_price'] = float(new_avg_price)
            self.cash_flow_history.append(float(-total_cost))

            print(f"{self.agent_id} bought {quantity} shares of {ticker} at {price}")
            self.log_trade(1, ticker, price, quantity)
        else:
            print(f"{self.agent_id} insufficient budget to buy {ticker}")
            self.invalid_action_count += 1
            self.log_trade(-5, ticker, price, quantity=0)

    def sell(self, ticker, price, quantity=1):
        """Sells specified quantity of shares and updates financial records."""
        if ticker in self.portfolio and self.portfolio[ticker]['quantity'] >= quantity:
            total_gain = price * quantity
            buy_price = self.portfolio[ticker]['average_price']
            profit_loss = (price - buy_price) * quantity

            self.total_trades += 1
            self.budget += total_gain
            self.portfolio[ticker]['quantity'] -= quantity
            if self.portfolio[ticker]['quantity'] == 0:
                del self.portfolio[ticker]

            # Update Wins and Losses only when sell action occurs
            if profit_loss > 0:
                self.wins += 1
            elif profit_loss < 0:
                self.losses += 1

            self.realized_PL += profit_loss

            self.cash_flow_history.append(float(total_gain))

            print(f"{self.agent_id} sold {quantity} shares of {ticker} at {price}")
            self.log_trade(-1, ticker, price, quantity)
        else:
            print(f"{self.agent_id} insufficient portfolio to sell {ticker}")
            self.invalid_action_count += 1
            self.log_trade(-6, ticker, price, quantity=0)

    def hold(self, ticker, price):
        """Records a hold action without altering financial state."""
        print(f"{self.agent_id} held {ticker} at price {price}")
        self.cash_flow_history.append(0.0)

        self.log_trade(0, ticker, price, quantity=0)

    # ---------------- Historical Data and Metrics -------------------
    def update_financials(self):
        """Ensures portfolio value, net worth, and returns are updated at every step."""
        self.portfolio_value = self.get_portfolio_value()
        self.net_worth_history.append(float(self.get_net_worth()))
        self.update_win_rate()
        self.update_unrealized_pl()
        self.update_MWRR()
        self.update_cumulative_return()
        self.update_profitability_score()
        self.update_trade_volatility()
        self.update_performance_score()

    def update_MWRR(self):
        """Updates Money-Weighted Rate of Return (MWRR) over various intervals."""
        self.daily_mwrr = self.compute_MWRR(2)
        self.monthly_mwrr = self.compute_MWRR(21)
        self.quarter_mwrr = self.compute_MWRR(63)
        self.yearly_mwrr = self.compute_MWRR(252)

    def compute_MWRR(self, period):
        """Computes a weighted Money-Weighted Rate of Return (MWRR), ensuring portfolio value changes are considered even if no trades occur."""

        if len(self.cash_flow_history) < period:
            if self.daily_mwrr != 0.0:
                if period == 21:
                    self.monthly_mwrr = self.daily_mwrr
                if period == 63:
                    self.quarter_mwrr = self.monthly_mwrr
                if period == 252:
                    self.yearly_mwrr = self.quarter_mwrr
            else:
                return 0.0

        relevant_cash_flows = self.cash_flow_history[-period:]

        # === Ensure starting portfolio value is valid ===
        beginning_value = self.net_worth_history[0]

        # === Compute net portfolio value change ===
        ending_value = self.get_net_worth()

        # Compute time-weighted contribution from deposits/withdrawals
        total_days = period
        weighted_cash_flows = []
        valid_cash_flows = [cf for cf in relevant_cash_flows if cf != 0]
        max_cash_flow = max(abs(cf) for cf in valid_cash_flows) if valid_cash_flows else 1  # Avoid division by zero

        for idx, amount in enumerate(relevant_cash_flows):
            if amount != 0:  # Only include non-zero cash flows
                time_weight = (total_days - idx) / total_days  # Older cash flows get larger weights
                size_weight = abs(amount) / max_cash_flow  # Normalize by max cash flow
                adjusted_cash_flow = amount * time_weight * size_weight  # Apply both adjustments
                weighted_cash_flows.append(adjusted_cash_flow)

        # Compute final return using portfolio value and weighted cash flow adjustments
        mwrr = (sum(weighted_cash_flows) + ending_value - beginning_value) / beginning_value
        return mwrr * 100  # Convert to percentage

    def update_unrealized_pl(self):
        """Updates the unrealized profit and loss based on current stock prices."""
        self.unrealized_PL = sum(
            (self.get_stock_price(ticker) - self.portfolio[ticker]['average_price'])
            * self.portfolio[ticker]['quantity'] for ticker in self.portfolio
        )

    def update_win_rate(self):
        """Updates the agent's win rate."""
        if self.total_trades > 0:
            self.win_rate = (self.wins / self.total_trades) * 100
        else:
            self.win_rate = -1

    def get_portfolio_value(self):
        """Returns current market value of portfolio holdings."""
        return sum(
            self.portfolio[ticker]['quantity'] * self.get_stock_price(ticker)
            for ticker in self.portfolio if self.portfolio[ticker]['quantity'] > 0
        )

    def get_net_worth(self):
        """Calculates net worth based on current portfolio value and budget."""
        return self.budget + self.get_portfolio_value()

    def update_cumulative_return(self):
        """Updates the cumulative return."""
        if not self.net_worth_history or self.net_worth_history[0] == 0:
            self.cumulative_return = 0.0
            return
        self.cumulative_return = 100*((self.get_net_worth() - self.net_worth_history[0]) / self.net_worth_history[0] )

    def update_profitability_score(self, window=7):
        """Computes a stable and reliable profitability score by handling extreme values."""

        recent_trades = self.trade_history[-window:]  # Use last 'window' trades
        if not recent_trades:
            self.profitability_score = 0.0
            return

        # Step 1: Compute Profitability Metrics
        profits = np.array([trade["Realized Profit/Loss"] for trade in recent_trades])
        avg_profit = np.mean(profits)

        std_profit = np.std(profits)
        std_profit = max(std_profit, 1)  # Prevent division by near-zero (1 as minimum standard deviation)

        # Step 2: Compute Exponential Moving Average (EMA) on Profit
        alpha = 0.2  # Smoothing factor (0.1 = slower, 0.3 = faster)
        if not hasattr(self, 'ema_profit'):
            self.ema_profit = avg_profit

        self.ema_profit = (alpha * avg_profit) + ((1 - alpha) * self.ema_profit)

        # Step 3: Compute Risk-Adjusted Profitability (Safe)
        risk_adjusted_profit = self.ema_profit / std_profit  # Now bounded
        risk_adjusted_profit = max(min(risk_adjusted_profit, 10), -10)  # Clip extreme values to [-10, 10]

        # Step 4: Compute Win Rate (Smoothing for Small Samples)
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades) if total_trades > 0 else 0

        # Smooth win rate to avoid extreme fluctuations
        win_rate = (win_rate * total_trades) / (total_trades + 10)  # Weighted to avoid extreme jumps for few trades

        # Step 5: Compute Cumulative Return (Safe)
        initial_net_worth = self.net_worth_history[0] if self.net_worth_history else self.budget
        final_net_worth = self.get_net_worth()

        if initial_net_worth > 0:
            cumulative_return = (final_net_worth - initial_net_worth) / initial_net_worth
        else:
            cumulative_return = 0

        # Clip cumulative return to avoid extreme outliers
        cumulative_return = max(min(cumulative_return, 10), -10)

        # Step 6: Compute Net Worth Ratio (Safe)
        avg_market_net_worth = np.mean([agent.get_net_worth() for agent in self.model.agents])
        avg_market_net_worth = max(avg_market_net_worth, 100)  # Prevent division by near-zero

        net_worth_ratio = final_net_worth / avg_market_net_worth
        net_worth_ratio = max(min(net_worth_ratio, 5), -5)  # Cap extreme values

        # Step 7: Compute Final Profitability Score (Safe)
        self.profitability_score = (
                (0.5 * risk_adjusted_profit) +  # Short-term profitability
                (0.2 * win_rate) +  # Trading consistency
                (0.2 * cumulative_return) +  # Long-term profitability
                (0.1 * net_worth_ratio)  # Portfolio growth
        )

        # Ensure final score is bounded
        self.profitability_score = max(min(self.profitability_score, 10), -10)

    def update_trade_volatility(self, window=7):
        """Computes a more robust trade volatility score based on frequency & trade volume."""
        recent_trades = self.trade_history[-window:]  # Consider last 'window' trades
        if len(recent_trades) < 2:
            self.trade_volatility = 0.0  # Not enough trades to calculate volatility
            return

        # === Step 1: Compute Trade Frequency Volatility ===
        trade_counts = np.array([1 if trade["Action"] in ["buy", "sell"] else 0 for trade in recent_trades])
        trade_frequency_std = np.std(trade_counts)  # Standard deviation of trade frequency

        # === Step 2: Compute Trade Size Volatility ===
        trade_sizes = np.array([trade["Quantity"] for trade in recent_trades if trade["Action"] in ["buy", "sell"]])
        trade_size_std = np.std(trade_sizes) if len(trade_sizes) > 1 else 0  # Avoid division by zero

        # Use Coefficient of Variation (CV) to adjust for small trade counts
        trade_size_mean = np.mean(trade_sizes) if len(trade_sizes) > 0 else 1e-6
        trade_size_cv = trade_size_std / (trade_size_mean + 1e-6)  # Normalize by mean trade size

        # === Step 3: Apply Exponential Moving Average (EMA) for Stability ===
        alpha = 0.2  # Adjust smoothing factor (0.1 = slower, 0.3 = faster)
        if not hasattr(self, 'ema_trade_volatility'):  # Initialize if not present
            self.ema_trade_volatility = trade_frequency_std

        self.ema_trade_volatility = (alpha * trade_frequency_std) + ((1 - alpha) * self.ema_trade_volatility)

        # === Step 4: Normalize Against Market Activity ===
        trade_counts = [
            len(agent.trade_history[-window:])
            for agent in self.model.agents if len(agent.trade_history) > window
        ]

        avg_market_trades = np.mean(trade_counts) if trade_counts else 0.0

        # Adjusted market normalization to prevent over-normalization
        market_trade_volatility = trade_frequency_std / (avg_market_trades + trade_frequency_std + 1e-6)

        # === Step 5: Compute Final Trade Volatility Score ===
        self.trade_volatility = (
                (0.5 * self.ema_trade_volatility) +  # EMA on trade frequency
                (0.3 * trade_size_cv) +  # Coefficient of Variation (normalized trade size volatility)
                (0.2 * market_trade_volatility)  # Compare vs. market-wide trade frequency
        )

    def update_performance_score(self):
        score = (0.5 * self.profitability_score) + (0.3 * self.cumulative_return) - (0.2 * self.trade_volatility)
        self.performance_score = score

    @staticmethod
    def calculate_volatility(prices, window):
        """Calculate rolling standard deviation as a measure of volatility."""
        return prices.pct_change().rolling(window=window).std().iloc[-1] * 100

    @staticmethod
    def moving_average(prices, window):
        """Compute moving average using Open prices only."""
        return prices.rolling(window=window).mean()

    def get_price_series(self, ticker):
        """Retrieve the stock's Open price series safely."""
        if ticker not in self.ticker_list:
            return pd.Series()
        return self.stock_data.xs(ticker, level=1, axis=1)['Open'].loc[:self.current_day].dropna()

    # --------------------- Logging -------------------------
    def log_trade(self, action, ticker, stock_price, quantity):
        """Logs trading activities to the agent and global simulation history. It's the update function of the agents """
        self.update_financials()
        trade_log = {
            'Date': self.current_day,
            'Agent': self.agent_id,
            'Action': "buy" if action == 1 else "sell" if action == -1 else "invalid buy" if action == -5 else "invalid sell" if action == -6 else "hold",
            'Ticker': ticker,
            'Stock Price': float(stock_price),
            'Quantity': quantity,
            'Budget': float(self.budget),
            'Portfolio': copy.deepcopy(self.portfolio),
            'Portfolio Value': float(self.portfolio_value),
            'Net Worth': float(self.get_net_worth()),
            'Daily MW Return Rate(%)': self.daily_mwrr,
            'Monthly MW Return Rate (%)': self.monthly_mwrr,
            'Quarter MW Return Rate (%)': self.quarter_mwrr,
            'Yearly MW Return Rate (%)': self.yearly_mwrr,
            'Cumulative Return (%)': self.cumulative_return,
            'Realized Profit/Loss': float(self.realized_PL),
            'Unrealized Profit/Loss': float(self.unrealized_PL),
            'Win Rate (%)': self.win_rate,
            'Total Trades': self.total_trades,
            'Profitability Score': self.profitability_score,
            'Trade Volatility': self.trade_volatility,
            'Performance Score': self.performance_score
        }
        self.trade_history.append(trade_log)
        self.model.history.append(trade_log)

    def get_trade_dataframe(self, final_entry_only=True):
        """
        Returns the agent's trade history as a DataFrame.

        :param final_entry_only: If True, returns only the last recorded trade (final state).
        :return: DataFrame containing trade records.
        """
        trade_df = pd.DataFrame(self.trade_history)

        if trade_df.empty:
            print(f"Warning: No trade history found for {self.agent_id}")
            return pd.DataFrame()

        return trade_df.iloc[[-1]] if final_entry_only else trade_df
