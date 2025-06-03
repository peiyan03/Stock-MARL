import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from AgentBase import AgentBase

"""
Reactive Agents file contains all Reactive Agents distinct design.
    - include the DQN-Controlled Agent here since they works together in the simulation
"""

class RandomBuyerAgent(AgentBase):
    def step(self):
        """Executes a random stock trading decision per stock per day."""
        self.update_current_day()
        if self.current_day is None:
            return
        for ticker in self.ticker_list:
            price = self.get_stock_price(ticker)
            if price is None:
                continue

            shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']
            max_buy_quantity = int(self.budget / price) if self.budget > price else 0

            # Default probabilities
            buy_probability = 0.4
            hold_probability = 0.2
            sell_probability = 0.4

            # Increase sell probability if budget falls below 15%
            if self.budget / (self.budget + self.get_portfolio_value()) < 0.15:
                sell_probability = 0.7
                buy_probability = 0.2  # Reduce buy probability
                hold_probability = 0.1

            #  Use weighted random selection to ensure buy/sell actions occur
            action = self.random_generator.choice([1, 0, -1], p=[buy_probability, hold_probability, sell_probability])

            if action == 1 and max_buy_quantity > 0:
                quantity = self.py_random_generator.randint(1, max_buy_quantity)
                self.buy(ticker, price, quantity)

            elif action == -1 and shares_owned > 0:
                quantity = self.py_random_generator.randint(1, shares_owned)
                self.sell(ticker, price, quantity)

            else:
                self.hold(ticker, price)


class DayTraderAgent(AgentBase):
    def setup(self, agent_seed=None, agent_id=None):
        """Assign individualized thresholds per agent using their unique seed."""
        super().setup(agent_seed, agent_id)
        # Unique profit and stop-loss thresholds per agent
        self.profit_threshold = self.random_generator.uniform(0.8, 3.0)
        self.stop_loss_threshold = -self.random_generator.uniform(0.8, 3.0)
        self.trade_sensitivity = self.random_generator.uniform(0.3, 1.5)

    def step(self):
        """
        Executes trades frequently based on short-term trends, momentum, and volatility.
        Focuses on short-term profits but may suffer long-term instability.
        Introduces randomness to simulate real-world uncertainty and unique agent behaviors.
        """
        self.update_current_day()
        if self.current_day is None:
            return

        for ticker in self.ticker_list:
            try:
                if self.model.t < 2:
                    stock_price = self.get_stock_price(ticker)
                    self.hold(ticker, stock_price)
                else:  # Random action to add some uncertainty in DayTrader
                    if self.py_random_generator.uniform(0, 1) < 0.05:
                        action = self.random_generator.choice(["buy", "sell", "hold"], p=[0.45, 0.45, 0.1])
                        stock_price = self.get_stock_price(ticker)
                        random_quantity = self.py_random_generator.randint(1, 10)

                        if action == "buy" and self.budget >= stock_price * random_quantity:
                            self.buy(ticker, stock_price, random_quantity)
                        elif action == "sell":
                            shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']
                            if shares_owned > 0:
                                self.sell(ticker, stock_price, min(random_quantity, shares_owned))
                        else:
                            self.hold(ticker, stock_price)
                        continue  # Skip normal decision-making

                    # Calculate financial indicators (momentum, volatility, trend)
                    indicators = self.calculate_indicators(ticker)
                    if indicators is None:
                        continue

                    momentum, volatility, uptrend, downtrend, current_open_price = indicators
                    shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']

                    # === Determine Buy and Sell Quantities Based on Price Movements === #
                    price_movement_factor = abs(momentum) / 5  # Scale quantity based on price movement
                    base_buy_quantity = max(1, int(price_movement_factor * (self.budget / current_open_price) * 0.5))
                    base_sell_quantity = max(1, int(price_movement_factor * shares_owned * 0.5))

                    # Introduce randomness: Scale trade size differently for buy and sell
                    buy_quantity = max(1, int(base_buy_quantity * self.py_random_generator.uniform(0.7, 1.5)))
                    sell_quantity = max(1, int(base_sell_quantity * self.py_random_generator.uniform(0.5, 1.2)))

                    # === More Sensitive Trading (Find Small Price Movements) === #
                    adjusted_momentum = momentum * self.trade_sensitivity

                    # === Trade Decision (Prioritizing Short-Term Gains) === #
                    avg_buy_price = self.portfolio[ticker]['average_price'] if ticker in self.portfolio else None
                    return_on_investment = (
                            (current_open_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price else 0

                    if avg_buy_price and return_on_investment >= self.profit_threshold:
                        self.sell(ticker, current_open_price, min(sell_quantity, shares_owned))
                    elif avg_buy_price and return_on_investment <= self.stop_loss_threshold:
                        self.sell(ticker, current_open_price, min(sell_quantity, shares_owned))  # Stop loss strategy
                    elif uptrend and shares_owned > 0 and adjusted_momentum > 0:
                        self.sell(ticker, current_open_price, min(sell_quantity, shares_owned))
                    elif downtrend and self.budget >= current_open_price * buy_quantity and adjusted_momentum < 0:
                        self.buy(ticker, current_open_price, buy_quantity)
                    else:
                        self.hold(ticker, current_open_price)
            except KeyError:
                print(f"Data for ticker {ticker} is not available on {self.current_day}")
                continue

    def calculate_indicators(self, ticker):
        """Calculates key financial indicators for trading decisions."""
        try:
            # Retrieve last 2 days' open prices
            prev_days = [self.stock_data.index[self.model.t - i] for i in range(1, 3)]
            prev_open_prices = [self.stock_data.loc[day, ('Open', ticker)] if day in self.stock_data.index else None for
                                day in prev_days]

            # Ensure all data is available
            if any(price is None or pd.isna(price) for price in prev_open_prices):
                return None

            # Get today's open price
            current_open_price = self.get_stock_price(ticker)
            if current_open_price is None or pd.isna(current_open_price):
                return None

            # === Financial Indicators === #
            momentum = (current_open_price - prev_open_prices[0]) / prev_open_prices[0] * 100  # % Change from yesterday
            volatility = abs(prev_open_prices[0] - prev_open_prices[1]) / prev_open_prices[
                1] * 100  # % Change between last 2 days

            # Trend Analysis
            uptrend = prev_open_prices[0] > prev_open_prices[1]  # Yesterday was higher
            downtrend = prev_open_prices[0] < prev_open_prices[1]  # Yesterday was lower

            return momentum, volatility, uptrend, downtrend, current_open_price

        except KeyError:
            return None


class MomentumTraderAgent(AgentBase):
    def setup(self, agent_seed=None, agent_id=None):
        """Initializes momentum-based thresholds with agent-specific randomness."""
        super().setup(agent_seed, agent_id)
        self.short_window = self.random_generator.integers(3, 7)
        self.long_window = self.random_generator.integers(15, 25)
        self.trade_interval = self.random_generator.integers(3, 7)
        self.profit_threshold = self.random_generator.uniform(3.0, 10.0)
        self.stop_loss_threshold = -self.random_generator.uniform(2.0, 4.0)
        self.momentum_sensitivity = self.random_generator.uniform(0.5, 1.5)
        self.hold_longer_threshold = self.random_generator.uniform(0.5, 1.5)

    def step(self):
        """Makes trading decisions each day based on momentum and price trends."""
        self.update_current_day()
        if self.current_day is None or self.model.t < self.long_window:
            for ticker in self.ticker_list:
                self.hold(ticker, self.get_stock_price(ticker))
            return

        for ticker in self.ticker_list:
            try:
                stock_price = self.get_stock_price(ticker)
                if stock_price is None or pd.isna(stock_price):
                    continue

                action, buy_qty, sell_qty = self.decide_action(ticker, stock_price)

                if action == "buy" and self.budget >= stock_price * buy_qty:
                    self.buy(ticker, stock_price, buy_qty)

                elif action == "sell" and self.portfolio.get(ticker, {'quantity': 0})['quantity'] >= sell_qty:
                    self.sell(ticker, stock_price, sell_qty)

                else:
                    self.hold(ticker, stock_price)

            except KeyError:
                print(f"[MomentumTrader] Data missing for {ticker} on {self.current_day}")
                continue

    def decide_action(self, ticker, stock_price):
        """Determines trading action based on MA crossover and ROI thresholds."""
        stock_data = self.get_price_series(ticker)
        if len(stock_data) < self.long_window:
            return "hold", 0, 0

        short_ma = self.moving_average(stock_data, self.short_window).iloc[-1]
        long_ma = self.moving_average(stock_data, self.long_window).iloc[-1]

        if short_ma is None or long_ma is None:
            return "hold", 0, 0

        momentum_strength = (short_ma - long_ma) / (long_ma + 1e-6) * 100
        shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']
        avg_buy_price = self.portfolio.get(ticker, {}).get('average_price', None)
        buy_qty, sell_qty = self.determine_trade_quantity(ticker, stock_price, shares_owned)

        # Sell logic: Profit-taking and stop-loss
        if avg_buy_price and avg_buy_price > 0:
            roi = (stock_price - avg_buy_price) / avg_buy_price * 100
            if roi >= self.profit_threshold * self.hold_longer_threshold * 0.8:
                return "sell", 0, sell_qty
            if roi <= self.stop_loss_threshold:
                return "sell", 0, sell_qty

        # Buy logic: Strong uptrend with enough momentum
        if short_ma > long_ma and momentum_strength > self.momentum_sensitivity:
            return "buy", buy_qty, 0

        return "hold", 0, 0

    def determine_trade_quantity(self, ticker, stock_price, shares_owned):
        """Determines trade size using momentum-sensitivity and volatility."""
        stock_data = self.get_price_series(ticker)
        if len(stock_data) < self.short_window:
            return 1, 1

        recent_volatility = stock_data.pct_change().rolling(window=self.short_window).std().iloc[-1]
        trade_intensity = min(1, recent_volatility * 10)

        max_buy_shares = 20
        max_sell_ratio = 0.6
        # === Buy Quantity: Based on budget and trade intent ===
        raw_buy_qty = int(trade_intensity * max_buy_shares * self.momentum_sensitivity)
        affordable_qty = int(self.budget // stock_price)
        buy_qty = max(1, min(raw_buy_qty, affordable_qty))

        # === Sell Quantity: Based on shares owned ===
        raw_sell_qty = int(shares_owned * trade_intensity * self.momentum_sensitivity)
        sell_qty = max(1, min(raw_sell_qty, int(shares_owned * max_sell_ratio))) if shares_owned > 0 else 0

        return buy_qty, sell_qty


class RiskTraderAgent(AgentBase):
    """
    A high-risk, high-reward trader that seeks to capture extreme price swings.
    Trades larger volumes based on volatility, aggressively buying dips or trading breakouts.
    Each agent has a unique risk profile for varied behavior.
    """

    def setup(self, agent_seed=None, agent_id=None):
        super().setup(agent_seed, agent_id)

        # Assign discrete risk behaviors
        self.trading_style = self.random_generator.choice(["dip_buyer", "breakout_trader", "volatility_scaler"])
        self.risk_factor = self.random_generator.uniform(1.5, 3.0)  # Aggressiveness in trading
        self.stop_loss_threshold = -self.random_generator.uniform(3.0, 6.0)  # Larger stop-loss range
        self.take_profit_threshold = self.random_generator.uniform(5.0, 10.0)  # Higher take-profit range
        self.max_trade_size = self.random_generator.integers(10, 25)  # Different position sizes

    def step(self):
        """Executes trades based on extreme price movements and volatility."""
        self.update_current_day()
        if self.current_day is None:
            return

        for ticker in self.ticker_list:
            try:
                stock_price = self.get_stock_price(ticker)
                if stock_price is None:
                    continue

                action, buy_quantity, sell_quantity = self.decide_action(ticker, stock_price)

                if action == "buy" and self.budget >= stock_price * buy_quantity:
                    self.buy(ticker, stock_price, buy_quantity)
                elif action == "sell" and self.portfolio.get(ticker, {'quantity': 0})['quantity'] >= sell_quantity:
                    self.sell(ticker, stock_price, sell_quantity)
                else:
                    self.hold(ticker, stock_price)

            except KeyError:
                print(f"Data for ticker {ticker} is not available on {self.current_day}")
                continue

    def decide_action(self, ticker, stock_price):
        """Determines whether to buy, sell, or hold based on risk strategy."""
        stock_data = self.get_price_series(ticker)
        if len(stock_data) < 20:
            return "hold", 0, 0

        short_ma = self.moving_average(stock_data, 5).iloc[-1]
        long_ma = self.moving_average(stock_data, 20).iloc[-1]
        volatility = self.calculate_volatility(stock_data, 5)
        prev_price = stock_data.iloc[-2]  # Previous day's price
        dip_strength = (stock_price - prev_price) / prev_price * 100  # % drop from yesterday

        buy_quantity, sell_quantity = self.determine_trade_quantity(ticker, stock_price, volatility)

        # Different risk trader behaviors
        if self.trading_style == "dip_buyer" and dip_strength < -3.0:
            return "buy", buy_quantity, 0

        if self.trading_style == "breakout_trader" and short_ma > long_ma and volatility > 1.5:
            return "buy", buy_quantity, 0

        if self.trading_style == "volatility_scaler" and volatility > self.risk_factor:
            return "buy", buy_quantity, 0

        holdings = self.portfolio.get(ticker, {'quantity': 0})['quantity']
        if holdings > 0:
            avg_buy_price = self.portfolio[ticker]['average_price']
            return_change = (stock_price - avg_buy_price) / avg_buy_price * 100

            if return_change > self.take_profit_threshold or return_change < self.stop_loss_threshold:
                return "sell", 0, sell_quantity

        return "hold", 0, 0

    def determine_trade_quantity(self, ticker, stock_price, volatility):
        """Determines different trade sizes for buy and sell based on volatility and risk appetite."""
        trade_intensity = min(1, volatility * self.risk_factor)
        buy_quantity = max(1, int(trade_intensity * self.max_trade_size * 1.2))  # Buy more aggressively
        sell_quantity = max(1, int(trade_intensity * self.max_trade_size * 0.8))  # Sell more cautiously

        max_affordable = self.budget // stock_price
        buy_quantity = min(buy_quantity, max_affordable) if max_affordable > 0 else 1

        holdings = self.portfolio.get(ticker, {'quantity': 0})['quantity']
        sell_quantity = min(sell_quantity, holdings)

        return buy_quantity, sell_quantity


class RiskAverseTraderAgent(AgentBase):
    """
    A highly conservative trader that avoids high-risk situations and prioritizes portfolio stability.
    Trades infrequently, focusing on safe, low-volatility opportunities with strict stop-loss and take-profit rules.
    """

    def setup(self, agent_seed=None, agent_id=None):
        super().setup(agent_seed, agent_id)

        # Assign individualized risk-averse behaviors
        self.safe_trade_window = self.random_generator.integers(10, 30)  # Longer-term trend tracking
        self.volatility_threshold = self.random_generator.uniform(1.0, 2.5)  # Avoids high volatility stocks
        self.safe_profit_threshold = self.random_generator.uniform(2.0, 4.0)  # Conservative profit targets
        self.safe_stop_loss = -self.random_generator.uniform(1.0, 1.5)  # Tight stop-loss to minimize risk
        self.trade_frequency = self.random_generator.uniform(0.2, 0.5)  # Less frequent trading

    def step(self):
        """Executes trades only in stable market conditions with a focus on risk avoidance."""
        self.update_current_day()
        if self.current_day is None:
            return

        for ticker in self.ticker_list:
            try:
                stock_price = self.get_stock_price(ticker)
                if stock_price is None:
                    continue

                action, buy_quantity, sell_quantity = self.decide_action(ticker, stock_price)

                if action == "buy" and self.budget >= stock_price * buy_quantity:
                    self.buy(ticker, stock_price, buy_quantity)
                elif action == "sell" and self.portfolio.get(ticker, {'quantity': 0})['quantity'] >= sell_quantity:
                    self.sell(ticker, stock_price, sell_quantity)
                else:
                    self.hold(ticker, stock_price)

            except KeyError:
                print(f"Data for ticker {ticker} is not available on {self.current_day}")
                continue

    def decide_action(self, ticker, stock_price):
        """Determines whether to buy, sell, or hold based on stable market conditions, reducing panic selling."""
        stock_data = self.get_price_series(ticker)
        if len(stock_data) < self.safe_trade_window:
            return "hold", 0, 0
        if self.random_generator.uniform(0, 1) > self.trade_frequency:
            return "hold", 0, 0

        long_term_ma = self.moving_average(stock_data, self.safe_trade_window).iloc[-1]
        volatility = self.calculate_volatility(stock_data, 10)

        buy_quantity, sell_quantity = self.determine_trade_quantity(ticker, stock_price, volatility)

        # Buy only if stock is stable (low volatility) and price is in a long-term uptrend
        if volatility < self.volatility_threshold and stock_price > long_term_ma:
            return "buy", buy_quantity, 0

        holdings = self.portfolio.get(ticker, {'quantity': 0})['quantity']
        if holdings > 0:
            avg_buy_price = self.portfolio[ticker]['average_price']
            return_change = (stock_price - avg_buy_price) / avg_buy_price * 100

            if return_change > self.safe_profit_threshold:
                return "sell", 0, sell_quantity

            # **Holding buffer before panic selling**
            elif return_change < self.safe_stop_loss:
                prev_price = stock_data.iloc[-2]  # Previous day’s price
                price_trend = stock_price - prev_price  # Detect price direction

                # If price trend is still **stable or slightly increasing**, do NOT sell yet
                if price_trend > 0:
                    return "hold", 0, 0
                else:
                    return "sell", 0, sell_quantity

        return "hold", 0, 0

    def determine_trade_quantity(self, ticker, stock_price, volatility):
        """Determines trade size conservatively based on decision thresholds."""
        trade_intensity = max(0.1,
                              1 - (volatility / self.volatility_threshold))  # Avoid large trades in volatile markets
        max_affordable = self.budget // stock_price  # Ensure budget limit
        holdings = self.portfolio.get(ticker, {'quantity': 0})['quantity']

        buy_quantity = int(trade_intensity * max_affordable) if max_affordable > 0 else 1
        sell_quantity = int(trade_intensity * holdings)

        return buy_quantity, sell_quantity


class HerdingTraderAgent(AgentBase):
    def setup(self, agent_seed=None, agent_id=None):
        super().setup(agent_seed, agent_id)
        self.herd_sensitivity = self.random_generator.uniform(0.5, 2.0)

    def step(self):
        """Trades based on the majority action of other agents in the market."""
        self.update_current_day()
        if self.current_day is None:
            return

        for ticker in self.ticker_list:
            majority_action, herd_strength = self.get_majority_action(ticker)
            price = self.get_stock_price(ticker)
            if price is None:
                continue
            # 15% probability for the herding agent to act differently from the majority (increased randomness)
            if self.random_generator.uniform(0, 1) < 0.15:
                majority_action = self.random_generator.choice(["Buy", "Sell", "Hold"])

            # Get current budget and shares owned
            shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']
            max_buy_quantity = self.budget // price if price > 0 else 0  # Maximum affordable shares

            # === New: Scale Trade Quantity Based on Budget ===
            budget_factor = min(1, self.budget / 50000) ** 0.5 # Scale trades relative to a max budget of 50k
            herd_factor = min(1, abs(herd_strength))  # Scale based on herd strength (capped at 1)

            # Adjust trade quantity dynamically
            base_trade_size = 5
            trade_quantity = max(3, int(base_trade_size * herd_factor*5 * budget_factor))

            if majority_action == "Buy" and max_buy_quantity > 0:
                buy_quantity = min(trade_quantity, max_buy_quantity)
                if buy_quantity > 0:
                    self.buy(ticker, price, buy_quantity)

            elif majority_action == "Sell" and shares_owned > 0:
                sell_quantity = min(trade_quantity, shares_owned)
                if sell_quantity > 0:
                    self.sell(ticker, price, sell_quantity)

            else:
                self.hold(ticker, price)

    def get_majority_action(self, ticker):
        """Determines majority market action and strength based on recent history."""
        recent_actions = [
            entry for entry in self.model.history
            if entry['Date'] == self.current_day and entry.get('Ticker') == ticker
        ]

        if not recent_actions:
            return "Hold", 0

        total_agents = max(1, len(self.model.agents))  # Prevent division by zero
        buy_count = sum(1 for action in recent_actions if action['Action'].lower() == "buy")
        sell_count = sum(1 for action in recent_actions if action['Action'].lower() == "sell")

        # Compute herd strength with a normalization factor
        herd_strength = (buy_count - sell_count) / total_agents
        herd_strength = max(-1, min(1, herd_strength * self.herd_sensitivity))

        # Adjusted thresholds for responsiveness
        if herd_strength > 0.1:
            return "Buy", herd_strength
        elif herd_strength < -0.1:
            return "Sell", herd_strength
        return "Hold", herd_strength


class ReinforcementAgent(AgentBase):
    def setup(self, agent_seed=None, agent_id=None):
        super().setup(agent_seed, agent_id)
        # Remember last cumulative return
        self.prev_cumulative_return = 0.0
        self.prev_win_rate = 0.0
        self.total_holdings = 0
        self.hold_count = 0
        self.patience = 20
        self.today_invalid_action_penalty = 0.0

    def step(self, action):
        """
        Executes multi-stock actions received from the RL model.
        :param action: Encoded integer action from DQN, which represents actions for all stocks.
        """
        self.update_current_day()  # Ensure we're on the right day

        if self.current_day is None:
            return

        self.prev_cumulative_return = self.cumulative_return
        self.prev_win_rate = self.win_rate

        # === Execute Individual Actions for Each Stock ===
        # With penalty calculation here for DQN model.
        for i, ticker in enumerate(self.ticker_list):
            price = self.get_stock_price(ticker)
            if price is None:
                continue  # Skip if no valid price

            trade_quantity = self.calculate_trade_quantity()

            if action[i] == 0:  # Hold
                p0 = self.evaluate_invalid_action_penalty(ticker, 0, price)
                self.today_invalid_action_penalty += p0
                self.hold_count += 1
                self.total_holdings += 1
                self.hold(ticker, price)
            elif action[i] == 1:  # Sell
                p1 = self.evaluate_invalid_action_penalty(ticker, 1, price)
                self.today_invalid_action_penalty += p1
                self.sell(ticker, price, quantity=trade_quantity)
            elif action[i] == 2:  # Buy
                p2 = self.evaluate_invalid_action_penalty(ticker, 2, price)
                self.today_invalid_action_penalty += p2
                self.buy(ticker, price, quantity=trade_quantity)

    def calculate_trade_quantity(self):
        """
        Determines buy/sell quantity dynamically based on the RL agent's own performance.
        Returns a value between 1 and 5.
        """
        profitability = self.profitability_score  # Stability of past trades
        win_rate = self.win_rate if self.win_rate > 0 else 50  # Default to neutral win rate (50%)

        # Normalize profitability between 0 and 1 (assuming range [-1,1])
        profitability_normalized = (profitability + 1) / 2  # Converts [-1,1] to [0,1]

        # Normalize win rate between 50%-100% to [0,1] scale
        win_rate_normalized = (win_rate - 50) / 50

        # Adjust trade quantity based only on the agent's performance
        trade_quantity = 1 + 4 * (0.7 * profitability_normalized + 0.3 * win_rate_normalized)  # Weighted formula

        # Ensure trade quantity stays within [1,5]
        return max(1, min(5, int(trade_quantity)))

    def evaluate_invalid_action_penalty(self, ticker, action, price):
        """
        Evaluates penalty for a single stock-action pair.
        Includes hold fatigue, invalid buy/sell logic.
        Updates internal today_invalid_action_penalty.
        """
        penalty = 0.0
        shares_owned = self.portfolio.get(ticker, {'quantity': 0})['quantity']
        max_buy_quantity = int(self.budget / price) if price > 0 else 0

        # === Penalty for too many hold action to ensure RL agent try to be more active in the market
        if action == 0:  # Hold
            self.hold_count += 1
            if self.hold_count >= self.patience:
                penalty -= 1.0
                self.hold_count = 0  # Reset hold fatigue
            else:
                penalty -= 0.1

        elif action == 1:  # Sell
            if shares_owned == 0:
                penalty -= 1.0  # Hard fail
            elif shares_owned < 1:
                share_deficit = (5 - shares_owned) / 5
                penalty -= share_deficit

        elif action == 2:  # Buy
            if max_buy_quantity == 0:
                penalty -= 1.0
            elif max_buy_quantity < 5:
                budget_deficit = (5 - max_buy_quantity) / 5
                penalty -= budget_deficit
        return penalty