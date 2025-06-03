import numpy as np
import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment import Env
from utils.tools import EpisodeTracker
import warnings
warnings.simplefilter("error", RuntimeWarning)


def load_stock_data():
    """Load stock market data for training."""
    return pd.read_csv("../../resources/datasets/train_dataV1.csv", header=[0, 1], index_col=0)


def initialize_env(stock_data):
    """Create and configure the trading environment."""
    agent_counts = {
        'RandomBuyerAgent': 5,
        'DayTraderAgent': 7,
        'MomentumTraderAgent': 6,
        'RiskTraderAgent': 6,
        'RiskAverseTraderAgent': 7,
        'HerdingTraderAgent': 4,
        'ReinforcementAgent': 1,
    }
    max_steps_per_episode = 252
    return Env(stock_data=stock_data, agent_counts=agent_counts, max_steps_per_episode=max_steps_per_episode)


def train_rl_agent(env, tracker, total_timesteps=700*252): #1500*252
    """Trains the RL agent with performance tracking and early stopping."""
    model = DQN(
        "MlpPolicy", env,
        buffer_size=500000,  # Large replay buffer
        batch_size=128,  # Stable batch size 
        train_freq=(4 , "step"),  # Update Q-network every 4 steps
        gradient_steps=1,  # One gradient update per step
        target_update_interval=5000,  # Target network update delay in steps
        exploration_fraction=0.35,  # exploration percentage over total steps
        exploration_final_eps=0.1,  # 10% randomness remains at end
        verbose=0,
        device="cuda"
    )
    env.rl_model = model
    print(model.device)

    # Define a custom callback to track progress and integrate with EpisodeTracker
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.step = 0

        def _on_step(self):
            self.step += 1
            # Check for early stopping
            # if tracker.should_stop_early():
            #     print("!!!  Early stopping triggered. Training stopped. !!!")
            #     return False  # Returning False stops training

            if self.step == total_timesteps:
                print(f"✅ Training stopped at {total_timesteps} steps!")
                reward = self.locals['rewards']
                done = self.locals['dones']
                tracker.track_step(reward, done, env)
                return False
            # Access step-level data from the locals dictionary
            reward = self.locals['rewards']
            done = self.locals['dones']

            # Track the step using the EpisodeTracker
            tracker.track_step(reward, done, env)
            return True  # Continue training

    # Create the callback instance
    callback = CustomCallback()

    # Train the model using the learn method
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def main():
    """Main execution workflow."""
    print("🚀 Starting RL Training for Stock Market Simulation...")

    # Load stock data
    stock_data = load_stock_data()

    # Initialize environment
    env = initialize_env(stock_data)

    # Setup performance tracker
    tracker = EpisodeTracker()

    # Train the RL agent
    model = train_rl_agent(env, tracker)

    # Optionally save the model
    model.save("dqn_trading_agent")

    print(" ✅ Training completed. Model saved as 'dqn_trading_agent'.")


if __name__ == "__main__":
    main()
