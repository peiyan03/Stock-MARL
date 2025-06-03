import pandas as pd
from stable_baselines3 import DQN
from simulation.environment import Env
from simulation.tools import EpisodeTracker
import warnings
warnings.simplefilter("error", RuntimeWarning)


def load_stock_data():
    """Load stock market data for training."""
    return pd.read_csv("../../../resources/datasets/test_dataV1.csv", header=[0, 1], index_col=0)


def initialize_env(stock_data):
    """Create and configure the trading environment."""
    agent_counts = {
        'RandomBuyerAgent': 5,
        'DayTraderAgent': 7,
        'MomentumTraderAgent':6,
        'RiskTraderAgent': 6,
        'RiskAverseTraderAgent':7,
        'HerdingTraderAgent': 4,
        'ReinforcementAgent': 1,
    }
    max_steps_per_episode = len(stock_data)
    return Env(stock_data=stock_data, agent_counts=agent_counts)


def run_trained_model():
    """Runs the trained RL model on a full dataset without episodes."""
    print("🚀 Running trained RL agent on new dataset...")

    # Load new stock data
    stock_data = load_stock_data()

    # Initialize environment
    env = initialize_env(stock_data)

    # Load trained RL model from ZIP file
    model = DQN.load("DQN_models/final_DQN_agent-576674.zip", device="cuda")

    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    print("✅ Simulation completed. ")


if __name__ == "__main__":
    run_trained_model()
