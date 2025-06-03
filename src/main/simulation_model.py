import agentpy as ap
import pandas as pd
import numpy as np
from ReactiveAgents import RandomBuyerAgent, HerdingTraderAgent, DayTraderAgent, MomentumTraderAgent, RiskTraderAgent, RiskAverseTraderAgent, ReinforcementAgent

class StockMarketModel(ap.Model):
    def setup(self):
        """
        Purpose: Defines the StockMarketModel class, an AgentPy-based multi-agent stock market simulation.

        Responsibilities:
            - Initialize stock market simulation and manage reactive and reinforcement agents.
            - Track the progression of simulated market events and trading actions.
            - Maintain structured logs of agent initialization for reproducibility and debugging.
            - Facilitate interaction between agents, including decision-making influenced by reinforcement learning.

        Main Components:
            - StockMarketModel.setup():
                Initializes the simulation environment, creates reactive and RL agents, and logs detailed initialization information.

            - StockMarketModel.step(rl_action=None):
                Executes one simulation timestep, processing all agent actions, including actions from RL model.

            - StockMarketModel.end():
                Completes the simulation, converting trade history into a structured DataFrame for analysis.

        Usage Context:
            Primarily used within a Gym-compatible environment (`environment.py`) to perform episodic reinforcement learning simulations,
            enabling evaluation and optimization of trading strategies in dynamic market conditions.
        """
        self.t = 0

        self.global_seed = self.p.get("global_seed", None)
        self.stock_data = self.p.get('stock_data')
        agent_counts = self.p.get('agent_counts', {})

        self.history = []
        self.history_df = pd.DataFrame()

        if self.global_seed is None:
            raise ValueError("Error: A 'global_seed' must be provided in the parameters for reproducibility.")
        if self.stock_data is None or self.stock_data.empty:
            raise ValueError("Stock data is missing or empty inside StockMarketModel!")
        if agent_counts is None:
            raise ValueError("Agent Counts is missing!")

        # Use NumPy’s recommended random generator
        self.rng = np.random.default_rng(self.global_seed)

        agent_classes = {
            'ReinforcementAgent': ReinforcementAgent,
            'RandomBuyerAgent': RandomBuyerAgent,
            'DayTraderAgent': DayTraderAgent,
            'MomentumTraderAgent': MomentumTraderAgent,
            'RiskTraderAgent': RiskTraderAgent,
            'RiskAverseTraderAgent': RiskAverseTraderAgent,
            'HerdingTraderAgent': HerdingTraderAgent,
        }

        self.agents = ap.AgentList(self, [])
        self.agent_initialize_log = pd.DataFrame(columns=["Agent Type", "Agent ID", "Seed"])
        self.rl_agent = None  # Currently only one rl_agent

        for agent_name, agent_class in agent_classes.items():
            count = agent_counts.get(agent_name, 0)
            self.agent_id_counter = {}
            self.agent_id_counter[agent_name] = 0

            for _ in range(count):
                self.agent_id_counter[agent_name] += 1
                agent_id = f"{agent_name}_{self.agent_id_counter[agent_name]}"
                agent_seed = self.rng.integers(0, 10 ** 6)
                agent = agent_class(self, agent_seed=agent_seed, agent_id=agent_id)  # Pass the unique seed

                if agent_class == ReinforcementAgent:
                    self.rl_agent = agent

                self.agents.append(agent)  # Add multiple agents of each type
                self.agent_initialize_log.loc[len(self.agent_initialize_log)] = {
                    "Agent Type": agent_name,
                    "Agent ID": agent_id,
                    "Seed": agent_seed
                }

        if self.agent_initialize_log is None:
            raise ValueError("Agent Initialization is not successful!")

    def step(self, rl_action=None):
        """Executes a step for all agents in the simulation. Make sure Rl action agent is listened to the Rl model"""
        print(f" -------------- Day {self.t} at {self.stock_data.index[self.model.t]} is processing ... -------------- ")

        # If it is Rl agent, use the DQN model's action. Otherwise, follow the default reactive agent's simulation
        for agent in self.agents:
            if isinstance(agent, ReinforcementAgent):
                agent.step(rl_action)
            else:
                agent.step()

    def end(self):
        """Ends the simulation."""
        print("\n --------------       Simulation complete.    -------------- ")

        if self.history:
            self.history_df = pd.DataFrame(self.history)  # Convert trade history to DataFrame for analysis
        else:
            raise ValueError("StockModel: No History data available!")
