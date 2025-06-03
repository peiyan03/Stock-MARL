## Trained DQN Models Overview

The following Deep Q-Network (DQN) agents were trained within a consistent environment setup, using a Gym-compatible interface that wraps an AgentPy-based multi-agent stock market simulation. The RL agent operates over a discrete action space of size 3⁴ (Buy/Sell/Hold for 4 assets), learning via an improved reward function that balances net worth growth and money-weighted rate of return (MWRR).

---

### Model: `final_DQN_agent-576674.zip`
- Final Population Config selected for final evaluation and demonstration.
- Trained for 378,000 steps over 25,562 episodes.
- Dataset used: `train_dataV1.csv` including AAPL, META, XOM, and V.

**Agent Configuration:**

| Agent Type               | Count  |
|--------------------------|--------|
| RandomBuyerAgent         | 5      |
| DayTraderAgent           | 7      |
| MomentumTraderAgent      | 6      |
| RiskTraderAgent          | 6      |
| RiskAverseTraderAgent    | 7      |
| HerdingTraderAgent       | 4      |
| ***ReinforcementAgent*** | 1      |
| **Total Agents**         | 35 + 1 |

---

## Validation Models trained for 5040 steps: 
### Additional Checkpoints
- `dqn_trading_agent-111111.zip`
- `dqn_trading_agent-142342.zip`  
- `dqn_trading_agent-333333.zip`  
- `dqn_trading_agent-531311.zip`  
- `dqn_trading_agent-101010101010.zip`  

These represent intermediate training checkpoints used for experimentation with hyperparameters, environment stability, and model comparison throughout the development cycle. All models share the same simulation parameters and agent composition.

---
