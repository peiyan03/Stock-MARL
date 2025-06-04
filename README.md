# StockMARL: A Multi-Agent Reinforcement Learning System for Financial Market Simulation

**Author:** Peiyan Zou  
**Supervisor:** Dr Peer-Olaf Siebers  
**Institution:** University of Nottingham  
**Date:** April 2025  

---

## Project Overview

**StockMARL** is a novel hybrid framework that integrates **Multi-Agent Simulation (MAS)** with **Reinforcement Learning (RL)** to improve financial market forecasting and trading decision-making.  
The system allows a Deep Q-Network (DQN)-controlled RL agent to learn trading strategies by observing the behaviours and performances of diverse **rule-based reactive agents**, rather than relying solely on historical price data.

The implementation leverages:
- **AgentPy** for agent-based simulation,
- **Gymnasium** for RL environment wrapping,
- **Stable-Baselines3** for training the DQN model.

---

## System Features

- **Diverse Reactive Agents**: Includes Random Buyers, Day Traders, Momentum Traders, Risk/Risk-Averse Traders, and Herding Traders.
- **RL Agent**: Learns from peer behaviours (buy/sell/hold decisions and performance metrics).
- **Behaviour-Driven Observation Space**: Captures agent actions, profitability, win rates, and market sentiment.
- **Multi-Asset Support**: Supports simulation and trading over multiple real-world stocks (e.g., AAPL, META, VISA, XOM).
- **Reward Engineering**: Custom reward function aligned with trading performance, risk control, and behavioural realism.

---

##  How It Works

1. **Agent Initialisation**: All agents are instantiated with stochastic seeds to ensure behavioural diversity and reproducibility.
2. **Training Loop**:
   - Each episode represents a simulated trading lifecycle (252 trading days).
   - The RL agent receives an observation vector and outputs a single integer action (decoded to multi-stock decisions).
   - The environment executes all trades, updates financial states, and returns rewards.
3. **Reward Calculation**:
   - 70% weight on Net Worth and Portfolio Growth.
   - 30% on Money-Weighted Rate of Return (MWRR).
   - Penalties for invalid trading and excessive holding.

---

## Evaluation Highlights

- **RL agent outperforms** most reactive agents in generalisation test.
- Achieved **12.23% Yearly MWRR** and **Profitability Score: 4.85 ± 1.63** with **low volatility**.
- Final configuration tested across **700×252 trading days** in simulation.

---
