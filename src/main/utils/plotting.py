from tools import Tools
import pandas as pd

"""
Loading the simulation history files from different location for simulation visualisation
    - input_files: trade_history_ep{i}.csv, trade history files
    - reward_files: episode_{i}.csv, reward history files
    
    - Trade_History_Buffer: Store different simulation runs with different parameters
    
    Using Tools.py to plot the visualisations from tools.py
"""

input_files = [f'../Trade_History/trade_history_ep{i}.csv' for i in range(1000, 1020)]
reward_files = [f'../Trade_History/Episode_Track/episode_{i}.csv' for i in range(1010,1020)]
file = '../Trade_History/trade_history_ep10.csv'

# input_files = [f'../Trade_History_Buffer/Trade_History-111111/trade_history_ep{i}.csv' for i in range(1, 4)]
# reward_files = [f'../Trade_History_Buffer/Trade_History-111111/Episode_Track/episode_{i}.csv' for i in range(1, 4)]

final_performance = Tools.summarize_agents_final_performance(input_files)
final_performance.to_csv('outputs/final_average_performance.csv', index=False)

Tools.plot_reward_trend_per_episode(reward_files)

Tools.plot_cumulative_return_trend(input_files)

# Tools.plot_single_trade_history(file)

