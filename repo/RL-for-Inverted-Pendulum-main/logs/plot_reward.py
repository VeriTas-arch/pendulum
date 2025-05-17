from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
current_dir = Path(__file__).parent
file_path = f"{current_dir}/DDPG.csv"  # Replace with the path to the CSV log file
data = pd.read_csv(file_path)

# Calculate the total reward for each episode
total_rewards_per_episode = data.groupby("ep")["reward"].sum()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(total_rewards_per_episode.index, total_rewards_per_episode.values, marker="o")
plt.xlabel("Episode Number")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.show()
