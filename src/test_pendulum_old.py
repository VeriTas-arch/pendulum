from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC

ENV_TYPE = 0
MODEL_TYPE = "SAC"  # SAC or PPO
MODE = "stable"  # test for swing up, stable for stable control
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if ENV_TYPE != 0:
    raise ValueError("This script is only for Pendulum-v1. ")
if MODE != "stable":
    raise ValueError("This script is only for stable control. ")

env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="human")

if MODEL_TYPE == "SAC":
    model = SAC.load(f"{DATA_DIR}/sac_pendulum_{MODE}.zip", env=env)
elif MODEL_TYPE == "PPO":
    model = PPO.load(f"{DATA_DIR}/ppo_pendulum_{MODE}.zip", env=env)

# 运行模型
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
        print("reset")
