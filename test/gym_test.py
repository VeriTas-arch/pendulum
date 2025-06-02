from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN

env_name = "LunarLander-v3"
env = gym.make(env_name, render_mode="human")

DATA_DIR = f"{Path(__file__).parent.parent}/data"
MODEL_DIR = f"{DATA_DIR}/dqn_lunarlander_model_test.zip"
model = DQN.load(MODEL_DIR, env=env)

episodes = 10
for episode in range(1, episodes + 1):
    state, _ = env.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        env.render()
        action, _ = model.predict(state)
        n_state, reward, terminated, truncated, info = env.step(action)
        score += reward
        state = n_state
    print("Episode : {}, Score : {}".format(episode, score))

env.close()
