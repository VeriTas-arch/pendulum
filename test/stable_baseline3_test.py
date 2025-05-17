from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

"""
Tuned hyperparameters for LunarLander-v3 using Stable Baselines3:
LunarLander-v3:
    n_timesteps: !!float 1e5
    policy: 'MlpPolicy'
    learning_rate: !!float 6.3e-4
    batch_size: 128
    buffer_size: 50000
    learning_starts: 0
    gamma: 0.99
    target_update_interval: 250
    train_freq: 4
    gradient_steps: -1
    exploration_fraction: 0.12
    exploration_final_eps: 0.1
    policy_kwargs: "dict(net_arch=[256, 256])"
"""

env_name = "LunarLander-v3"
env = gym.make(env_name)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# 创建模型
model = DQN(
    "MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs={"net_arch": [256, 256]},
)
model.learn(total_timesteps=1e5)

# 保存模型
DATA_DIR = f"{Path(__file__).parent.parent}/data"
model.save(f"{DATA_DIR}/dqn_lunarlander_model_test.zip")

# 加载模型（可选）
# model = DQN.load(f"{DATA_DIR}/dqn_lunarlander_model_test.zip", env=env)

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()

# 输出评估结果
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
