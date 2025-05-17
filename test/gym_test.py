from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN

env_name = "LunarLander-v3"
env = gym.make(env_name, render_mode="human")

# 加载训练好的模型
DATA_DIR = f"{Path(__file__).parent.parent}/data"
MODEL_DIR = f"{DATA_DIR}/dqn_lunarlander_model_test.zip"
model = DQN.load(MODEL_DIR, env=env)

episodes = 10
for episode in range(1, episodes + 1):
    state, _ = env.reset()  # reset 可能返回 (state, info)，需要解包
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):  # 检查 terminated 和 truncated
        env.render()  # 将当前的状态化成一个frame，再将该frame渲染到小窗口上
        action, _ = model.predict(state)  # 使用模型预测动作
        n_state, reward, terminated, truncated, info = env.step(action)  # 解包返回值
        score += reward
        state = n_state  # 更新状态
    print("Episode : {}, Score : {}".format(episode, score))

env.close()  # 关闭窗口
