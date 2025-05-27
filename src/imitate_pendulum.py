from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import utils
import gymnasium as gym

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

MODE = "test"

# 创建环境（dummy，一般用 n_envs=1）
gym.register(
    id="CustomRotaryInvertedDoublePendulum-v1",
    entry_point="custom_envs:CustomRotaryInvertedDoublePendulumEnv",
)
dummy_env = make_vec_env(
    "CustomRotaryInvertedDoublePendulum-v1",
    n_envs=32,
    wrapper_class=gym.wrappers.TimeLimit,
    wrapper_kwargs={"max_episode_steps": 1000},
    env_kwargs={
        "mode": MODE,
        # "render_mode": "human"
    },
)

# 初始化 SAC 模型
dummy_model = SAC("MlpPolicy", dummy_env)

# 提取 Actor 网络（可用于行为克隆训练）
policy = dummy_model.actor

# 加载数据
good_dir = utils.IMITATION_DIR
obs = np.load(f"{good_dir}/observation.npy")
actions = np.load(f"{good_dir}/action.npy")
obs = torch.tensor(obs, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
min_len = min(obs.shape[0], actions.shape[0])
obs = obs[:min_len]
actions = actions[:min_len]
if actions.ndim == 1:
    actions = actions.unsqueeze(1)
dataset = TensorDataset(obs, actions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    for batch_obs, batch_act in loader:
        pred_act = policy(batch_obs)
        loss = loss_fn(pred_act, batch_act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存参数
torch.save(policy.state_dict(), f"{utils.IMITATION_DIR}/bc_actor.pt")

# 创建正式环境（比如 32 并行）
env = make_vec_env(
    "CustomRotaryInvertedDoublePendulum-v1",
    n_envs=32,
    wrapper_class=gym.wrappers.TimeLimit,
    wrapper_kwargs={"max_episode_steps": 1000},
    env_kwargs={
        "mode": MODE,
        "custom_xml_file": utils.PINOCCHIO_XML_DIR,
    },
)

# 正式模型
model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-4)

# 迁移行为克隆 actor 参数
bc_sd = torch.load(f"{utils.IMITATION_DIR}/bc_actor.pt")
model.actor.load_state_dict(bc_sd)

# 验证迁移是否成功（可加 log）
print("✅ 行为克隆参数成功迁移到 SAC actor")

# 可选：进行 fine-tune
# model.learn(total_timesteps=200_000)
model.save(f"{utils.IMITATION_DIR}/imitation_sac_finetuned.zip")
