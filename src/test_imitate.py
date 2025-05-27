from imitation.data.types import TrajectoryWithRew
from imitation.data.serialize import save
import numpy as np

# 假设这些是你现有的数据
observations = np.load("observations.npy")  # shape: (N, obs_dim)
actions = np.load("actions.npy")            # shape: (N, act_dim)
dones = np.load("dones.npy")                # shape: (N,) - bool array

# 拆分成 trajectory（episode）列表
trajectories = []
start_idx = 0

for i, done in enumerate(dones):
    if done:
        end_idx = i + 1
        traj_obs = observations[start_idx:end_idx]
        traj_acts = actions[start_idx:end_idx]
        traj_dones = dones[start_idx:end_idx]
        trajectories.append(TrajectoryWithRew(
            obs=traj_obs,
            acts=traj_acts,
            rews=None,      # GAIL不需要奖励
            terminal=True
        ))
        start_idx = end_idx

# 保存为 imitation 的格式
save("expert_data", trajectories)
