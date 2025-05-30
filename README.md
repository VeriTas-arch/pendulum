# Implement RL

首先获取当前的状态量，函数如下所示。

```python
def _get_obs(self):
    qpos = self.data.qpos  # [theta_base, theta1, theta2]
    qvel = self.data.qvel  # [dtheta_base, dtheta1, dtheta2]

    theta_base = qpos[0]
    theta1 = qpos[1]
    theta2 = qpos[2]

    dtheta_base = qvel[0]
    dtheta1 = qvel[1]
    dtheta2 = qvel[2]

    obs = np.array(
        [
            np.cos(theta_base),
            np.sin(theta_base),
            np.cos(theta1),
            np.sin(theta1),
            np.cos(theta2),
            np.sin(theta2),
            dtheta_base,
            dtheta1,
            dtheta2,
        ],
        dtype=np.float32,
    )

    return obs
```

加载模型后，使用如下代码获取当前状态量对应的动作。则`action[0]`为当前输出力矩的归一量（最大为1，最小为-1）。

```python
action, _ = model.predict(obs, deterministic=True)
```

TODO: Sim2Real
