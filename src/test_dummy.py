import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np  # noqa: F401

# 加载模型
MODEL_NAME = "rotary_inverted_double_pendulum"  # "rotary_inverted_double_pendulum"

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/{MODEL_NAME}.xml"

model = mujoco.MjModel.from_xml_path(XML_DIR)
data = mujoco.MjData(model)

# 初始姿态：轻微扰动
data.qpos[:] = [0.0, np.pi, 0.0]

# 控制参数
STEP = 0.005
TOQUE_STEP = 5  # 施加力矩，计算方式为 最大扭矩 / 步长 * 比例系数

torque = 0.0


def keyboard_callback(key):
    global torque
    if key == glfw.KEY_LEFT:
        torque = TOQUE_STEP
        print(f"Impulse applied: {torque}")
    elif key == glfw.KEY_RIGHT:
        torque = -TOQUE_STEP
        print(f"Impulse applied: {torque}")
    elif key == glfw.KEY_ESCAPE:
        # 退出程序
        print("Exiting...")
        glfw.set_window_should_close(glfw.get_current_context(), True)


def control_callback(model, data):
    global torque
    # 只在当前帧施加冲量
    data.ctrl[0] = torque
    # 施加后立即清零
    torque = 0.0


def compute_reward(model, data):
    # 目标末端位置
    x, _, y = data.site_xpos[4]
    target_pos = np.array([0, 0, 0.5365])
    theta = data.qpos[0]
    v0, v1 = data.qvel
    _healthy_reward = 0

    posture_reward = 0
    if y > 0.3:
        posture_reward = 3 * y

    ctrl_penalty = np.sum(data.ctrl[0] ** 2)

    alive_bonus = _healthy_reward - 10 * (y - target_pos[2]) ** 2
    dist_penalty = 1e-2 * (x - 0.2159) ** 2 + 1e-2 * abs(theta)
    vel_penalty = (7 * v0**2 + 1 * v1**2) * 7e-3 + 7e-2 * ctrl_penalty

    reward = alive_bonus - dist_penalty - vel_penalty + posture_reward
    reward_info = {
        "reward_survive": alive_bonus,
        "distance_penalty": -dist_penalty,
        "velocity_penalty": -vel_penalty,
    }

    return reward, reward_info


# 启动 Viewer（被动模式）
with mujoco.viewer.launch_passive(
    model, data, key_callback=keyboard_callback
) as viewer:

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flag")
    tip_pos = data.site_xpos[site_id]
    # print("link2 末端位置：", tip_pos)

    while viewer.is_running():
        control_callback(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        # print(data.qvel)
        # tip_pos = data.site_xpos[site_id]
        # print("link2 末端位置：", tip_pos)
        # print(data.qpos)
        x, _, y = data.site_xpos[4]
        print("x:", x, "y:", y)

        # reward, reward_info = compute_reward(model, data)
        # print("reward:", reward)
        # print(reward_info)

        time.sleep(STEP)
