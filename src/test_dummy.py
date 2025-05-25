import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np  # noqa: F401

from custom_envs import CustomRotaryInvertedDoublePendulumEnv

# 加载模型
XML_NAME = "rotary_inverted_double_pendulum"  # "rotary_inverted_double_pendulum"

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/{XML_NAME}.xml"

env = CustomRotaryInvertedDoublePendulumEnv(mode="test", custom_xml_file=XML_DIR)

model = env.model
data = env.data

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
        tip_pos = data.site_xpos[site_id]
        tip_linear_velocity = data.site_xvelp[site_id][:3]
        tip_vel = np.linalg.norm(tip_linear_velocity)
        print("link2 末端速度：", tip_vel)
        # print("link2 末端位置：", tip_pos)
        x, _, y = data.site_xpos[4]
        # print("x:", x, "y:", y)

        reward, reward_info = env.compute_reward(x, y, False)
        # print("reward:", reward)
        # print(reward_info)

        time.sleep(STEP)
