import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from custom_envs import CustomRotaryInvertedDoublePendulumEnv

# 加载模型
XML_NAME = "rotary_inverted_double_pendulum"

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/{XML_NAME}.xml"

env = CustomRotaryInvertedDoublePendulumEnv(mode="test", custom_xml_file=XML_DIR)

model = env.model
data = env.data

# 初始姿态
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
        print("Exiting...")
        glfw.set_window_should_close(glfw.get_current_context(), True)


def control_callback(model, data):
    global torque
    data.ctrl[0] = torque
    torque = 0.0


with mujoco.viewer.launch_passive(
    model, data, key_callback=keyboard_callback
) as viewer:

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flag")
    tip_pos = data.site_xpos[site_id]

    while viewer.is_running():
        control_callback(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()

        x, _, y = data.site_xpos[4]
        reward, reward_info = env.compute_reward_test(x, y, False)
        # print("reward:", reward)
        # print(reward_info)

        time.sleep(STEP)
