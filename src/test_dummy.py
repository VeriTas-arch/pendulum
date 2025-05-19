import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
MODEL_NAME = "rotary_inverted_double_pendulum"  # "rotary_inverted_double_pendulum"

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/{MODEL_NAME}.xml"
MODEL = mujoco.MjModel.from_xml_path(XML_DIR)
DATA = mujoco.MjData(MODEL)

# 初始姿态：轻微扰动
DATA.qpos[:] = [0.0, np.pi, 0.0]

# 控制参数
STEP = 0.005
IMPULSE_STEP = 0.5 / STEP * 0.05  # 施加冲量，计算方式为 最大扭矩 / 步长 * 比例系数

impulse = 0.0


def keyboard_callback(key):
    global impulse
    if key == glfw.KEY_LEFT:
        impulse = IMPULSE_STEP
        print(f"Impulse applied: {impulse}")
    elif key == glfw.KEY_RIGHT:
        impulse = -IMPULSE_STEP
        print(f"Impulse applied: {impulse}")
    elif key == glfw.KEY_ESCAPE:
        # 退出程序
        print("Exiting...")
        glfw.set_window_should_close(glfw.get_current_context(), True)


def control_callback(model, data):
    global impulse
    # 只在当前帧施加冲量
    data.qfrc_applied[0] = impulse
    # 施加后立即清零
    impulse = 0.0


# 启动 Viewer（被动模式）
with mujoco.viewer.launch_passive(
    MODEL, DATA, key_callback=keyboard_callback
) as viewer:
    while viewer.is_running():
        control_callback(MODEL, DATA)
        mujoco.mj_step(MODEL, DATA)
        viewer.sync()
        time.sleep(STEP)
