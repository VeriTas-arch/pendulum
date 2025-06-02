import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/rotary_pendulum.xml"
model = mujoco.MjModel.from_xml_path(XML_DIR)
data = mujoco.MjData(model)

# 初始姿态
data.qpos[:] = [0.0, 1.5, 0.5]

# 控制参数
impulse = 0.0
impulse_step = 10


def keyboard_callback(key):
    global impulse
    if key == glfw.KEY_LEFT:
        impulse = impulse_step
        print(f"Impulse applied: {impulse}")
    elif key == glfw.KEY_RIGHT:
        impulse = -impulse_step
        print(f"Impulse applied: {impulse}")


def control_callback(model, data):
    global impulse
    # 只在当前帧施加冲量
    data.qfrc_applied[0] = impulse
    impulse = 0.0


# 启动 Viewer
with mujoco.viewer.launch_passive(
    model, data, key_callback=keyboard_callback
) as viewer:
    while viewer.is_running():
        control_callback(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
