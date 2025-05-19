import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
MODEL_NAME = "rotary_inverted_double_pendulum"  # "rotary_inverted_double_pendulum"

ASSET_DIR = f"{Path(__file__).parent.parent}/assets"
XML_DIR = f"{ASSET_DIR}/{MODEL_NAME}.xml"
MODEL = mujoco.MjModel.from_xml_path(XML_DIR)

data = mujoco.MjData(MODEL)

# 模拟的状态序列（假设提前准备好的 N×nq 的位置序列）
T = 1000  # 时间步数
STEP = 0.005

qpos_trajectory = np.zeros((T, 3))  # shape: (T, model.nq)
qpos_trajectory[:, 0] = np.linspace(-np.pi, np.pi, T)  # qpos[0]
qpos_trajectory[:, 1] = np.linspace(np.pi + np.pi / 3, np.pi - np.pi / 3, T)  # qpos[1]
qpos_trajectory[:, 2] = np.linspace(-np.pi / 4, np.pi / 4, T)  # qpos[2]

# 创建渲染器
with mujoco.viewer.launch_passive(MODEL, data) as viewer:
    for qpos in qpos_trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(MODEL, data)  # 更新场景
        viewer.sync()
        time.sleep(STEP)
