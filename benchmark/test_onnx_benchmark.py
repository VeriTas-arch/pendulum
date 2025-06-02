import time
import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

try:
    import utils
except ImportError:
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(f"{ROOT_DIR}/src")
    import utils

# Config
MODEL_NAME = "sac_rotary_inverted_double_pendulum_stable_high_speed"
MODEL_PATH = f"{utils.DATA_DIR}/{MODEL_NAME}.zip"
ONNX_PATH = f"{utils.ONNX_DIR}/{MODEL_NAME}.onnx"
N = 1000
OPSET = 11

# Register custom env
gym.register(
    id="CustomRotaryInvertedDoublePendulum-v3",
    entry_point="v_envs:CustomRotaryInvertedDoublePendulumEnv",
)
env = make_vec_env(
    "CustomRotaryInvertedDoublePendulum-v3",
    n_envs=1,
    env_kwargs={"mode": "stable", "custom_xml_file": utils.HIGH_SPEED_XML_DIR},
)

# Load SB3 SAC model
model = SAC.load(MODEL_PATH, env=env)
actor = model.policy.actor.eval()
obs_dim = env.observation_space.shape[0]

# Generate random observations
samples_np = np.random.randn(N, obs_dim).astype(np.float32)
samples_torch = torch.from_numpy(samples_np)


# Utility: benchmark function
def benchmark_torch(actor, device, samples, label):
    actor = actor.to(device)
    samples = samples.to(device)
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = actor(samples[0:1])

        # Time loop
        t0 = time.time()
        for i in range(N):
            _ = actor(samples[i : i + 1])
        if device.type != "cpu":
            torch.xpu.synchronize()
        t = time.time() - t0
    print(f"PyTorch ({label}) inference - {N} times consumes: {t:.4f} seconds")


def benchmark_onnx(onnx_path, samples_np):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warm up
    for _ in range(10):
        _ = sess.run(None, {input_name: samples_np[0:1]})

    # Time loop
    t0 = time.time()
    for i in range(N):
        _ = sess.run(None, {input_name: samples_np[i : i + 1]})
    t = time.time() - t0
    print(f"ONNX (CPU) inference - {N} times consumes: {t:.4f} seconds")


# Benchmark: PyTorch XPU
if torch.xpu.is_available():
    benchmark_torch(actor, torch.device("xpu"), samples_torch, "XPU")
else:
    print("XPU not available, skipping XPU benchmark.")

# Benchmark: PyTorch CPU
benchmark_torch(actor, torch.device("cpu"), samples_torch, "CPU")

# Benchmark: ONNX Runtime CPU
benchmark_onnx(ONNX_PATH, samples_np)
