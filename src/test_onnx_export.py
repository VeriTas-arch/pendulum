import time

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

import utils

# Basic configuration
MODEL_NAME = "sac_rotary_inverted_double_pendulum_stable_high_speed"

MODEL_PATH = f"{utils.DATA_DIR}/{MODEL_NAME}.zip"
ONNX_PATH = f"{utils.ONNX_DIR}/{MODEL_NAME}.onnx"
LOAD_MODEL = True
OPSET = 11  # ONNX opset version

# Register the custom environment
gym.register(
    id="CustomRotaryInvertedDoublePendulum-v3",
    entry_point="v_envs:CustomRotaryInvertedDoublePendulumEnv",
)
env = make_vec_env(
    "CustomRotaryInvertedDoublePendulum-v3",
    n_envs=1,
    env_kwargs={"mode": "stable", "custom_xml_file": utils.HIGH_SPEED_XML_DIR},
)

# Load the model
if LOAD_MODEL:
    model = SAC.load(MODEL_PATH, env=env)
else:
    raise ValueError("You must LOAD a model before exporting to ONNX.")

# Extract the actor network
actor = model.policy.actor.eval()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Construct dummy input
dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

# Export to ONNX model
torch.onnx.export(
    actor,
    dummy_input,
    ONNX_PATH,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
    opset_version=OPSET,
)

print(f"ONNX model save to: {ONNX_PATH}")

# === Optional: Verify with ONNX Runtime ===
print("Test with ONNX Runtime...")
sess = ort.InferenceSession(ONNX_PATH)
sample = np.random.randn(1, obs_dim).astype(np.float32)
output = sess.run(None, {"obs": sample})
print("Output action: ", output[0])

# Compare inference time between PyTorch and ONNX Runtime
N = 100
samples = np.random.randn(N, obs_dim).astype(np.float32)
torch_samples = torch.from_numpy(samples)

# PyTorch inference timing
actor.eval()
t0 = time.time()
with torch.no_grad():
    for i in range(N):
        _ = actor(torch_samples[i : i + 1])
torch_time = time.time() - t0

# ONNX inference timing
t0 = time.time()
for i in range(N):
    _ = sess.run(None, {"obs": samples[i : i + 1]})
onnx_time = time.time() - t0

print(f"Pytorch inferencing {N} times consumes: {torch_time:.4f} seconds")
print(f"ONNX inferencing {N} times consumes: {onnx_time:.4f} seconds")
