import numpy as np
import onnx_utils as utils
import onnxruntime as ort
import torch
import torch.nn as nn
import time

model_dir = utils.model_dir


# 1. 读取 PyTorch 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# 加载pytorch模型
pytorch_model = SimpleCNN()
pytorch_model.load_state_dict(
    torch.load(f"{model_dir}/mnist_cnn.pth", map_location="cpu")
)
pytorch_model.eval()

# 2. 读取ONNX模型
onnx_session = ort.InferenceSession(f"{model_dir}/mnist_cnn.onnx")
input_name = onnx_session.get_inputs()[0].name

# 3. 构造输入
N = 100
inputs = np.random.rand(N, 1, 28, 28).astype(np.float32)
torch_inputs = torch.from_numpy(inputs)

# 4. PyTorch推理计时
t0 = time.time()
with torch.no_grad():
    for i in range(N):
        _ = pytorch_model(torch_inputs[i : i + 1])
torch_time = time.time() - t0

# 5. ONNX推理计时
t0 = time.time()
for i in range(N):
    _ = onnx_session.run(None, {input_name: inputs[i : i + 1]})
onnx_time = time.time() - t0

print(f"PyTorch模型推理{N}次耗时: {torch_time:.4f} 秒")
print(f"ONNX模型推理{N}次耗时: {onnx_time:.4f} 秒")
