import onnx_utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

model_dir = utils.model_dir
data_dir = utils.data_dir


# 定义模型结构
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


# 加载 MNIST 数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(str(data_dir), train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True,
)

# 训练模型
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 6):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 保存模型为 PyTorch 格式
torch.save(model.state_dict(), f"{model_dir}/mnist_cnn.pth")

# 加载训练好的模型权重
model = SimpleCNN()
model.load_state_dict(torch.load(f"{model_dir}/mnist_cnn.pth"))
model.eval()

# 示例输入
dummy_input = torch.randn(1, 1, 28, 28)

# 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    f"{model_dir}/mnist_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print("model successfully exported as mnist_cnn.onnx")
