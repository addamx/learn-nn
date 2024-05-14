import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成非线性数据
np.random.seed(0)
x = np.linspace(-1, 1, 200)
y = x ** 3 + np.random.normal(0, 0.1, x.shape)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# 将数据转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型、定义损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测并绘制结果
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).detach().numpy()

# for name in model.state_dict():
#     print(name)

for name, paramer in model.named_parameters():
    print(name, paramer)

plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predicted, 'b-', label='Fitted line')
plt.legend()
plt.show()


