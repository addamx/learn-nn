import torch
from torch import nn
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np


# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 设定随机数生成器的种子
    # 使随机数序列每次运行时都是确定的
    np.random.seed(0)
    red, _ = make_blobs(n_samples=num, centers=[[0, 0]], cluster_std=0.15)
    green, _ = make_circles(n_samples=num, noise=0.02, factor=0.7)
    blue, _ = make_blobs(n_samples=num, centers=[[-1.2, -1.2], [-1.2, 1.2], [1.2, -1.2], [1.2, 1.2]], cluster_std=0.2)
    return red, green, blue


def draw_data():
    red, green, blue = make_data(100)
    # 创建-4到4的平面画板
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    axis.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title='Neural Network', xlabel='X', ylabel='Y')
    # 绘制数据
    plt.scatter(green[:, 0], green[:, 1], color='green')
    plt.scatter(red[:, 0], red[:, 1], color='red')
    plt.scatter(blue[:, 0], blue[:, 1], color='blue')


# 多分类模型
class MultiClassClassificationModel(nn.Module):
    # 对于多分类任务，输出维度等于类别的数量
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)

    # 前向传播
    def forward(self, x):
        x = self.layer1(x)  # 线性变换
        x = torch.relu(x)  # 激活
        return self.layer2(x)  # 输出


def make_model():
    # 环境变量设置
    n_features = 2  # 特征数，训练数据是2维的，所以特征数必须是2（每一层的输入和输出维度必须匹配。n_features 确保上一层的输出与下一层的输入相匹配，从而正确传递数据）
    n_hidden = 10  # 隐藏层神经元数量，实验发现n_hidden出现过拟合（损失突然上升）
    n_classes = 3  # 类别数，这里只有 绿蓝红，不能少于3
    n_epoches = 1000  # 迭代次数
    learning_rate = 0.001  # 学习速率

    # 准备训练数据
    red, green, blue = make_data(1000)
    green = torch.FloatTensor(green)  # 1000x2
    blue = torch.FloatTensor(blue)
    red = torch.FloatTensor(red)

    # 组成训练数据 3000x2
    data = torch.cat((green, blue, red), dim=0)

    # 设置标签，用于计算损失 [0, 0, ....1, 1, ....2, 2]
    label = torch.LongTensor([0] * len(green) + [1] * len(blue) + [2] * len(red))

    # 创建神经网络模型实例
    # 3000x2(训练数据)矩阵 相乘 2x15(特征数*每个隐藏层神经单元数量)
    # 由于左乘时，左侧矩阵的列必须等于右侧矩阵的行，
    model = MultiClassClassificationModel(n_features, n_hidden, n_classes)
    # criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务。【！！】而且对比时，数据和标签的维度是一致，所以不适合当前的任务
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于分类任务。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoches):
        output = model(data)
        loss = criterion(output, label)
        loss.backward()  # 通过自动微积分计算梯度
        optimizer.step()  # 更新参数，使得损失函数结果减少
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')

    return model


# 生成用于绘制决策边界的等高线数据
# min-x到max-x是画板的横轴范围，min-y到max-y是画板的纵轴范围
# 不同的类别结果对应不同的高度
# 基于数据点的坐标与高度数据，绘制等高线
def train_and_predict(minx, maxx, miny, maxy, model):
    # 生成网格数据点
    # 每个点的距离是0.02，这样点可以覆盖平面全部范围
    # xx, xy 为 400x400矩阵
    xx, xy = np.meshgrid(np.arange(minx, maxx, 0.02),
                         np.arange(miny, maxy, 0.02))
    # 数据点的横坐标、纵坐标和高度
    # xs, ys 为 1x160000矩阵
    xs = xx.ravel()
    ys = xy.ravel()
    z = list()
    for x, y in zip(xs, ys):
        test_point = torch.FloatTensor([[x, y]])
        output = model(test_point)
        # 选择概率最大的类别（即index)
        _, mostlyClassIndex = torch.max(output, 1)   # 取1维（行）上最大值，返回[max_values, max_value_indexes]
        z.append(mostlyClassIndex.item())  # 添加到高度z中
    # 将z重新设置为何xx相同的形状，即160000列表整形为400x400矩阵
    z = np.array(z).reshape(xx.shape)
    return xx, xy, z


if __name__ == '__main__':
    xx, xy, z = train_and_predict(-4, 4, -4, 4, make_model())

    draw_data()
    # contour绘制多分类的决策边界(等高线图)
    plt.contour(
        xx,  # X轴（等高线图的网格点坐标。X 和 Y 必须具有相同的形状。如果省略，默认为网格点的索引）
        xy,  # Y轴
        z,  # 绘制的等高线数据，
        colors=['orange']
    )
    plt.show()
