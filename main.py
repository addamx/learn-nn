from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 设定随机数生成器的种子
    # 使随机数序列每次运行时都是确定的
    np.random.seed(0)
    red, _ = make_blobs(n_samples=num,centers=[[0,0]], cluster_std=0.15)
    green, _ = make_circles(n_samples=num, noise=0.02, factor=0.7)
    blue, _ = make_blobs(n_samples=num, centers=[[-1.2, -1.2], [-1.2, 1.2], [1.2, -1.2], [1.2, 1.2]], cluster_std=0.2)
    return red, green, blue

if __name__ == '__main__':
    red, green, blue = make_data(100)
    # 创建-4到4的平面画板
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    axis.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], title='Neural Network', xlabel='X', ylabel='Y')
    # 绘制数据
    plt.scatter(green[:, 0], green[:, 1], color='green')
    plt.scatter(red[:, 0], red[:, 1], color='red')
    plt.scatter(blue[:, 0], blue[:, 1], color='blue')
    plt.show()


