import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles


def demo_make_blobs():
    # pointsLabels: 每个样本对应的簇标签
    points, pointsLabels = make_blobs(
        n_samples=100,
        n_features=2,  # n_features: 样本维度(默认2，注意, center的中心点维度也要一致才生效)
        centers=[[0, 0]],
        cluster_std=0.15,  # cluster_std: 簇的标准差
        random_state=0  # random_state: 随机种子，0表示每次样本都一样
    )
    return points

def demo_make_circles():
    return make_circles(
        n_samples=100,
        noise=0.02,  # 高斯噪声的标准差
        factor=0.7,  # 内圈和外圈之间的比例
        random_state=0
    )[0]

if __name__ == '__main__':
    points = demo_make_circles()

    board = plt.figure()
    # 在图形对象 board 上添加了一个占据整个画板的子图
    # nrows=1, ncols=1, index=1
    axis = board.add_subplot(1, 1, 1)
    axis.set(
        xlim=[-1.5, 1.5],  # x轴显示范围
        ylim=[-1.5, 1.5],  # y轴显示范围
        title='Neural Network', xlabel='X', ylabel='Y')

    plt.scatter(
        points[:, 0],  # 坐标列操作，取第1列
        points[:, 1],  # 坐标列操作，取第2列
        c='red',
        s=20,  # 数据点的大小
        alpha=1,  # 透明度
        # c=pointsLabels,  # 数据点颜色可以是一个颜色字符串或一个颜色序列。
        # cmap='viridis'  # 颜色映射表，如果样本有不同label，可以传给c然后通过cmap展示不同颜色的样本
    )
    plt.show()
