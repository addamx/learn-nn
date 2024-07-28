import numpy as np
import matplotlib.pyplot as plt


class SimpleLearRegressionSelf:
    """
    **一元线性回归算法**
    - 使用最小二乘法求得最小误差的一元线性方程的参数a、b（y=a*x+b）
    """

    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        denominator = 0.0
        numerator = 0.0
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) ** 2
        self.a = numerator / denominator
        self.b = y_mean - self.a * x_mean
        return self

    def predict(self, x_test_group):
        return np.array([self._predict(x_test) for x_test in x_test_group])

    def _predict(self, x_test):
        return self.a * x_test + self.b


if __name__ == '__main__':
    x = np.array([1, 2, 4, 6, 8])
    y = np.array([2, 5, 7, 8, 9])

    model = SimpleLearRegressionSelf()
    model.fit(x, y)
    a = model.a
    b = model.b

    y_predict = a * x + b
    plt.scatter(x, y, color='b')  # 散点图
    plt.plot(x, y_predict, color='r')  # 线形图
    plt.xlabel('管子的长度', fontproperties='SimHei', fontsize=15)
    plt.ylabel('收费', fontproperties='SimHei', fontsize=15)
    plt.show()
