import numpy as np
import matplotlib.pyplot as plt


class MLinearRegression:
    def __init__(self):
        self.coef_ = None  # 权重
        self.interception_ = None  # 截距
        self._theta = None  # 权重 + 截距

    def fit(self, X_train, y_train):
        """
        :param X_train: 样本
        :param y_train: 标签
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        ones = np.ones((X_train.shape[0], 1))  # 拼接恒为1的列，用于与截距相乘
        X_b = np.hstack([ones, X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict])
        return X_b.dot(self._theta)

    def mean_squared_error(self, y_predict, y_true):
        return np.sum((y_predict - y_true) ** 2) / len(y_predict)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return 1 - self.mean_squared_error(y_predict, y_test) / np.var(y_test)

