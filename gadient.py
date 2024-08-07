import numpy as np
import matplotlib.pyplot as plt


def J(theta):  # 损失函数
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')


def dJ(theta):  # 梯度，即：损失函数的导数
    return 2 * theta - 5


def draw_gradient_descent():
    plot_x = np.linspace(-1, 6, 141)
    plot_y = (plot_x - 2.5) ** 2 - 1
    plt.plot(plot_x, plot_y)
    plt.xlabel('theta')
    plt.ylabel('loss')

    epoches = 50
    theta = 0.0
    theta_history = [theta]
    eta = 1.1  # 步长
    epsilon = 1e-8  # 精度
    epoch_i = 0
    while epoch_i < epoches:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
        epoch_i += 1

    plt.plot(plot_x, J(plot_x), color='r')
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='b', marker='+')
    plt.show()
    print(f'总共走了{len(theta_history)}次')


if __name__ == '__main__':
    draw_gradient_descent()
