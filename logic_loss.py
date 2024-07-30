import numpy as np
import matplotlib.pyplot as pt

if __name__ == '__main__':
    def logp(x):
        return -np.log(1 - x)

    plot_x = np.linspace(0, 0.99, 50)
    plot_y = logp(plot_x)
    pt.plot(plot_x, plot_y)
    pt.show()
