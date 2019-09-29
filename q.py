import numpy as np
import matplotlib.pyplot as plt


def main():
    # 线的绘制
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    c, s = np.cos(x), np.sin(x)
    # 绘制
    plt.figure(1)
    # 自变量 因变量
    plt.plot(x, c)
    # 自变量 因变量
    plt.plot(x, s)
    plt.show()
    plt.savefig("one.png")


if __name__ == "__main__":
    main()