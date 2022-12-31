import numpy as np
import matplotlib.pyplot as plt

def gen_grid_points():
    return np.stack(np.meshgrid(np.linspace(start=0, stop=1, num=40),
                                np.linspace(start=0, stop=1, num=40)), axis=2)

def gen_line_points():
    return np.linspace(start=[0.1, 0.8], stop=[0.7, 0.3], num=100)

def plot_grid(xy, line):
    xy_cont = xy.reshape(-1, 2)
    plt.scatter(xy_cont[:, 0], xy_cont[:, 1], s=0.5)
    for i in range(len(xy)):
        plt.plot(xy[i, :, 0], xy[i, :, 1], c='b', linewidth=0.4)
        plt.plot(xy[:, i, 0], xy[:, i, 1], c='b', linewidth=0.4)
    plt.plot(line[:, 0], line[:, 1], c='r', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.show()

def apply_transform(xy):
    return np.power(xy, 1.2)

xy_grid = gen_grid_points()
xy_line = gen_line_points()

plot_grid(xy_grid, xy_line)

xy_grid = apply_transform(xy_grid)
xy_line = apply_transform(xy_line)

plot_grid(xy_grid, xy_line)