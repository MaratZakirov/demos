import numpy as np
import matplotlib.pyplot as plt

def gen_grid_points(a=0., b=1., c=0., d=1.):
    return np.stack(np.meshgrid(np.linspace(start=a, stop=b, num=40),
                                np.linspace(start=c, stop=d, num=40)), axis=2)

def gen_line_points(a=[0.1, 0.8], b=[0.7, 0.3]):
    return np.linspace(start=a, stop=b, num=100)

def plot_grid(xy, lines=None):
    xy_cont = xy.reshape(-1, 2)
    plt.scatter(xy_cont[:, 0], xy_cont[:, 1], s=0.5)
    for i in range(len(xy)):
        plt.plot(xy[i, :, 0], xy[i, :, 1], c='b', linewidth=0.4)
        plt.plot(xy[:, i, 0], xy[:, i, 1], c='b', linewidth=0.4)
    if not lines is None:
        colors = ['r', 'g', 'm', 'black']
        for i in range(len(lines)):
            plt.plot(lines[i][:, 0], lines[i][:, 1], c=colors[i], linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_grid_3d(xyz, line):
    return 0

def apply_transform(xy):
    return np.power(xy, 1.4)

# NYI
#def some_f(xy):
#    return 2 - (xy ** 4).sum(-1)[..., None]

def some_tr(xy):
    return -8 * np.power(xy, 3)

def some_inv_tr(xy):
    return -0.5 * np.power(np.abs(xy), 1/3) * np.sign(xy)

def run1():
    xy_grid = gen_grid_points()
    xy_line = gen_line_points()

    plot_grid(xy_grid, xy_line)

    xy_grid = apply_transform(xy_grid)
    xy_line = apply_transform(xy_line)

    plot_grid(xy_grid, xy_line)

def run2():
    xy_grid = gen_grid_points(a=-1, b=1.0, c=-1.0, d=1.0)
    plot_grid(xy_grid)

    xy_grid = some_tr(xy_grid)

    xy_line = gen_line_points([-3, -6], [6, 6])
    xy_line2 = gen_line_points([-3, -6], [4, 2])
    xy_line3 = gen_line_points([6, 6], [4, 2])

    plot_grid(xy_grid, [xy_line, xy_line2, xy_line3])

    xy_grid = some_inv_tr(xy_grid)
    xy_line = some_inv_tr(xy_line)
    xy_line2 = some_inv_tr(xy_line2)
    xy_line3 = some_inv_tr(xy_line3)

    plot_grid(xy_grid, [xy_line, xy_line2, xy_line3])

    #xyz_grid = np.stack([xy_grid, some_f(xy_grid)], axis=2)
    #xyz_line = np.stack([xy_line, some_f(xy_line)], axis=2)
    #plot_grid_3d(xyz_grid, xyz_line)

run2()