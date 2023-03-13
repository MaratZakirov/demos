import matplotlib.pyplot as plt
import numpy as np

xs = np.random.choice(np.linspace(start=-np.pi, stop=np.pi, num=10000), 1000)
ys = np.sin(xs) + 0.1 * np.random.randn(len(xs))
zs = np.sin(xs * ys)

D = np.stack([xs, ys, zs], axis=1)

plt.scatter(xs, ys, s=1.0)
plt.scatter(xs, zs, s=1.0)
plt.scatter(ys, zs, s=1.0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, s=3.0)
plt.show()