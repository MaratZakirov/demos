import numpy as np
from scipy import signal
from skimage.util import view_as_blocks

def optical_flow_fast(I1g, I2g, window_size, tau=1e-2):
    assert I1g.shape == I2g.shape
    assert I1g.shape[0] % window_size == 0 and I1g.shape[1] % window_size == 0

    kernel_x = np.array([[-1., 1.],
                         [-1., 1.]])
    kernel_y = np.array([[-1., -1.],
                         [1., 1.]])
    kernel_t = np.array([[1., 1.],
                         [1., 1.]])  # *.25

    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels

    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode)\
         + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)

    # Taken from https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
    # x.reshape(x.shape[0] // 2, 2, x.shape[1] // 3, 3).swapaxes(1, 2).reshape(-1, 2, 3)
    # torch.unfold my be also used
    fx = view_as_blocks(fx, (window_size, window_size))
    fy = view_as_blocks(fy, (window_size, window_size))
    ft = view_as_blocks(ft, (window_size, window_size))

    sh0, sh1 = fx.shape[0:2]

    A = np.stack([fx.reshape(-1, window_size * window_size),
                  fy.reshape(-1, window_size * window_size)], axis=2)
    b = ft.reshape(-1, window_size * window_size)
    V = np.matmul(np.linalg.pinv(A), b[..., None])

    # Filtering results
    V[np.linalg.eigvals(np.matmul(A.transpose(0, 2, 1), A)).min(axis=1) < tau] = 0

    V = V.reshape(sh0, sh1, 2)
    V[..., 1] *= -1

    return V
