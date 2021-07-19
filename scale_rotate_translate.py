from PIL import Image, ImageDraw, ImageFont
from matplotlib import rc
import numpy as np

# angle of rotation
# center of scale and rotation
# new_center is displacement
# scale is scale
def scale_rotate_translate(image, angle, sr_center=None, displacment=None, scale=None):
    if sr_center is None:
        sr_center = 0, 0
    if displacment is None:
        displacment = 0, 0
    if scale is None:
        scale = 1, 1

    angle = -angle / 180.0 * np.pi

    C = np.array([[1, 0, -sr_center[0]],
                  [0, 1, -sr_center[1]],
                  [0, 0, 1]])

    C_1 = np.linalg.inv(C)

    S = np.array([[1/scale[0], 0, 0],
                  [0, 1/scale[1], 0],
                  [0,        0, 1]])

    R = np.array([[np.cos(angle), np.sin(angle), 0],
                  [-np.sin(angle), np.cos(angle), 0],
                  [0,                         0, 1]])

    D = np.array([[1, 0, displacment[0]],
                  [0, 1, displacment[1]],
                  [0, 0,            1]])

    Mt = np.dot(D, np.dot(C_1, np.dot(R, np.dot(S, C))))

    a, b, c = Mt[0]
    d, e, f = Mt[1]

    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
