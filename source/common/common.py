import numpy as np
from PIL import Image


def numpy2pil(array):
    return Image.fromarray(np.transpose(array, (1, 2, 0)))
