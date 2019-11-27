import math
import numpy as np
import pandas as pd


x_shifts = 0
y_shifts = 0
length_shifts = 2.5
width_shifts = 1.0
rotate_vars = np.linspace(-math.pi / 2.0, math.pi / 2, 5)[1:] - math.pi / 8
# rotate_vars = np.linspace(-math.pi, math.pi, 5)[1:] - math.pi / 4
height_shifts = 1.0

img_size = 768

dbox_params = np.stack(np.meshgrid(x_shifts, y_shifts, length_shifts, width_shifts, rotate_vars, height_shifts), -1)
dbox_params = dbox_params.reshape(-1, 6).astype(np.float32)
dbox_params = pd.DataFrame(
    dbox_params,
    columns=['x_shifts', 'y_shifts', 'length_shifts', 'width_shifts', 'rotate_vars', 'height_shifts']
)

print(dbox_params)
