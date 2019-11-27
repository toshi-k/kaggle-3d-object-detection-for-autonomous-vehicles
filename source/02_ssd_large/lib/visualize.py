import json
import random
from functools import partial

import numpy as np
from skimage.draw import polygon, polygon_perimeter

from lib.predict import predict_boxes_numpy


def coord_draw(func, mean_x, mean_y, length, width, rotate, img_size=768, img_base=None, weight=1.0, color=None):

    if img_base is None:
        if color is None:
            # gray scale image
            img = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            # color image
            img = np.zeros((3, img_size, img_size), dtype=np.uint8)
    else:
        img = img_base

    if np.isnan(mean_x) and np.isnan(mean_y):
        return img

    mean_x = float(mean_x)
    mean_y = float(mean_y)
    length = float(length)
    width = float(width)
    rotate = float(rotate)

    W2 = np.array([[np.cos(rotate), np.sin(rotate)], [-np.sin(rotate), np.cos(rotate)]])

    c = np.array([[-length/2, -width/2], [length/2, -width/2], [length/2, width/2], [-length/2, width/2]])
    c = (W2 @ c.T).T + np.array([mean_x, mean_y])

    try:
        rr, cc = func(c[:, 0], c[:, 1], shape=(img_size, img_size))
        if color is None:
            img[rr, cc] = np.maximum(img[rr, cc].flatten(), int(255 * weight))
        else:
            img[0, rr, cc] = color[0] * weight
            img[1, rr, cc] = color[1] * weight
            img[2, rr, cc] = color[2] * weight
    except:
        raise RuntimeError('error in drawing')

    return img


def coord2_img(mean_x, mean_y, length, width, rotate, img_size=768, img_base=None, color=None):

    return coord_draw(polygon, mean_x, mean_y, length, width, rotate, img_size=img_size, img_base=img_base, color=color)


def coord2_boarder(mean_x, mean_y, length, width, rotate, img_size, img_base=None, weight=1.0, color=None):

    polygon_perimeter_clip = partial(polygon_perimeter, clip=True)

    return coord_draw(polygon_perimeter_clip, mean_x, mean_y, length, width, rotate,
                      img_size=img_size, img_base=img_base, weight=weight, color=color)


class Visualizer:

    def __init__(self, path_json):

        with open(path_json) as f:
            self.color_palette = json.load(f)

        self.names = list(self.color_palette.keys())
        self.colors = list(self.color_palette.values())

    def draw_mask_from_coords(self, coords, img_size=768, boarder=False):

        if boarder:
            draw_func = coord2_boarder
        else:
            draw_func = coord2_img

        img_boxes = np.zeros((3, img_size, img_size), dtype=np.uint8)

        for i in range(len(coords)):

            coord = coords.iloc[i]

            if 'class' in coords:
                color = self.colors[int(coord['class'])]
            else:
                color = self.color_palette[coord['name']]

            img_boxes = draw_func(
                coord['x'],
                coord['y'],
                coord['length'],
                coord['width'],
                coord['rotate'],
                img_size=img_size,
                img_base=img_boxes,
                color=color
            )

        return img_boxes

    def draw_predicted_boxes(self, predicted_tensors, dbox_params, img_size=768, rate=1.0, list_tuple=None):

        img_boxes = np.zeros((3, img_size, img_size), dtype=np.uint8)

        num_error = 0

        for l, input_tensor in enumerate(predicted_tensors):

            step = img_size / input_tensor.shape[2]

            x_points = np.arange(step / 2 - 0.5, img_size, step)
            y_points = np.arange(step / 2 - 0.5, img_size, step)

            for x, x_point in enumerate(x_points):
                for y, y_point in enumerate(y_points):
                    for i in range(len(dbox_params)):

                        if random.random() > rate:
                            continue

                        if (list_tuple is not None) and ((l, x, y, i) not in list_tuple):
                            continue

                        assignment, predict_x, predict_y, pr_length, pr_width, pr_rotate = predict_boxes_numpy(
                            input_tensor, i, x, y
                        )

                        if float(assignment[-1]) > 0.99:
                            continue

                        pred_class = np.argmax(assignment[:-1])

                        try:
                            img_boxes = coord2_boarder(
                                predict_x, predict_y, pr_length, pr_width, pr_rotate, img_size,
                                img_base=img_boxes, weight=assignment[pred_class], color=self.colors[pred_class]
                            )
                        except:
                            num_error += 1

        if num_error > 0:
            print('error in drawing (x{})'.format(num_error))

        return img_boxes


if __name__ == '__main__':

    viz = Visualizer('colors.json')
    print(viz.color_palette)
    print(viz.colors)
