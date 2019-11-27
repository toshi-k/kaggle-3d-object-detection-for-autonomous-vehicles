import numpy as np


def calc_matching_degree(img1, img2):

    area_intersect = np.sum(img1 * img2)
    area_union = np.sum(img1) + np.sum(img2) - area_intersect

    if area_union < 1e-5:
        return 0

    matching_degree = area_intersect / area_union
    return matching_degree
