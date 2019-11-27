import numpy as np
from collections import deque

from lib.predict import predict_boxes_numpy_3d, predict_assignment
from lib.visualize import coord2_img


def thresholding(predicted_tensors, dbox_param, threshold=0.7):

    dict_conf = get_confidences(predicted_tensors, dbox_param)
    list_tuple = [key for key, item in dict_conf.items() if item > threshold]

    return list_tuple


def get_confidences(predicted_tensors, dbox_params):

    dict_conf = dict()

    for l, input_tensor in enumerate(predicted_tensors):

        x_step = input_tensor.shape[-2]
        y_step = input_tensor.shape[-1]

        for x in range(x_step):
            for y in range(y_step):
                for i in range(len(dbox_params)):

                    # confidence
                    dict_conf[(l, x, y, i)] = predict_assignment(input_tensor, i, x, y)

    return dict_conf


def calc_overlap(target, all_imgs):
    return np.sum(all_imgs * target) / np.sum(target)


def suppression(predicted_tensors, dbox_params, threshold, threshold2, overlap, img_size=2048):

    dict_conf = get_confidences(predicted_tensors, dbox_params)
    dict_conf = sorted(dict_conf.items(), key=lambda x: x[1][-1])

    list_tuple_candidate = deque([(key, item) for key, item in dict_conf
                                  if item[-1] < threshold  # assignment for null class is less than threshold
                                  ])

    list_tuple = list()
    list_predicted_img = list()
    list_pred_coords = list()

    min_conf = 1.0

    base_img = np.zeros((img_size, img_size), dtype=np.uint8)

    while True:

        if len(list_tuple_candidate) == 0 or len(list_tuple) > 100:
            if min_conf < threshold2:
                return list_predicted_img, list_pred_coords
            else:
                return list(), list()

        pred, conf = list_tuple_candidate.popleft()

        min_conf = min(min_conf, conf[-1])

        predicted_tensor = predicted_tensors[pred[0]]

        assignment, pred_x, pred_y, pred_length, pred_width, pred_rotate, pred_z, pred_height = predict_boxes_numpy_3d(
            predicted_tensor, pred[3], pred[1], pred[2]
        )

        if pred_x < 0 or pred_x >= img_size or pred_y < 0 or pred_y >= img_size:
            continue

        xy = base_img[int(pred_x), int(pred_y)]

        if xy > 0:
            continue

        img_box = coord2_img(pred_x, pred_y, pred_length, pred_width, pred_rotate, img_size)
        degree = calc_overlap(img_box/255.0, base_img/255.0)

        if degree < overlap:
            # when overlap region is small enough
            list_tuple.append(pred)
            list_predicted_img.append(img_box)
            list_pred_coords.append((pred_x, pred_y, pred_length, pred_width, pred_rotate,
                                     pred_z, pred_height, np.argmax(assignment[:-1]), np.max(assignment[:-1])))

            base_img = coord2_img(pred_x, pred_y, pred_length, pred_width, pred_rotate, img_size, img_base=base_img)


def suppression_single(predicted_tensors, dbox_params, threshold, overlap, target_class, img_size):

    dict_conf = get_confidences(predicted_tensors, dbox_params)
    dict_conf = sorted(dict_conf.items(), key=lambda x: x[1][target_class], reverse=True)

    list_tuple_candidate = deque([(key, item) for key, item in dict_conf
                                  if item[target_class] > threshold  # assignment for null class is less than threshold
                                  ])

    if len(list_tuple_candidate) == 0:
        for key, item in dict_conf:
            list_tuple_candidate = deque([(key, item)])
            break  # take only first element

    list_tuple = list()
    list_predicted_img = list()
    list_pred_coords = list()

    base_img = np.zeros((img_size, img_size), dtype=np.uint8)

    while True:

        if len(list_tuple_candidate) == 0 or len(list_tuple) > 100:
            return list_predicted_img, list_pred_coords

        pred, conf = list_tuple_candidate.popleft()

        predicted_tensor = predicted_tensors[pred[0]]

        assignment, pred_x, pred_y, pred_length, pred_width, pred_rotate, pred_z, pred_height = predict_boxes_numpy_3d(
            predicted_tensor, pred[3], pred[1], pred[2]
        )

        if pred_x < 0 or pred_x >= img_size or pred_y < 0 or pred_y >= img_size:
            continue

        xy = base_img[int(pred_x), int(pred_y)]

        if xy > 0:
            continue

        img_box = coord2_img(pred_x, pred_y, pred_length, pred_width, pred_rotate, img_size)
        degree = calc_overlap(img_box/255.0, base_img/255.0)

        if degree < overlap:
            # when overlap region is small enough
            list_tuple.append(pred)
            list_predicted_img.append(img_box)
            list_pred_coords.append((pred_x, pred_y, pred_length, pred_width, pred_rotate, pred_z, pred_height,
                                     target_class))

            base_img = coord2_img(pred_x, pred_y, pred_length, pred_width, pred_rotate, img_size, img_base=base_img)


def suppression_multi(predicted_tensors, dbox_params, threshold, overlap, img_size=2048):

    list_predicted_img = list()
    list_pred_coords = list()

    for target_class in range(9):
        list_predicted_img_target, list_pred_coords_target = suppression_single(
            predicted_tensors, dbox_params, threshold, overlap, target_class, img_size=img_size)

        list_predicted_img.extend(list_predicted_img_target)
        list_pred_coords.extend(list_pred_coords_target)

    return list_predicted_img, list_pred_coords
