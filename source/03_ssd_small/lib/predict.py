
def predict_boxes(input_tensor, i, x, y, b=0):

    assignment = input_tensor[b, 17 * i: 17 * i + 10, x, y]
    predict_x = input_tensor[b, 17 * i + 10, x, y]
    predict_y = input_tensor[b, 17 * i + 11, x, y]
    predict_length = input_tensor[b, 17 * i + 12, x, y]
    predict_width = input_tensor[b, 17 * i + 13, x, y]

    predict_rotate = input_tensor[b, 17 * i + 16, x, y]

    return assignment, predict_x, predict_y, predict_length, predict_width, predict_rotate


def predict_boxes_numpy(input_tensor, i, x, y):

    assignment = input_tensor[17 * i: 17 * i + 10, x, y]
    predict_x = input_tensor[17 * i + 10, x, y]
    predict_y = input_tensor[17 * i + 11, x, y]
    predict_length = input_tensor[17 * i + 12, x, y]
    predict_width = input_tensor[17 * i + 13, x, y]

    predict_rotate = input_tensor[17 * i + 16, x, y]

    return assignment, predict_x, predict_y, predict_length, predict_width, predict_rotate


def predict_boxes_numpy_3d(input_tensor, i, x, y):

    assignment = input_tensor[17 * i: 17 * i + 10, x, y]
    predict_x = input_tensor[17 * i + 10, x, y]
    predict_y = input_tensor[17 * i + 11, x, y]
    predict_length = input_tensor[17 * i + 12, x, y]
    predict_width = input_tensor[17 * i + 13, x, y]
    predict_z = input_tensor[17 * i + 14, x, y]
    predict_height = input_tensor[17 * i + 15, x, y]

    predict_rotate = input_tensor[17 * i + 16, x, y]

    return assignment, predict_x, predict_y, predict_length, predict_width, predict_rotate, predict_z, predict_height


def predict_assignment(input_tensor, i, x, y):
    return input_tensor[17 * i: 17 * i + 10, x, y]
