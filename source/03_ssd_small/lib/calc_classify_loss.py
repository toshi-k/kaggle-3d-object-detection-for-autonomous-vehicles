from logging import getLogger

import torch
from torch import nn

criteria = nn.CrossEntropyLoss(weight=torch.Tensor([
    2,  # car
    5,  # other_vehicle
    15,  # pedestrian
    15,  # bicycle
    15,  # truck
    30,  # bus
    100,  # motorcycle
    100,  # animal
    100,  # emergency_vehicle
    0.4  # null
]).cuda(), reduction='sum')


def calc_classify_loss(net_out_l, class_concat, l):

    logger = getLogger('root')

    class_concat_target = class_concat.query('l==@l')

    batch_size = net_out_l.shape[0]
    step_x = net_out_l.shape[-2]
    step_y = net_out_l.shape[-1]

    net_out = net_out_l.reshape(batch_size, 4, 17, step_x, step_y)[
        class_concat_target['b'].tolist(),
        class_concat_target['i'].tolist(),
        :10,
        class_concat_target['x'].tolist(),
        class_concat_target['y'].tolist()
    ]
    logger.debug('num sample: {}'.format(len(class_concat_target)))
    target = torch.Tensor(class_concat_target['class'].tolist()).long().cuda()

    v = criteria(net_out, target)

    return v
