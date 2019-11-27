import math
import numpy as np
import pandas as pd

import torch
from torch import nn

from lib.default_box import dbox_params


def calc_localization_loss(net_out_l, boxes, l, batch_targets):

    list_target = list()
    list_x = list()
    list_y = list()
    list_b = list()
    list_i = list()

    for b, (poss_all, _, target_indeces) in enumerate(boxes):

        pos_l = [pos[0] == l for pos in poss_all]
        target_pos_l = batch_targets[b].iloc[target_indeces].loc[pos_l]
        list_target.append(target_pos_l[['x', 'y', 'length', 'width', 'z', 'height', 'rotate']])

        list_x.extend([pos[1] for pos in poss_all if pos[0] == l])
        list_y.extend([pos[2] for pos in poss_all if pos[0] == l])
        list_i.extend([pos[3] for pos in poss_all if pos[0] == l])
        list_b.extend([b] * len(target_pos_l))

    loc_target = pd.concat(list_target, axis=0)

    step_x = net_out_l.shape[-2]
    step_y = net_out_l.shape[-1]

    outs = torch.reshape(net_out_l, (len(net_out_l), len(dbox_params), -1, step_x, step_y))[
           list_b,
           list_i,
           10:,  # x, y, length, width, z, height, rotate
           list_x,
           list_y,
           ]

    target_rotate = torch.from_numpy(loc_target.iloc[:, 6].values.astype(np.float32)).cuda()

    angle1 = (target_rotate - outs[:, 6]) % math.pi
    angle2 = (outs[:, 6] - target_rotate) % math.pi

    # angle1 = (target_rotate - outs[:, 6]) % (math.pi * 2)
    # angle2 = (outs[:, 6] - target_rotate) % (math.pi * 2)

    v_rotate = nn.SmoothL1Loss(reduction='sum')(torch.min(angle1, angle2) * 2.0, torch.zeros(len(angle1)).cuda()) * 0.5

    v_else = nn.SmoothL1Loss(reduction='sum')(
        outs[:, :6], torch.from_numpy(loc_target.iloc[:, :6].values.astype(np.float32)).cuda())

    return v_else + v_rotate
