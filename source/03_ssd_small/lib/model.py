import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from PIL import Image

from lib.default_box import dbox_params
from lib.visualize import Visualizer

from common import numpy2pil


def set_batch_norm_eval(model):

    bn_count = 0
    bn_training = 0

    for module in model.modules():

        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            if module.training:
                bn_training += 1
            module.eval()
            bn_count += 1

            module.weight.requires_grad = False
            module.bias.requires_grad = False

    print('{} BN modules are set to eval'.format(bn_count))


class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.num_classes = 10
        self.outoput_channel = self.num_classes + 7

        resnet34 = models.resnet34(pretrained=True)

        self.resnet34_main = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3
        )

        self.conv_ex1 = resnet34.layer4

        self.conv_ex2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 512, kernel_size=2, padding=0, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # self.conv_ex3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
        #                               nn.ReLU(inplace=True)
        #                               )

        # self.ex0_intermediate = nn.Conv2d(256, 4 * self.outoput_channel, kernel_size=3, padding=1, stride=1)

        self.ex1_intermediate = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
            nn.Softplus(),
            nn.Conv2d(512, 4 * self.outoput_channel, kernel_size=1, padding=0, stride=1)
        )

        # self.ex2_intermediate = nn.Conv2d(512, 4 * self.outoput_channel, kernel_size=3, padding=1, stride=1)
        # self.ex3_intermediate = nn.Conv2d(256, 32, kernel_size=3, padding=1, stride=1)

    @staticmethod
    def header(h, img_size):

        batch_size = len(h)
        step = img_size / h.shape[-1]
        points = np.arange(step / 2 - 0.5, img_size, step, dtype=np.float32)

        assignment, x, y, length, width, z, height, rotate = torch.split(
            h, [10, 1, 1, 1, 1, 1, 1, 1], dim=2)

        x_points = np.tile(points.reshape(1, 1, 1, h.shape[-1], 1), (batch_size, len(dbox_params), 1, 1, h.shape[-1]))
        y_points = np.tile(points.reshape(1, 1, 1, 1, h.shape[-1]), (batch_size, len(dbox_params), 1, h.shape[-1], 1))

        rotate_vars = dbox_params['rotate_vars'].values
        rotate_vars = np.tile(rotate_vars.reshape(1, len(rotate_vars), 1, 1, 1),
                              (batch_size, 1, 1, h.shape[-1], h.shape[-1]))

        length_shifts = dbox_params['length_shifts'].values
        length_shifts = np.tile(length_shifts.reshape(1, len(length_shifts), 1, 1, 1),
                                (batch_size, 1, 1, h.shape[-1], h.shape[-1]))

        width_shifts = dbox_params['width_shifts'].values
        width_shifts = np.tile(width_shifts.reshape(1, len(width_shifts), 1, 1, 1),
                               (batch_size, 1, 1, h.shape[-1], h.shape[-1]))

        height_shifts = dbox_params['height_shifts'].values
        height_shifts = np.tile(height_shifts.reshape(1, len(height_shifts), 1, 1, 1),
                                (batch_size, 1, 1, h.shape[-1], h.shape[-1]))

        assignment = torch.softmax(assignment, dim=2)  # [batch_size, dbox, channel, x, y]
        x_abs = torch.tanh(x) * step + torch.from_numpy(x_points).cuda()
        y_abs = torch.tanh(y) * step + torch.from_numpy(y_points).cuda()
        z_abs = z + 1010.0

        length_abs = torch.exp(length * 0.1 + math.log2(step) / 1.5) * torch.from_numpy(length_shifts).cuda() + 1
        width_abs = torch.exp(width * 0.1 + math.log2(step) / 1.5) * torch.from_numpy(width_shifts).cuda() + 1
        height_abs = torch.exp(height * 0.1 + math.log2(step) / 1.5) * torch.from_numpy(height_shifts).cuda() + 1
        rotate_abs = torch.atan(rotate) + torch.from_numpy(rotate_vars).cuda()

        return torch.cat([assignment, x_abs, y_abs, length_abs, width_abs, z_abs, height_abs, rotate_abs], dim=2)

    def forward_main(self, x):

        list_output = list()

        main_out = self.resnet34_main.forward(x)

        ex1_down = F.relu(self.conv_ex1(main_out))
        ex2_down = self.conv_ex2(ex1_down)

        ex1_up = self.conv_up2(ex2_down)

        ex1_out = torch.cat([ex1_down, ex1_up], 1)
        ex1_branch = self.ex1_intermediate(ex1_out)  # 24x24
        list_output.append(ex1_branch)

        return list_output

    def forward(self, x):

        list_output = list()

        list_main = self.forward_main(x)

        for out in list_main:
            size = out.shape[-1]
            h = self.header(out.reshape(-1, 4, self.outoput_channel, size, size), img_size=x.shape[-1])
            list_output.append(h.reshape(-1, 4 * self.outoput_channel, size, size))

        return list_output


def build_model():

    model = Model()
    model.cuda()
    return model


if __name__ == '__main__':

    dir_debug = Path('_debug')
    dir_debug.mkdir(exist_ok=True)

    model = build_model()
    print(model)

    viz = Visualizer('colors.json')

    # 768 x 768

    in_arr1 = np.zeros((2, 3, 768, 768), dtype=np.float32)
    in_tensor1 = torch.from_numpy(in_arr1)

    out_vars1 = model.forward(in_tensor1.cuda())

    [print(out_var.shape) for out_var in out_vars1]

    out_var_numpy1 = [tensor.cpu().data.numpy() for tensor in out_vars1]
    out_var_numpy_batch1 = [[tensor[b, :, :, :] for tensor in out_var_numpy1] for b in range(2)]

    img = viz.draw_predicted_boxes(out_var_numpy_batch1[0], dbox_params, img_size=in_arr1.shape[-1])
    numpy2pil(img).save(dir_debug / 'sample_1-0.png')

    img = viz.draw_predicted_boxes(out_var_numpy_batch1[1], dbox_params, img_size=in_arr1.shape[-1])
    numpy2pil(img).save(dir_debug / 'sample_1-1.png')

    # 1024 x 1024

    in_arr2 = np.zeros((2, 3, 1024, 1024), dtype=np.float32)
    in_tensor2 = torch.from_numpy(in_arr2)

    out_vars2 = model.forward(in_tensor2.cuda())

    [print(out_var.shape) for out_var in out_vars2]

    out_var_numpy2 = [tensor.cpu().data.numpy() for tensor in out_vars2]
    out_var_numpy_batch2 = [[tensor[b, :, :, :] for tensor in out_var_numpy2] for b in range(2)]

    img = viz.draw_predicted_boxes(out_var_numpy_batch2[0], dbox_params, img_size=in_arr2.shape[-1])
    numpy2pil(img).save(dir_debug / 'sample_2-0.png')

    img = viz.draw_predicted_boxes(out_var_numpy_batch2[1], dbox_params, img_size=in_arr2.shape[-1])
    numpy2pil(img).save(dir_debug / 'sample_2-1.png')
