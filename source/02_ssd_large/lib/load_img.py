import os
import random
import math
from pathlib import Path
from logging import getLogger

import numpy as np
import pandas as pd
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SampleDataset(Dataset):

    def __init__(self, dir_img, coord_path=None, use_augmentation=False, list_imgs=None, crop_type=1,
                 target_classes=None):

        logger = getLogger('root')

        self.dir_img = Path(dir_img)
        self.use_augmentation = use_augmentation
        self.crop_type = crop_type

        if coord_path is not None:
            self.have_coord = True
            self.coord = pd.read_csv(coord_path)
            if target_classes is not None:
                self.coord = self.coord.query('name in @target_classes')
        else:
            self.have_coord = False

        self.to_tensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if list_imgs is None:
            logger.info('(load_img) use all images in {}'.format(str(dir_img)))
            self.list_file_names = sorted(os.listdir(str(self.dir_img)))
        else:
            logger.info('(load_img) use images selected in advance')
            self.list_file_names = list_imgs

    def __len__(self):
        return len(self.list_file_names)

    def __getitem__(self, idx):

        if isinstance(idx, int):
            target_name = self.list_file_names[idx]
        elif isinstance(idx, str):
            target_name = idx
        else:
            raise TypeError('idx must be int or str')

        # load image
        img = Image.open(self.dir_img / target_name)

        if self.have_coord:
            found_coord = self.coord[self.coord['token'] == Path(target_name).stem]
            found_coord = pd.DataFrame(found_coord)
        else:
            found_coord = None

        if self.use_augmentation:

            # try:

            cx_img = 2048/2
            cy_img = 2048/2
            theta = random.random() * 2 * math.pi
            # theta = random.choice([0, math.pi*0.5, math.pi, math.pi*1.5])

            affine_param = (math.cos(theta), -math.sin(theta), cx_img - cx_img * math.cos(theta) + cy_img * math.sin(theta),
                            math.sin(theta), math.cos(theta), cy_img - cx_img * math.sin(theta) - cy_img * math.cos(theta))
            img = img.transform(img.size, Image.AFFINE, affine_param, Image.BILINEAR)

            cx = 2048/2
            cy = 2048/2

            left = np.array([[math.cos(theta), -math.sin(theta), cx - cx * math.cos(theta) + cy * math.sin(theta)],
                             [math.sin(theta), math.cos(theta), cy - cx * math.sin(theta) - cy * math.cos(theta)],
                             [0, 0, 1]])
            right = np.vstack([np.stack([found_coord.x, found_coord.y]), np.ones(len(found_coord))])

            lr = left @ right
            found_coord['x'] = lr[0, :]
            found_coord['y'] = lr[1, :]

            found_coord.loc[:, 'rotate'] -= theta
            found_coord.loc[found_coord.rotate < -math.pi / 2, 'rotate'] += math.pi
            found_coord.loc[found_coord.rotate < -math.pi / 2, 'rotate'] += math.pi

        # -----

        img_size = 2048
        cropped_size = 1024
        margin = (img_size - cropped_size) / 2

        crop_type = self.crop_type
        if self.crop_type == 3:
            if random.random() > 0.5:
                crop_type = 2
            else:
                crop_type = 1

        if crop_type == 1:

            x_left = margin
            y_upper = margin

        elif crop_type == 2:

            x_left = random.randint(0, margin*2-1)
            y_upper = random.randint(0, margin*2-1)

        if crop_type >= 1:

            img = img.crop((y_upper, x_left, y_upper + cropped_size, x_left + cropped_size))

            assert img.height == cropped_size
            assert img.width == cropped_size

            found_coord['x'] = found_coord['x'] - x_left
            found_coord['y'] = found_coord['y'] - y_upper

        img_input = self.normalizer(self.to_tensor(img))

        return img_input.float(), found_coord, img
