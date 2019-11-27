import time
import math
import random
from pathlib import Path
from multiprocessing import Process, cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from matplotlib import cm


class Dataset:

    def __init__(self, dataset_type, dir_output, lidar_range, img_size):

        self.dataset_type = dataset_type
        self.dir_output = dir_output
        self.lidar_range = lidar_range
        self.img_size = img_size
        self.level5data = LyftDataset(data_path='../../dataset/{}'.format(dataset_type),
                                      json_path='../../dataset/{}/data/'.format(dataset_type), verbose=True)

    def preprocess(self, token):

        my_sample = self.level5data.get('sample', token)
        # print(my_sample)

        lidar_top = self.level5data.get('sample_data', my_sample['data']['LIDAR_TOP'])

        try:
            pc = LidarPointCloud.from_file(Path('../../dataset/{}'.format(self.dataset_type)) / lidar_top['filename'])
        except Exception:
            print('failed to load: {}'.format(token))
            return

        # print(pc.points.shape)  # x, y, z, intensity

        # brightness

        hist_2d = np.histogram2d(pc.points[0], pc.points[1], bins=self.img_size,
                                 range=[[-self.lidar_range, self.lidar_range],
                                        [-self.lidar_range, self.lidar_range]])
        grid, a, b = hist_2d
        grid_max = 4.0
        grid[grid > grid_max] = grid_max

        brightness = grid / grid_max

        # hue

        z_stacks, _ = np.histogramdd(
            pc.points[:3].T,
            bins=[self.img_size, self.img_size, 60],
            range=[[-self.lidar_range, self.lidar_range],
                   [-self.lidar_range, self.lidar_range],
                   [-2.5, 2.5]])

        z_stacks = z_stacks.transpose(2, 0, 1)

        weighted_sum = np.sum(
            z_stacks * np.tile(np.reshape(np.arange(len(z_stacks)), (-1, 1, 1)), (1, self.img_size, self.img_size)),
            axis=0)
        z_mode_points = weighted_sum / (np.sum(z_stacks, axis=0) + 1e-50)

        c_map = cm.get_cmap('gist_rainbow_r', len(z_stacks))

        hue = c_map((len(z_stacks) - z_mode_points) / len(z_stacks))[:, :, :3]

        # concat

        img = hue * np.stack([brightness, brightness, brightness], 2) * 255.0
        mask = cv2.circle(np.zeros((self.img_size, self.img_size, 3)),
                          (self.img_size // 2, self.img_size // 2), self.img_size // 2, (1, 1, 1), -1)
        img = img * mask

        img = img.astype(np.uint8)
        Image.fromarray(img).save(self.dir_output / '{}.png'.format(token))


def process_job(dataset_type, dir_output, tokens, lidar_range, img_size, p_num):

    dataset = Dataset(dataset_type, dir_output, lidar_range, img_size)

    for token in tqdm(tokens, desc=f'{p_num}'):
        dataset.preprocess(token)


def convert(dir_output: Path, lidar_range: int, is_train: bool):

    dir_output.mkdir(exist_ok=True)

    dataset_type = 'train' if is_train else 'test'

    list_token = list()
    list_x = list()
    list_y = list()
    list_z = list()
    list_length = list()
    list_width = list()
    list_height = list()
    list_rotate = list()
    list_name = list()

    if is_train:
        dataset_df = pd.read_csv('../../dataset/train.csv')
    else:
        dataset_df = pd.read_csv('../../dataset/sample_submission.csv')

    tokens = dataset_df['Id'].tolist()

    # tokens = tokens[:200]

    list_jobs = list()

    num_core = cpu_count()
    random.shuffle(tokens)
    num_batch = -(-len(tokens) // num_core)

    img_size = 2048

    for c in range(num_core):

        p = Process(target=process_job,
                    args=(dataset_type, dir_output, tokens[num_batch*c:num_batch*(c+1)], lidar_range, img_size, c))
        list_jobs.append(p)

    for job in list_jobs:
        job.start()

    for job in list_jobs:
        job.join()

    if is_train:

        level5data = LyftDataset(data_path='../../dataset/{}'.format(dataset_type),
                                 json_path='../../dataset/{}/data/'.format(dataset_type), verbose=True)

        for token in tqdm(tokens):

            my_sample = level5data.get('sample', token)
            my_sample_data = level5data.get('sample_data', my_sample['data']['LIDAR_TOP'])

            try:
                _, boxes, _ = level5data.get_sample_data(my_sample_data['token'])
            except Exception:
                print('failed to load: {}'.format(token))
                continue

            for box in boxes:

                x = round((box.center[0] + lidar_range) * (img_size-1) / (lidar_range * 2))
                y = round((box.center[1] + lidar_range) * (img_size-1) / (lidar_range * 2))
                z = round((box.center[2] + lidar_range) * (img_size-1) / (lidar_range * 2))

                rotate = -box.orientation.yaw_pitch_roll[0]

                if rotate > math.pi / 2:
                    rotate -= math.pi
                if rotate < -math.pi / 2:
                    rotate += math.pi

                width = box.wlh[0] * (img_size-1) / (lidar_range * 2)
                length = box.wlh[1] * (img_size-1) / (lidar_range * 2)
                height = box.wlh[2] * (img_size-1) / (lidar_range * 2)

                list_token.append(token)
                list_x.append(x)
                list_y.append(y)
                list_z.append(z)
                list_length.append(length)
                list_width.append(width)
                list_height.append(height)
                list_rotate.append(rotate)
                list_name.append(box.name)

        result = pd.DataFrame()
        result['token'] = list_token
        result['x'] = list_x
        result['y'] = list_y
        result['z'] = list_z
        result['length'] = list_length
        result['width'] = list_width
        result['height'] = list_height
        result['rotate'] = list_rotate
        result['name'] = list_name

        result.to_csv(dir_output.parent / 'coordinates.csv', index=False, float_format='%.4f')


def main():

    tic = time.time()

    Path('../../input/ds_range100').mkdir(exist_ok=True)
    convert(Path('../../input/ds_range100/train'), 100, True)
    convert(Path('../../input/ds_range100/test'), 100, False)

    Path('../../input/ds_range50').mkdir(exist_ok=True)
    convert(Path('../../input/ds_range50/train'), 50, True)
    convert(Path('../../input/ds_range50/test'), 50, False)

    toc = time.time() - tic
    print('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    main()
