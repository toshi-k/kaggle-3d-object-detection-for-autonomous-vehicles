import math
import argparse
import time
from pathlib import Path
from distutils.util import strtobool

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
from tqdm import tqdm

from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset

from lib.log import init_logger


def reverse_box(box, level5data, token):

    # Retrieve sensor & pose records
    sd_record = level5data.get("sample_data", token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    # sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])

    #  Move box to sensor coord system
    box.rotate(Quaternion(cs_record["rotation"]))
    box.translate(np.array(cs_record["translation"]))

    # Move box to ego vehicle coord system
    box.rotate(Quaternion(pose_record["rotation"]))
    box.translate(np.array(pose_record["translation"]))

    return box


def main():

    tic = time.time()

    if params.debug:
        logger = init_logger('_log/03_make_sub_debug.log', level=10)
    else:
        logger = init_logger('_log/03_make_sub.log', level=20)

    level5data = LyftDataset(data_path='../../dataset/test',
                             json_path='../../dataset/test/data/', verbose=True)

    sub_pre = pd.read_csv(params.path_sub_pre)
    print(sub_pre.head())

    sample = pd.read_csv('../../dataset/sample_submission.csv')
    print(sample.head())

    target_tokens = sample['Id']

    if params.debug:
        target_tokens = target_tokens[:20]

    list_subs = list()

    img_size = 2048
    lidar_range = 100

    for i, target_token in enumerate(tqdm(target_tokens)):

        target_subs = sub_pre.query('token==@target_token')

        list_sub_token = list()

        for _, target_sub in target_subs.iterrows():

            x = target_sub['x']
            y = target_sub['y']
            z = target_sub['z']
            length = target_sub['length']
            width = target_sub['width']
            height = target_sub['height']
            rotate = target_sub['rotate']

            width = width * (lidar_range * 2) / (img_size-1)
            length = length * (lidar_range * 2) / (img_size-1)
            height = height * (lidar_range * 2) / (img_size-1)

            x = x * (lidar_range * 2) / (img_size-1) - lidar_range
            y = y * (lidar_range * 2) / (img_size-1) - lidar_range
            z = z * (lidar_range * 2) / (img_size-1) - lidar_range

            rotate = -rotate

            quat = Quaternion(math.cos(rotate / 2), 0, 0, math.sin(rotate / 2))
            print(quat.yaw_pitch_roll)

            pred_box = Box(
                [x, y, z],
                [width, length, height],
                quat
            )

            my_sample = level5data.get('sample', target_token)
            rev_token = level5data.get('sample_data', my_sample['data']['LIDAR_TOP'])['token']

            rev_box = reverse_box(pred_box, level5data, rev_token)

            sub_i = '{:.9f} '.format(target_sub['confidence']) + \
                    ' '.join(['{:.3f}'.format(v) for v in rev_box.center]) + \
                    ' ' + ' '.join(['{:.3f}'.format(v) for v in rev_box.wlh]) + \
                    ' {:.3f}'.format(rev_box.orientation.yaw_pitch_roll[0]) + ' {}'.format(target_sub['name'])

            logger.debug('sub_i')
            logger.debug(sub_i)

            list_sub_token.append(sub_i)

        if len(list_sub_token) == 0:
            sub_token = ''
        else:
            sub_token = ' '.join(list_sub_token)

        logger.info('submit token !')
        logger.info(sub_token)

        list_subs.append(sub_token)

    submission = pd.DataFrame()
    submission['Id'] = target_tokens
    submission['PredictionString'] = list_subs

    dir_sub = Path('_submission')
    dir_sub.mkdir(exist_ok=True)

    submission.to_csv(dir_sub / params.path_sub_pre.name, index=False)

    logger.info('elapsed time: {:.1f} [min]'.format((time.time() - tic) / 60.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        default=False, type=strtobool)

    parser.add_argument('--path_sub_pre',
                        default=Path('_submission_pre/submit_model_th0.5_gth0.3_ov0.1.csv'), type=Path)

    params = parser.parse_args()

    main()
