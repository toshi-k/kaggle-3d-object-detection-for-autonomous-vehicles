import argparse
import time
from pathlib import Path
from distutils.util import strtobool
from functools import partial
import multiprocessing
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import torch

from lib.load_img import SampleDataset
from lib.default_box import dbox_params
from lib.visualize import Visualizer
from lib.suppression import suppression
from lib.log import init_logger

from common import numpy2pil


def predict_one(inputs, dir_save, threshold, threshold2, overlap):

    viz = Visualizer('lib/colors.json')

    i, target_name, numpy_batch_num = inputs

    list_predicted_img, pred_coords = suppression(numpy_batch_num, dbox_params, threshold, threshold2, overlap)

    sub_i = pd.DataFrame()
    sub_i['token'] = [target_name] * len(pred_coords)
    sub_i_coord = pd.DataFrame(pred_coords,
                               columns=['x', 'y', 'length', 'width', 'rotate', 'z', 'height', 'class', 'confidence'])
    sub_i = pd.concat([sub_i, sub_i_coord], axis=1)

    sub_i['name'] = [viz.names[c] for c in sub_i['class']]

    if i < 50:

        img_predicted = viz.draw_predicted_boxes(numpy_batch_num, dbox_params, rate=1.0, img_size=2048)
        numpy2pil(img_predicted).save(dir_save / '{}_predited_{}.png'.format(i, target_name[:12]))

        img_submit = viz.draw_mask_from_coords(sub_i_coord, img_size=2048)
        numpy2pil(img_submit).save(dir_save / '{}_submit_{}_{}.png'.format(i, target_name[:12], len(list_predicted_img)))

    return sub_i


def predict_main(model_name, dataset_name, threshold, threshold2, overlap):

    tic = time.time()

    # load model
    dir_model = Path('_models')
    model = torch.load(dir_model / model_name)

    list_sample_submission = pd.read_csv('../../dataset/sample_submission.csv')

    list_test_img = list_sample_submission['Id'].tolist()

    if params.debug:
        logger = init_logger('_log/02_detect_debug.log', level=10)
        logger.info('debug mode !')

        list_test_img = list_test_img[:100]  # use only 100 images for dry-run

    else:
        logger = init_logger('_log/02_detect.log', level=20)

    cpu_count = multiprocessing.cpu_count()
    logger.info('num_cpu: {}'.format(cpu_count))

    model.eval()

    list_subs = list()

    dataset = SampleDataset(
        dir_img=f'../../input/{dataset_name}/test',
        coord_path=f'../../input/{dataset_name}/coordinates.csv',
        crop_type=0
    )

    dir_save = Path('./_test')
    dir_save.mkdir(exist_ok=True)

    step = 64

    predict_one_wrap = partial(predict_one,
                               dir_save=dir_save,
                               threshold=threshold,
                               threshold2=threshold2,
                               overlap=overlap)

    for j in tqdm(range(0, len(list_test_img), step), desc='batch loop'):

        list_inputs = list()

        # for i, target_name in enumerate(tqdm(list_test_img)):
        for i in tqdm(range(j, min(j+step, len(list_test_img))), desc='gpu loop'):

            target_name = list_test_img[i]
            img_input, _, original = dataset[f'{target_name}.png']

            input_tensor = torch.unsqueeze(img_input, 0)
            # input_tensor = augment_input(img_input)

            with torch.no_grad():
                net_out = model.forward(input_tensor.cuda())

            net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]

            net_out_numpy_batch = [tensor[0, :, :, :] for tensor in net_out_numpy]
            # net_out_numpy_batch = aggregate_output(net_out_numpy)

            list_inputs.append((i, target_name, net_out_numpy_batch))

            if i < 50:
                original.save(dir_save / '{}_original_{}.png'.format(i, target_name[:12]))

        # list_subs_batch = [predict_one_wrap(ip) for ip in tqdm(list_inputs)]

        with Pool(cpu_count) as p:
            list_subs_batch = list(tqdm(p.imap_unordered(predict_one_wrap, list_inputs),
                                        total=len(list_inputs), desc='nmp loop'))

        list_subs.extend(list_subs_batch)

    submission = pd.concat(list_subs, axis=0)

    # save submission

    dir_submission = Path('_submission_pre')

    dir_submission.mkdir(exist_ok=True)
    submission.to_csv(dir_submission / 'submit_{}_th{}_gth{}_ov{}.csv'.format(
        params.model.stem, params.threshold, params.threshold2, params.overlap),
                      index=False, float_format='%.9f')

    logger.info('elapsed time: {:.1f} [min]'.format((time.time() - tic) / 60.0))


def main():

    predict_main(params.model, params.dataset, params.threshold, params.threshold2, params.overlap)


if __name__ == '__main__':

    # parse arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, type=strtobool)

    parser.add_argument('--model', type=Path, default='model.pt', help='name of model')

    parser.add_argument('--threshold', type=float, default=0.6, help='confidence threshold')

    parser.add_argument('--threshold2', type=float, default=0.6, help='global threshold')

    parser.add_argument('--overlap', type=float, default=0.1, help='maximum overlap degree')

    parser.add_argument('--dataset', default='ds_range50', type=str, help='dataset')

    params = parser.parse_args()

    main()
