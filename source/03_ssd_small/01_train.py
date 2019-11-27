import os
import argparse
import random
import time
import math
from pathlib import Path
from distutils.util import strtobool
from multiprocessing import Pool
from logging import getLogger

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
from scipy.spatial import cKDTree
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from lib.load_img import SampleDataset
from lib.default_box import dbox_params
from lib.model import build_model
from lib.calc_matching_degree import calc_matching_degree
from lib.visualize import Visualizer
from lib.calc_localization_loss import calc_localization_loss
from lib.calc_classify_loss import calc_classify_loss
from lib.suppression import suppression
from lib.visualize import coord2_img
from lib.log import init_logger

from common import numpy2pil

from tensorboardX import SummaryWriter
writer = SummaryWriter()

random.seed(1048)
np.random.seed(1048)


def arg_min2(v):
    return np.argsort(v)[:2]


def concat_only_tensors(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, tuple):
        return elem_type([*(concat_only_tensors(samples) for samples in zip(*batch))])
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)

    return batch


def name_to_class(name):

    if name == 'car':
        return 0
    elif name == 'other_vehicle':
        return 1
    elif name == 'pedestrian':
        return 2
    elif name == 'bicycle':
        return 3
    elif name == 'truck':
        return 4
    elif name == 'bus':
        return 5
    elif name == 'motorcycle':
        return 6
    elif name == 'animal':
        return 7
    elif name == 'emergency_vehicle':
        return 8
    elif name is None:
        return 9
    else:
        raise KeyError


def explore(inputs):

    b, targets, net_out_batch, tree, positive_iou, negative_iou, distance_upper_bound = inputs

    poss_all = dict()
    negs_all = list()
    ins_ind_all = list()
    target_imgs_cache = dict()

    target_rotates = np.repeat(targets[['rotate']].values, len(dbox_params), 1)
    dbox_rotates = np.repeat(np.expand_dims(dbox_params['rotate_vars'], 0), len(targets), 0)

    angle1 = (target_rotates - dbox_rotates) % math.pi
    angle2 = (dbox_rotates - target_rotates) % math.pi

    ideal_is = np.apply_along_axis(arg_min2, 1, np.minimum(angle1, angle2))

    for l, l_tensor in enumerate(net_out_batch):

        size_x = l_tensor.shape[-2]
        size_y = l_tensor.shape[-1]

        ll = l_tensor.reshape(len(dbox_params), -1, size_x, size_y)
        predict_xy = np.stack([ll[:, 10, :, :].flatten(),  # x
                               ll[:, 11, :, :].flatten()  # y
                               ]).T

        dists, target_indeces = tree.query(predict_xy, distance_upper_bound=distance_upper_bound)

        hit_indeces = target_indeces[dists != np.inf]

        # ins_ind_all.extend(hit_indeces)

        hit_ixy = np.arange(len(dbox_params) * size_x * size_y)[dists != np.inf]
        hit_i = hit_ixy // (size_x * size_y)
        hit_x = hit_ixy % (size_x * size_y) // size_y
        hit_y = hit_ixy % (size_x * size_y) % size_y

        for i, x, y, index in zip(hit_i, hit_x, hit_y, hit_indeces):

            if i not in ideal_is[index]:
                negs_all.append((l, x, y, i, b))
                continue

            if index in target_imgs_cache:
                target_img = target_imgs_cache[index]
            else:
                target_coord = targets.iloc[index]
                target_img = coord2_img(target_coord['x'], target_coord['y'], target_coord['length'],
                                        target_coord['width'], target_coord['rotate']) / 255.0
                target_imgs_cache[index] = target_img

            pred_img = coord2_img(ll[i, 10, x, y],  # x
                                  ll[i, 11, x, y],  # y
                                  ll[i, 12, x, y],  # length
                                  ll[i, 13, x, y],  # width
                                  ll[i, 16, x, y]  # rotate
                                  ) / 255.0

            deg = calc_matching_degree(target_img, pred_img)
            # logger.debug('deg: {:.3f}'.format(deg))

            if deg < negative_iou:
                negs_all.append((l, x, y, i, b))

            elif deg > positive_iou:

                poss_all[(l, x, y, i, b)] = name_to_class(targets.iloc[index]['name'])
                ins_ind_all.append(index)

        non_hit_ixy = np.arange(len(dbox_params) * size_x * size_y)[dists == np.inf]
        non_hit_i = non_hit_ixy // (size_x * size_y)
        non_hit_x = non_hit_ixy % (size_x * size_y) // size_y
        non_hit_y = non_hit_ixy % (size_x * size_y) % size_y

        random_far_negatives = list()

        for i, x, y in zip(non_hit_i, non_hit_x, non_hit_y):
            random_far_negatives.append(((l, x, y, i, b), ll[i, 9, x, y]))

        hard_far_negatives = [p[0] for p in sorted(random_far_negatives, key=lambda x: x[1])][:500]
        negs_all.extend(random.sample(hard_far_negatives, 100))

    negs_all = list(set(negs_all) - set(poss_all.keys()))

    assert len(poss_all) == len(ins_ind_all)

    return poss_all, negs_all, ins_ind_all


def train_main(model, dataset, optimizer, list_train_img, target_classes, num_iter, epoch, distance_upper_bound):

    logger = getLogger('root')

    model.train()

    # load target coord

    dataset = SampleDataset(
        dir_img=f'../../input/{dataset}/train',
        coord_path=f'../../input/{dataset}/coordinates.csv',
        use_augmentation=epoch < 20,
        list_imgs=list_train_img,
        crop_type=3,
        target_classes=target_classes
    )

    # batch size
    batch_size = params.batch_size

    # threshold for positive degree

    positive_iou = float(np.linspace(0.10, 0.60, 20)[min(epoch, 19)])
    negative_iou = float(np.linspace(0.01, 0.40, 20)[min(epoch, 19)])

    assert positive_iou > negative_iou

    loss_cls_ep = 0.0
    loss_loc_ep = 0.0
    num_positive = 0
    num_negative = 0

    sampler = RandomSampler(dataset, num_samples=num_iter*batch_size, replacement=True)

    num_worker = 8

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker,
                             sampler=sampler, collate_fn=concat_only_tensors)

    # start iteration

    for batch_input, target_coords, _ in tqdm(data_loader):

        # load input and target

        batch_targets = list()
        list_trees = list()

        for found_coord in target_coords:

            batch_targets.append(found_coord)
            list_trees.append(cKDTree(found_coord[['x', 'y']].values))

        # forward

        optimizer.zero_grad()
        net_out = model.forward(batch_input.float().cuda())

        net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
        net_out_numpy_batch = [[tensor[b, :, :, :] for tensor in net_out_numpy] for b in range(batch_size)]

        inputs = list()
        for tup in zip(range(batch_size), batch_targets, net_out_numpy_batch, list_trees):
            inputs.append(tup + (positive_iou, negative_iou, distance_upper_bound))

        # don't use multiprocessing
        # boxes = [explore(tup) for tup in inputs]

        # # use multiprocessing
        with Pool(num_worker) as p:
            boxes = list(p.imap(explore, inputs))

        # calc classification loss

        loss_cls = 0.0

        poss_concat = pd.concat([pd.DataFrame(poss_all.keys(), columns=['l', 'x', 'y', 'i', 'b'])
                                 for poss_all, _, _ in boxes], axis=0)
        poss_concat['class'] = pd.concat([pd.Series(list(poss_all.values()), name='class')
                                          for poss_all, _, _ in boxes], axis=0)

        num_positive += len(poss_concat)

        negs_concat = pd.concat([pd.DataFrame(negs_all, columns=['l', 'x', 'y', 'i', 'b'])
                                 for _, negs_all, _ in boxes], axis=0)
        negs_concat['class'] = name_to_class(None)

        num_negative += len(negs_concat)

        class_concat = pd.concat([poss_concat, negs_concat], axis=0)

        for l, net_out_l in enumerate(net_out):
            v = calc_classify_loss(net_out_l, class_concat, l)
            loss_cls = loss_cls + v
            loss_cls_ep += float(v)

        # calc localization loss

        loss_loc = 0.0

        for l, net_out_l in enumerate(net_out):

            v = calc_localization_loss(net_out_l, boxes, l, batch_targets)
            loss_loc = loss_loc + v * 0.1
            loss_loc_ep += float(v)

        loss = loss_cls + loss_loc

        # backward

        loss.backward()
        optimizer.step()

    loss_cls_ep = float(loss_cls_ep / num_iter)
    loss_loc_ep = float(loss_loc_ep / num_iter)

    logger.info('loss: {:.3f} (loss_cls: {:.3f} loss_loc: {:.3f})'.format(
        loss_cls_ep + loss_loc_ep, loss_cls_ep, loss_loc_ep))

    num_positive /= num_iter
    num_negative /= num_iter

    logger.info('num positive ave: {:.3f} num negative ave: {:.3f}'.format(num_positive, num_negative))

    writer.add_scalar('loss/class', loss_cls_ep, epoch)
    writer.add_scalar('loss/location', loss_loc_ep, epoch)

    writer.add_scalar('loss_per/class_per', loss_cls_ep / (num_negative + num_positive), epoch)
    writer.add_scalar('loss_per/location_per', loss_loc_ep / num_positive, epoch)

    writer.add_scalar('ref/num_positive', num_positive, epoch)
    writer.add_scalar('ref/num_negative', num_negative, epoch)

    writer.add_scalar('threshold/positive_iou', positive_iou, epoch)
    writer.add_scalar('threshold/negative_iou', negative_iou, epoch)

    return model


def validate(model, dataset, list_valid_img, target_classes, epoch):

    dir_save = Path('./_valid/ep{}'.format(epoch))

    dir_save.parent.mkdir(exist_ok=True)
    dir_save.mkdir(exist_ok=True)

    model.eval()

    dataset = SampleDataset(
        dir_img=f'../../input/{dataset}/train',
        coord_path=f'../../input/{dataset}/coordinates.csv',
        use_augmentation=epoch < 20,
        list_imgs=list_valid_img,
        crop_type=0,
        target_classes=target_classes
    )

    viz = Visualizer('lib/colors.json')

    for target_name in list_valid_img[:16]:

        img_input, found_coord, original = dataset[target_name]

        writer.add_image('original/{}'.format(target_name[:12]), np.asarray(original), epoch, dataformats='HWC')
        original.save(dir_save / '{}_original.png'.format(target_name[:12]))

        input_tensor = torch.unsqueeze(img_input, 0)
        # input_tensor = augment_input(img_input)

        with torch.no_grad():
            net_out = model.forward(input_tensor.float().cuda())

        net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
        net_out_numpy_batch = [tensor[0, :, :, :] for tensor in net_out_numpy]
        # net_out_numpy_batch = aggregate_output(net_out_numpy)

        img_predicted = viz.draw_predicted_boxes(net_out_numpy_batch, dbox_params, rate=1.0, img_size=original.height)
        writer.add_image('predicted/{}'.format(target_name[:12]), img_predicted, epoch)
        numpy2pil(img_predicted).save(dir_save / '{}_predited.png'.format(target_name[:12]))

        mask = viz.draw_mask_from_coords(found_coord, img_size=original.height)
        writer.add_image('mask/{}'.format(target_name[:12]), mask, epoch)
        numpy2pil(mask).save(dir_save / '{}_mask.png'.format(target_name[:12]))

        list_predicted_img, list_pred_coords = suppression(net_out_numpy_batch, dbox_params, 0.5, 0.3, 0.1)

        pred_coords_df = pd.DataFrame(
            list_pred_coords,
            columns=['x', 'y', 'length', 'width', 'rotate', 'z', 'height', 'class', 'confidence'])

        img_submit = viz.draw_mask_from_coords(pred_coords_df, img_size=original.height)
        writer.add_image('submit/{}'.format(target_name[:12]), img_submit, epoch)
        numpy2pil(img_submit).save(dir_save / '{}_submit_{}.png'.format(target_name[:12], len(list_predicted_img)))


def main():

    if params.debug:
        logger = init_logger('_log/01_train_debug.log', level=10)
    else:
        logger = init_logger('_log/01_train.log', level=20)

    tic = time.time()

    logger.info('parameters')
    logger.info(vars(params))

    num_iter = 100 if params.debug else params.num_iter
    num_epoch = 2 if params.debug else params.num_epoch

    list_train_img_all = os.listdir(f'../../input/{params.dataset}/train')
    random.shuffle(list_train_img_all)

    coords = pd.read_csv(f'../../input/{params.dataset}/coordinates.csv')
    target_classes = params.target_classes.split(',')

    logger.info(f'target_classes: {target_classes}')
    coords = coords.query('name in @target_classes')
    target_imgs = [f'{s}.png' for s in coords['token']]

    list_train_img_all = list(set(list_train_img_all) & set(target_imgs))

    rate_valid = 0.1

    list_train_img = list_train_img_all[:-int(len(list_train_img_all) * rate_valid)]
    list_valid_img = list_train_img_all[-int(len(list_train_img_all) * rate_valid):]

    if params.debug:
        list_train_img = list(list_train_img[:16])
        list_valid_img = list(list_train_img)
    else:
        assert len(set(list_train_img) & set(list_valid_img)) == 0

    logger.info('number of train images: {}'.format(len(list_train_img)))
    logger.info('number of valid images: {}'.format(len(list_valid_img)))

    # build model
    model = build_model()

    # optimizer

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    dir_model = Path('_models')
    dir_model.mkdir(exist_ok=True)

    # train for each epoch

    for ep in range(num_epoch):

        logger.info('')
        logger.info('==> start epoch {}'.format(ep))

        # train
        model = train_main(model, params.dataset, optimizer, list_train_img,
                           target_classes, num_iter, ep, params.distance_upper_bound)

        # validate
        validate(model, params.dataset, list_valid_img, target_classes, epoch=ep)

        # change learning rate
        for param_group in optimizer.param_groups:

            param_group['lr'] *= 0.95
            logger.info('change learning rate into: {:.6f}'.format(param_group['lr']))

        # save model
        torch.save(model, dir_model / 'model_ep{}.pt'.format(ep))

    # save model
    torch.save(model, dir_model / 'model.pt')

    # show elapsed time

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=True, type=strtobool)

    parser.add_argument('--num_iter', type=int, default=1000, help='number of iteration')

    parser.add_argument('--num_epoch', type=int, default=30, help='number of epoch')

    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    parser.add_argument('--dataset', default='ds_range50', type=str, help='dataset')

    parser.add_argument('--distance_upper_bound', default=32, type=int, help='distance upper bound')

    parser.add_argument('--target_classes', default='pedestrian,bicycle,motorcycle,animal',
                        type=str, help='target classes')

    params = parser.parse_args()

    main()
