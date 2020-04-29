import os
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn import linear_model
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from time import gmtime, strftime
import numpy as np
import cv2
import imageio
from skimage import io, img_as_float32

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

from modules.segmentation_module import SegmentationModule
from logger import load_segmentation_module

class FramesDataset(data.Dataset):
    def __init__(self, root_dir, is_train=True, root_dir_masks=None):
        super(FramesDataset, self).__init__()
        self.is_train = is_train
        self.root_dir_masks = root_dir_masks

        train_images, test_images = os.listdir(os.path.join(root_dir, 'train')), os.listdir(os.path.join(root_dir, 'test'))

        if self.is_train:
            self.images = train_images
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.images = test_images
            self.root_dir = os.path.join(root_dir, 'test')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        path = os.path.join(self.root_dir, name)

        video = img_as_float32(io.imread(path))

        out = {}
        image = np.array(video, dtype='float32')

        out['img'] = image.transpose((2, 0, 1)).astype('float32')
        out['name'] = name

        if self.root_dir_masks is not None:
            path_mask = os.path.join(self.root_dir_masks, name)
            mask = img_as_float32(io.imread(path_mask))
            out['mask'] = np.expand_dims(mask, axis=0)

        return out

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T / y_max * 2 - 1.0

    x_map = torch.from_numpy(x_map.astype(np.float32))
    y_map = torch.from_numpy(y_map.astype(np.float32))

    return x_map, y_map

def get_center(part_map):
    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    return x_center, y_center


def sort_column(df):
    df = df.sort_values('file_name')
    return df


def regress_keypoints(df_kp):
    # train dataframes
    train_gnd = sort_column(df_kp['train_gnd'])
    train_pred = sort_column(df_kp['train_pred'])

    # test dataframes
    test_gnd = sort_column(df_kp['test_gnd'])
    test_pred = sort_column(df_kp['test_pred'])

    # convert dataframe to numpy
    train_gnd = np.stack(train_gnd['value'].values)
    train_pred = np.stack(train_pred['value'].values)

    test_gnd = np.stack(test_gnd['value'].values)
    test_pred = np.stack(test_pred['value'].values)

    scores = []
    num_gnd_kp = train_gnd.shape[1]
    for i in range(num_gnd_kp):
        for j in range(2):
            print('Fitting linear model for...{},{}'.format(i, j))
            index = train_gnd[:, i, j] != -1
            features = train_pred[index]
            features = features.reshape(features.shape[0], -1)
            label = train_gnd[index, i, j]
            reg = linear_model.LinearRegression()
            reg.fit(features, label)

            index_test = test_gnd[:, i, j] != -1
            features = test_pred[index_test]
            features = features.reshape(features.shape[0], -1)
            label = test_gnd[index_test, i, j]
            #score = reg.score(features, label) # using sklearn's score
            score = np.mean(np.abs(reg.predict(features) - label))
            scores.append(score)
    print(np.sum(scores))

def evaluate(config, segmentation_module, dataset, opt):
    trainloader = data.DataLoader(dataset['train'], batch_size=1, shuffle=False, drop_last=False)
    testloader = data.DataLoader(dataset['test'], batch_size=1, shuffle=False, drop_last=False)

    # put in eval mode
    segmentation_module.eval()

    size = config['dataset_params']['image_shape'][:2]

    # iterate over train images to obtain predicted keypoints for train set
    print('Computing keypoints on train set. Please wait...')
    # obtain keypoint for train images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []

    train_iter = iter(trainloader)
    with torch.no_grad():
        for _ in trange(len(trainloader.dataset)):

            batch = train_iter.next()

            image = batch['img']
            name = batch['name'][0]

            # get the model output
            output = segmentation_module(image.to(opt.device_ids[0]))['segmentation']
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')

            centers = []
            for j in range(1, output.shape[1]):  # ignore the background
                part_map = output[0, j, ...] + 1e-6
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))

            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)

        # save the landmarks in a pandas dataframe
        kp_train_df = pd.DataFrame(out_df)
        dataset_name_pkl = os.path.basename(opt.config).split('.')[0].split('-')[0]
        kp_train_df.to_pickle('landmarks/' + dataset_name_pkl + '_train_pred.pkl')

    # iterate over the test images to obtain predicted keypoints for test set
    print('Computing keypoints on test set. Please wait...')
    # obtain keypoint for test images
    out_df = {'file_name': [], 'value': []}
    lms_pred = []
    test_iter = iter(testloader)
    with torch.no_grad():
        for _ in trange(len(testloader.dataset)):
            batch = test_iter.next()

            image = batch['img']
            name = batch['name'][0]

            # get the model output
            output = segmentation_module(image.to(opt.device_ids[0]))['segmentation']
            output = F.interpolate(output.cpu(), size=size, mode='bilinear')

            centers = []
            for j in range(1, output.shape[1]):  # ignore the background
                part_map = output[0, j, ...] + 1e-6
                k = part_map.sum()
                part_map_pdf = part_map / k
                x_c, y_c = get_center(part_map_pdf)
                x_c = (x_c + 1.) / 2 * size[0]
                y_c = (y_c + 1.) / 2 * size[0]
                center = torch.stack((x_c, y_c), dim=0).unsqueeze(0)  # compute center of the part map
                centers.append(center)
            centers = torch.cat(centers, dim=0)
            lms_pred.append(centers.unsqueeze(0))

            out_df['value'].append(centers.numpy())
            out_df['file_name'].append(name)

        # save the landmarks in a pandas dataframe
        kp_train_df = pd.DataFrame(out_df)
        kp_train_df.to_pickle('landmarks/' + dataset_name_pkl + '_test_pred.pkl')

        # regress from predicted keypoints to ground truth landmarks
        df_kp = {}
        df_kp['train_gnd'] = pd.read_pickle('landmarks/' + dataset_name_pkl + '_train_gt.pkl')
        df_kp['train_pred'] = pd.read_pickle('landmarks/' + dataset_name_pkl + '_train_pred.pkl')
        df_kp['test_gnd'] = pd.read_pickle('landmarks/' + dataset_name_pkl + '_test_gt.pkl')
        df_kp['test_pred'] = pd.read_pickle('landmarks/' + dataset_name_pkl + '_test_pred.pkl')
        regress_keypoints(df_kp)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
    parser.add_argument("--config", required=True, help="path to the config file")
    parser.add_argument('--mode', default='evaluate')
    parser.add_argument("--root_dir", required=True, help="path to root folder of the train and test images")
    parser.add_argument("--checkpoint_path", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    segmentation_module = SegmentationModule(**config['model_params']['segmentation_module_params'],
                                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        segmentation_module.to(opt.device_ids[0])

    if opt.checkpoint_path is not None:
        checkpoint = torch.load(opt.checkpoint_path)
        load_segmentation_module(segmentation_module, checkpoint)

    dataset = {}
    dataset['train'] = FramesDataset(root_dir=opt.root_dir, is_train=True)
    dataset['test'] = FramesDataset(root_dir=opt.root_dir, is_train=False)

    evaluate(config, segmentation_module, dataset, opt)



