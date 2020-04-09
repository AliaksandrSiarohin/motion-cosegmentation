import matplotlib

matplotlib.use('Agg')
import os
from tqdm import trange
import torch
import yaml

from torch.utils.data import DataLoader
from sync_batchnorm import DataParallelWithCallback

from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from modules.reconstruction_module import ReconstructionModule
from modules.segmentation_module import SegmentationModule
from logger import Logger
from modules.model import FullModel

from frames_dataset import DatasetRepeater
from frames_dataset import FramesDataset


def train(config, reconstruction_module, segmentation_module, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_reconstruction_module = torch.optim.Adam(reconstruction_module.parameters(),
                                                       lr=train_params['lr_reconstruction_module'], betas=(0.5, 0.999))
    optimizer_segmentation_module = torch.optim.Adam(segmentation_module.parameters(),
                                                     lr=train_params['lr_segmentation_module'], betas=(0.5, 0.999))
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, reconstruction_module, segmentation_module,
                    optimizer_reconstruction_module, optimizer_segmentation_module)
    else:
        start_epoch = 0

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['num_workers'], drop_last=True)

    model = FullModel(segmentation_module, reconstruction_module, train_params)

    reconstruction_module_full_par = DataParallelWithCallback(model, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                losses_reconstruction_module, generated = reconstruction_module_full_par(x)

                loss_values = [val.mean() for val in losses_reconstruction_module.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_reconstruction_module.step()
                optimizer_reconstruction_module.zero_grad()
                optimizer_segmentation_module.step()
                optimizer_segmentation_module.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in
                          losses_reconstruction_module.items()}
                logger.log_iter(losses=losses)

            logger.log_epoch(epoch, {'reconstruction_module': reconstruction_module,
                                     'segmentation_module': segmentation_module,
                                     'optimizer_reconstruction_module': optimizer_reconstruction_module,
                                     'optimizer_segmentation_module': optimizer_segmentation_module}, inp=x,
                             out=generated)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
    log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    reconstruction_module = ReconstructionModule(**config['model_params']['reconstruction_module_params'],
                                                 **config['model_params']['common_params'])
    reconstruction_module.to(opt.device_ids[0])
    if opt.verbose:
        print(reconstruction_module)

    segmentation_module = SegmentationModule(**config['model_params']['segmentation_module_params'],
                                             **config['model_params']['common_params'])
    segmentation_module.to(opt.device_ids[0])
    if opt.verbose:
        print(segmentation_module)

    dataset = FramesDataset(is_train=True, **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    print("Training...")
    train(config, reconstruction_module, segmentation_module, opt.checkpoint, log_dir, dataset, opt.device_ids)
