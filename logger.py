import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


def partial_state_dict_load(module, state_dict):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
 
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def load_reconstruction_module(module, checkpoint):
    if 'generator' in checkpoint:
        partial_state_dict_load(module, checkpoint['generator'])
    else:
        module.load_state_dict(checkpoint['reconstruction_module'])


def load_segmentation_module(module, checkpoint):
    if 'kp_detector' in checkpoint:
        partial_state_dict_load(module, checkpoint['kp_detector'])
        module.state_dict()['affine.weight'].copy_(checkpoint['kp_detector']['jacobian.weight'])
        module.state_dict()['affine.bias'].copy_(checkpoint['kp_detector']['jacobian.bias'])
        module.state_dict()['shift.weight'].copy_(checkpoint['kp_detector']['kp.weight'])
        module.state_dict()['shift.bias'].copy_(checkpoint['kp_detector']['kp.bias'])
        if 'semantic_seg.weight' in checkpoint['kp_detector']:
            module.state_dict()['segmentation.weight'].copy_(checkpoint['kp_detector']['semantic_seg.weight'])
            module.state_dict()['segmentation.bias'].copy_(checkpoint['kp_detector']['semantic_seg.bias'])
        else:
            print ('Segmentation part initialized at random.')
    else:
        module.load_state_dict(checkpoint['segmentation_module'])



class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['target'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, reconstruction_module=None, segmentation_module=None,
                 optimizer_reconstruction_module=None, optimizer_segmentation_module=None):
        checkpoint = torch.load(checkpoint_path)
        if reconstruction_module is not None:
            load_reconstruction_module(reconstruction_module, checkpoint)

        if segmentation_module is not None:
            load_segmentation_module(segmentation_module, checkpoint)

        if optimizer_reconstruction_module is not None and 'generator' not in checkpoint :
            optimizer_reconstruction_module.load_state_dict(checkpoint['optimizer_reconstruction_module'])

        if optimizer_segmentation_module is not None and 'generator' not in checkpoint :
            optimizer_segmentation_module.load_state_dict(checkpoint['optimizer_segmentation_module'])

        return 0 if 'generator' in checkpoint else checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, target, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append(source)

        target = target.data.cpu().numpy()
        target = np.transpose(target, [0, 2, 3, 1])
        images.append(target)

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        ## Visibility map
        if 'visibility_map' in out:
            visibility_map = out['visibility_map'].data.cpu().repeat(1, 3, 1, 1)
            visibility_map = F.interpolate(visibility_map, size=source.shape[1:3]).numpy()
            visibility_map = np.transpose(visibility_map, [0, 2, 3, 1])
            images.append(visibility_map)

        if 'segmentation' in out['seg_target']:
            full_mask = []
            full_mask_bin = []

            mask_bin = F.interpolate(out['seg_target']['segmentation'], size=source.shape[1:3], mode='bilinear')
            mask_bin = (torch.max(mask_bin, dim=1, keepdim=True)[0] == mask_bin).float()
 
            for i in range(out['seg_target']['segmentation'].shape[1]):
                mask = out['seg_target']['segmentation'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3], mode='bilinear')
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))
                mask_bin_part = mask_bin[:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)

                mask_bin_part = np.transpose(mask_bin_part.numpy(), (0, 2, 3, 1))
 
                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['seg_target']['segmentation'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                full_mask.append(mask * color)
                full_mask_bin.append(mask_bin_part * color)
            images.append(sum(full_mask))
            images.append(0.3 * target + 0.7 * sum(full_mask))
            images.append(sum(full_mask_bin))
            images.append(0.3 * target + 0.7 * sum(full_mask_bin))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image

