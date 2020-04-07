import matplotlib

matplotlib.use('Agg')

import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import sys

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

import torch
from sync_batchnorm import DataParallelWithCallback
import torch.nn.functional as F

from modules.segmentation_module import SegmentationModule
from modules.reconstruction_module import ReconstructionModule
from logger import load_reconstruction_module, load_segmentation_module

from modules.util import AntiAliasInterpolation2d
from modules.dense_motion import DenseMotionNetwork


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


class PartSwapGenerator(ReconstructionModule):
    def __init__(self, blend_scale=1, first_order_motion_model=False, **kwargs):
        super(PartSwapGenerator, self).__init__(**kwargs)
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)

        if first_order_motion_model:
            self.dense_motion_network = DenseMotionNetwork()
        else:
            self.dense_motion_network = None

    def forward(self, source_image, target_image, seg_target, seg_source, blend_mask, use_source_segmentation=False):
        # Encoding of source image
        enc_source = self.first(source_image)
        for i in range(len(self.down_blocks)):
            enc_source = self.down_blocks[i](enc_source)

        # Encoding of target image
        enc_target = self.first(target_image)
        for i in range(len(self.down_blocks)):
            enc_target = self.down_blocks[i](enc_target)

        output_dict = {}
        # Compute flow field for source image
        if self.dense_motion_network is None:
            segment_motions = self.segment_motion(seg_target, seg_source)
            segment_motions = segment_motions.permute(0, 1, 4, 2, 3)
            mask = seg_target['segmentation'].unsqueeze(2)
            deformation = (segment_motions * mask).sum(dim=1)
            deformation = deformation.permute(0, 2, 3, 1)
        else:
            motion = self.dense_motion_network(source_image=source_image, seg_target=seg_target,
                                               seg_source=seg_source)
            deformation = motion['deformation']

        # Deform source encoding according to the motion
        enc_source = self.deform_input(enc_source, deformation)

        if self.estimate_visibility:
            if self.dense_motion_network is None:
                visibility = seg_source['segmentation'][:, 1:].sum(dim=1, keepdim=True) * \
                             (1 - seg_target['segmentation'][:, 1:].sum(dim=1, keepdim=True).detach())
                visibility = 1 - visibility
            else:
                visibility = motion['visibility']

            if enc_source.shape[2] != visibility.shape[2] or enc_source.shape[3] != visibility.shape[3]:
                visibility = F.interpolate(visibility, size=enc_source.shape[2:], mode='bilinear')
            enc_source = enc_source * visibility

        blend_mask = self.blend_downsample(blend_mask)
        # If source segmentation is provided use it should be deformed before blending
        if use_source_segmentation:
            blend_mask = self.deform_input(blend_mask, deformation)

        out = enc_target * (1 - blend_mask) + enc_source * blend_mask

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


def load_checkpoints(config, checkpoint, blend_scale=0.125, first_order_motion_model=False, cpu=False):
    with open(config) as f:
        config = yaml.load(f)

    reconstruction_module = PartSwapGenerator(blend_scale=blend_scale,
                                              first_order_motion_model=first_order_motion_model,
                                              **config['model_params']['reconstruction_module_params'],
                                              **config['model_params']['common_params'])

    if not cpu:
        reconstruction_module.cuda()

    segmentation_module = SegmentationModule(**config['model_params']['segmentation_module_params'],
                                             **config['model_params']['common_params'])
    if not cpu:
        segmentation_module.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)

    load_reconstruction_module(reconstruction_module, checkpoint)
    load_segmentation_module(segmentation_module, checkpoint)

    if not cpu:
        reconstruction_module = DataParallelWithCallback(reconstruction_module)
        segmentation_module = DataParallelWithCallback(segmentation_module)

    reconstruction_module.eval()
    segmentation_module.eval()

    return reconstruction_module, segmentation_module


def load_face_parser(cpu=False):
    from face_parsing.model import BiSeNet

    face_parser = BiSeNet(n_classes=19)
    if not cpu:
       face_parser.cuda()
       face_parser.load_state_dict(torch.load('face_parsing/cp/79999_iter.pth'))
    else:
       face_parser.load_state_dict(torch.load('face_parsing/cp/79999_iter.pth', map_location=torch.device('cpu')))
 
    face_parser.eval()

    mean = torch.Tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).view(1, 3, 1, 1)
    std = torch.Tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).view(1, 3, 1, 1)

    if not cpu:
        face_parser.mean = mean.cuda()
        face_parser.std = std.cuda()
    else:
        face_parser.mean = mean
        face_parser.std = std
 
    return face_parser


def make_video(swap_index, source_image, target_video, reconstruction_module, segmentation_module, face_parser=None,
               hard=False, use_source_segmentation=False, cpu=False):
    assert type(swap_index) == list
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        target = torch.tensor(np.array(target_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        seg_source = segmentation_module(source)

        for frame_idx in tqdm(range(target.shape[2])):
            target_frame = target[:, :, frame_idx]
            if not cpu:
                target_frame = target_frame.cuda()
 
            seg_target = segmentation_module(target_frame)

            # Computing blend mask
            if face_parser is not None:
                blend_mask = F.interpolate(source if use_source_segmentation else target_frame, size=(512, 512))
                blend_mask = (blend_mask - face_parser.mean) / face_parser.std
                blend_mask = torch.softmax(face_parser(blend_mask)[0], dim=1)
            else:
                blend_mask = seg_source['segmentation'] if use_source_segmentation else seg_target['segmentation']

            blend_mask = blend_mask[:, swap_index].sum(dim=1, keepdim=True)
            if hard:
                blend_mask = (blend_mask > 0.5).type(blend_mask.type())

            out = reconstruction_module(source, target_frame, seg_source=seg_source, seg_target=seg_target,
                                        blend_mask=blend_mask, use_source_segmentation=use_source_segmentation)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--target_video", default='sup-mat/source.png', help="path to target video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--swap_index", default="1,2,5", type=lambda x: list(map(int, x.split(','))),
                        help='index of swaped parts')
    parser.add_argument("--hard", action="store_true", help="use hard segmentation labels for blending")
    parser.add_argument("--use_source_segmentation", action="store_true", help="use source segmentation for swaping")
    parser.add_argument("--first_order_motion_model", action="store_true", help="use first order model for alignment")
    parser.add_argument("--supervised", action="store_true",
                        help="use supervised segmentation labels for blending. Only for faces.")

    parser.add_argument("--cpu", action="store_true", help="cpu mode")


    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)

    target_video = imageio.mimread(opt.target_video, memtest=False)
    source_image = resize(source_image, (256, 256))[..., :3]
    target_video = [resize(frame, (256, 256))[..., :3] for frame in target_video]

    blend_scale = (256 / 4) / 512 if opt.supervised else 1
    reconstruction_module, segmentation_module = load_checkpoints(opt.config, opt.checkpoint, blend_scale=blend_scale, 
                                                                  first_order_motion_model=opt.first_order_motion_model, cpu=opt.cpu)

    if opt.supervised:
        face_parser = load_face_parser(opt.cpu)
    else:
        face_parser = None
    predictions = make_video(opt.swap_index, source_image, target_video, reconstruction_module, segmentation_module,
                             face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu)

    # Read fps of the target video and save result with the same fps
    reader = imageio.get_reader(opt.target_video)
    fps = reader.get_meta_data()['fps']
    reader.close()

    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
