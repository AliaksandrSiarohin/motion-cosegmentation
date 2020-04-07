import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.util import make_coordinate_grid


class ReconstructionModule(nn.Module):
    """
    Reconstruct target from source and segmentation of target and part transformations
    """

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, num_segments, estimate_visibility=False, **kwargs):
        super(ReconstructionModule, self).__init__()

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_visibility = estimate_visibility
        self.num_channels = num_channels
        self.num_segments = num_segments

    def segment_motion(self, seg_target, seg_source):
        bs, _, h, w = seg_target['segmentation'].shape
        identity_grid = make_coordinate_grid((h, w), type=seg_source['shift'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - seg_target['shift'].view(bs, self.num_segments, 1, 1, 2)
        if 'affine' in seg_target:
            affine = torch.matmul(seg_source['affine'], torch.inverse(seg_target['affine']))
            affine = affine.unsqueeze(-3).unsqueeze(-3)
            affine = affine.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        target_to_source = coordinate_grid + seg_source['shift'].view(bs, self.num_segments, 1, 1, 2)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        return torch.cat([identity_grid, target_to_source], dim=1)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, seg_target, seg_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        output_dict = {}

        # Computing segment motion
        segment_motions = self.segment_motion(seg_target, seg_source)
        segment_motions = segment_motions.permute(0, 1, 4, 2, 3)
        mask = seg_target['segmentation'].unsqueeze(2)
        deformation = (segment_motions * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        # Deform according to motion
        out = self.deform_input(out, deformation)
        output_dict["deformed"] = self.deform_input(source_image, deformation)

        if self.estimate_visibility:
            visibility = seg_source['segmentation'][:, 1:].sum(dim=1, keepdim=True) * \
                         (1 - seg_target['segmentation'][:, 1:].sum(dim=1, keepdim=True).detach())
            visibility = 1 - visibility

            if out.shape[2] != visibility.shape[2] or out.shape[3] != visibility.shape[3]:
                visibility = F.interpolate(visibility, size=out.shape[2:], mode='bilinear')
            out = out * visibility
            output_dict['visibility_maps'] = visibility

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
