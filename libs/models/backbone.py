from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from libs.utils.misc import NestedTensor, is_main_process
from libs.models.position_encoding import build_position_encoding

# timm
import timm

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        if name == 'tf_efficientnetv2_s.in21k_ft_in1k':
            self.num_channels = [64,160,256] 
        elif name == 'tf_efficientnetv2_m.in21k_ft_in1k':
            self.num_channels = [80,176,512]
        elif name == 'tf_efficientnetv2_l.in21k_ft_in1k':
            self.num_channels = [96,224,640]
        elif name == 'tf_efficientnetv2_xl.in21k_ft_in1k':
            self.num_channels = [96,256,640]

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        if len(self.num_channels) > 1:
            for layer, x in enumerate(xs):
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out['layer{}'.format(layer)] = NestedTensor(x, mask)
        else:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=xs[-1].shape[-2:]).to(torch.bool)[0]
            out['final_layer'] = NestedTensor(xs[-1], mask)
        return out

# timm
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dropout: float,
                 dilation: bool,
                 pretrained: bool):
        backbone = timm.create_model(name,
                                      features_only=True, out_indices=(2,3,4),
                                      pretrained=pretrained)
        super().__init__(backbone,name,train_backbone)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.TRAIN.LR_BACKBONE > 0
    return_interm_layers = cfg.MODEL.MASKS
    backbone = Backbone(cfg.BACKBONE.NAME, train_backbone, cfg.TRANSFORMER.DROPOUT,
        cfg.BACKBONE.DIALATION, cfg.BACKBONE.PRETRAINED)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
