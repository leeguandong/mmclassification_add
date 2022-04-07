import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer, build_activation_layer,
                      constant_init)
from mmcv import _BatchNorm
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import make_divisible, SELayer


class GhostModule(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN")):
        super(GhostModule, self).__init__()

        self.planes = planes
        init_channels = math.ceil(planes / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential()
        self.cheap_operation = nn.Sequential()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, init_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, new_channels, postfix=2)
        self.primary_conv.add_module("conv1",
                                     build_conv_layer(
                                         conv_cfg,
                                         inplanes,
                                         init_channels,
                                         kernel_size,
                                         stride,
                                         kernel_size // 2,
                                         bias=False))
        self.primary_conv.add_module(self.norm1_name, norm1)
        self.primary_conv.add_module("relu1",
                                     nn.ReLU(inplace=True) if relu else nn.Sequential())

        self.cheap_operation.add_module("conv2",
                                        build_conv_layer(
                                            conv_cfg,
                                            init_channels,
                                            new_channels,
                                            dw_size,
                                            1,
                                            dw_size // 2,
                                            groups=init_channels,
                                            bias=False))
        self.cheap_operation.add_module(self.norm2_name, norm2)
        self.cheap_operation.add_module("relu2",
                                        nn.ReLU(inplace=True) if relu else nn.Sequential())

    @property
    def norm1(self):
        return getattr(self.primary_conv, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self.cheap_operation, self.norm2_name)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.planes, :, :]
        return out


class GhostBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 dw_kernel_size=3,
                 stride=1,
                 se_ratio=0,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN")):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0
        self.stride = stride
        self.with_cp = with_cp

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = build_conv_layer(
                conv_cfg,
                mid_channels,
                mid_channels,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_channels,
                bias=False)
            self.bn_dw_name, self.bn_dw = build_norm_layer(
                norm_cfg, mid_channels, postfix=3)

        # Squeeze-and-excitation
        if has_se:
            self.se = SELayer(mid_channels, ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # shortcut
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        if (in_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module("conv1",
                                     build_conv_layer(
                                         conv_cfg,
                                         in_channels,
                                         in_channels,
                                         dw_kernel_size,
                                         stride=stride,
                                         padding=(dw_kernel_size - 1) // 2,
                                         groups=in_channels,
                                         bias=False))
            self.shortcut.add_module(self.norm1_name, norm1)
            self.shortcut.add_module("conv2",
                                     build_conv_layer(
                                         conv_cfg,
                                         in_channels,
                                         out_channels,
                                         1,
                                         stride=1,
                                         padding=0,
                                         bias=False))
            self.shortcut.add_module(self.norm2_name, norm2)

    @property
    def norm1(self):
        return getattr(self.shortcut, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self.shortcut, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            residual = x
            x = self.ghost1(x)
            if self.stride > 1:
                x = self.conv_dw(x)
                x = self.bn_dw(x)
            if self.se is not None:
                x = self.se(x)
            x = self.ghost2(x)
            x += self.shortcut(residual)

            return x

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class GhostLayer(nn.Sequential):
    def __init__(self,
                 in_channels,
                 block,
                 cfg,
                 width,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN")):
        self.block = block
        self.in_channels = in_channels

        layers = []
        for k, exp_size, c, se_ratio, s in cfg:
            self.exp_size = exp_size
            out_channels = make_divisible(c * width, 4)
            hidden_channels = make_divisible(exp_size * width, 4)
            layers.append(
                block(
                    in_channels=self.in_channels,
                    mid_channels=hidden_channels,
                    out_channels=out_channels,
                    dw_kernel_size=k,
                    stride=s,
                    se_ratio=se_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            self.in_channels = out_channels
        super(GhostLayer, self).__init__(*layers)


@BACKBONES.register_module()
class GhostNet(BaseBackbone):
    """
    Creates a GhostNet Model as defined in:
    GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
    https://arxiv.org/abs/1911.11907
    Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
    """
    arch_setting = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]

    def __init__(self,
                 in_channels=3,
                 width=1.0,
                 drop_rate=0.2,
                 num_stages=5,
                 out_indices=(8,),
                 deep_stem=True,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN"),
                 with_cp=False,
                 norm_eval=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type="Kaiming", layer=["Conv2d"]),
                     dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"])
                 ]):
        super(GhostNet, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.deep_stem = deep_stem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.drop_rate = drop_rate
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        assert max(out_indices) < num_stages * 2
        self.zero_init_residual = zero_init_residual

        # building first layer
        out_channels = make_divisible(16 * width, 4)

        self._make_stem_layer(in_channels, out_channels)

        self.ghost_layers = []
        self.block = GhostBottleneck
        for i, cfg in enumerate(self.arch_setting):
            ghost_layer = self.make_ghost_layer(
                in_channels=out_channels,
                block=self.block,
                cfg=cfg,
                width=width,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            out_channels = ghost_layer.in_channels
            layer_name = f"layer{i+1}"
            self.add_module(layer_name, ghost_layer)
            self.ghost_layers.append(layer_name)

        self._freeze_stages()

    def make_ghost_layer(self, **kwargs):
        return GhostLayer(**kwargs)

    def _make_stem_layer(self, in_channels, out_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i+1}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(GhostNet, self).init_weights()

        if isinstance(self.init_cfg, dict) \
                and self.init_cfg["type"] == "Pretrained":
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, GhostModule):
                    constant_init(m.norm1, 0)
                    constant_init(m.norm2, 0)
                # elif isinstance(m, GhostBottleneck):
                #     constant_init(m.norm1, 0)
                #     constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)

        outs = []
        for i, layer_name in enumerate(self.ghost_layers):
            ghost_layer = getattr(self, layer_name)
            x = ghost_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(GhostNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
