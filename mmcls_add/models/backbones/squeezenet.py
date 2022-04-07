
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import build_conv_layer
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES


class Fire(nn.Module):
    def __init__(self,
                 num_blocks,
                 with_cp=False,
                 conv_cfg=None):
        super(Fire, self).__init__()
        self.with_cp = with_cp
        self.inplanes = num_blocks[0]
        squeeze_planes = num_blocks[1]
        expand1x1_planes = num_blocks[2]
        expand3x3_planes = num_blocks[3]

        self.squeeze = build_conv_layer(
            conv_cfg,
            self.inplanes,
            squeeze_planes,
            kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = build_conv_layer(
            conv_cfg,
            squeeze_planes,
            expand1x1_planes,
            kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = build_conv_layer(
            conv_cfg,
            squeeze_planes,
            expand3x3_planes,
            kernel_size=3,
            padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        def _inner_forward(x):
            x = self.squeeze_activation(self.squeeze(x))
            out = torch.cat(
                [self.expand1x1_activation(self.expand1x1(x)),
                 self.expand3x3_activation(self.expand3x3(x))], 1)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class FireLayer(nn.Sequential):
    def __init__(self,
                 stage,
                 version,
                 block,
                 num_blocks,
                 conv_cfg,
                 **kwargs):
        layers = []
        layers.append(
            block(num_blocks,
                  conv_cfg))
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        if version == "1_0":
            if stage == 2:
                layers.append(maxpool)
            elif stage == 6:
                layers.append(maxpool)
        elif version == "1_1":
            if stage == 1:
                layers.append(maxpool)
            elif stage == 3:
                layers.append(maxpool)
        super(FireLayer, self).__init__(*layers)


@BACKBONES.register_module()
class SqueezeNet(BaseBackbone):
    """

    """
    version_setting = {
        "1_0": (Fire, [(96, 16, 64, 64),
                       (128, 16, 64, 64),
                       (128, 32, 128, 128),
                       (256, 32, 128, 128),
                       (256, 48, 192, 192),
                       (384, 48, 192, 192),
                       (384, 64, 256, 256),
                       (512, 64, 256, 256)]),
        "1_1": (Fire, [(64, 16, 64, 64),
                       (128, 16, 64, 64),
                       (128, 32, 128, 128),
                       (256, 32, 128, 128),
                       (256, 48, 192, 192),
                       (384, 48, 192, 192),
                       (384, 64, 256, 256),
                       (512, 64, 256, 256)])}

    def __init__(self,
                 version,
                 in_channels=3,
                 out_indices=(6,),
                 style="pytorch",
                 deep_stem=True,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN"),
                 init_cfg=None):
        super(SqueezeNet, self).__init__(init_cfg)

        if version not in self.version_setting:
            raise KeyError(f"invalid version {version} for squeezenet")
        self.version = version
        self.deep_stem = deep_stem
        self.conv_cfg = conv_cfg
        self.frozen_stages = frozen_stages
        block, stage_blocks = self.version_setting[version]
        self.out_indices = out_indices
        assert max(out_indices) < len(stage_blocks)
        self._make_stem_layer(in_channels)

        self.fire_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            fire_layer = self.make_fire_layer(
                stage=i,
                version=self.version,
                block=block,
                num_blocks=num_blocks,
                conv_cfg=conv_cfg)
            layer_name = f"layer{i+1}"
            self.add_module(layer_name, fire_layer)
            self.fire_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = fire_layer[-1].inplanes

    def make_fire_layer(self, **kwargs):
        return FireLayer(**kwargs)

    def _make_stem_layer(self, in_channels):
        if self.version == "1_0":
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    96,
                    kernel_size=7,
                    stride=2),
                nn.ReLU(inplace=True))
        elif self.version == "1_1":
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    64,
                    kernel_size=3,
                    stride=2),
                nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(SqueezeNet, self).init_weights()

        if isinstance(self.init_cfg, dict) \
                and self.init_cfg["type"] == "Pretrained":
            return

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.fire_layers):
            fire_layer = getattr(self, layer_name)
            x = fire_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(SqueezeNet, self).train(mode)
        self._freeze_stages()
