'''
@Time    : 2021/8/25 14:31
@Author  : leeguandon@gmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_conv_layer
from mmcls.models.builder import NECKS
from mmcls.models.utils import make_divisible


@NECKS.register_module()
class GhostNeck(nn.Module):
    """ghostnet neck.
   """

    def __init__(self,
                 in_channels,
                 exp_size,
                 width=1.0,
                 dropout=0.2,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN")):
        super(GhostNeck, self).__init__()

        self.drop_rate = dropout
        out_channels = make_divisible(exp_size * width, 4)
        self.conv1_neck = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=False,
            inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2_neck = build_conv_layer(
            conv_cfg,
            out_channels,
            1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            outs = self.conv1_neck(inputs)
            outs = self.gap(outs)
            outs = self.conv2_neck(outs)
            outs = self.relu(outs)
            outs = outs.view(inputs.size(0), -1)
            if self.drop_rate > 0:
                outs = F.dropout(outs, p=self.drop_rate, training=self.training)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
