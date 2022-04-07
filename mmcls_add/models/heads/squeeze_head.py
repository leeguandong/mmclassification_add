import torch
import torch.nn as nn

from mmcv.cnn import build_conv_layer
from mmcls.models.heads import ClsHead
from mmcls.models import HEADS

@HEADS.register_module()
class SqueezeHead(ClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_rate=0.5,
                 conv_cfg=None,
                 init_cfg=dict(type="Normal", layer="Linear", std=0.1),
                 *args,
                 **kwargs):
        super(SqueezeHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        final_conv = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.num_classes,
            kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(self.drop_rate),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        x = self.pre_logits(x)
        x = self.classifier(x)
        cls_score = x.view(x.size(0), 2)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = torch.max(cls_score, 1)[1] if cls_score is not None else None

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        x = self.classifier(x)
        cls_score = torch.flatten(x, 1)
        losses = self.loss(cls_score, gt_label)
        return losses
