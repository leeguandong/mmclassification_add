from mmcls.datasets.builder import DATASETS
from mmcls.datasets import ImageNet


@DATASETS.register_module()
class ImageNetAdd(ImageNet):
    CLASSES = ["form", "letter"]
