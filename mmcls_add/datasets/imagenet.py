import numpy as np
from mmcls.datasets.builder import DATASETS
from mmcls.datasets import ImageNet


@DATASETS.register_module()
class ImageNetAdd(ImageNet):
    CLASSES = ["0", "1"]

    def __getitem__(self, idx):
        try:
            return self.prepare_data(idx)
        except:
            return self.__getitem__(np.random.randint(self.__len__()))
