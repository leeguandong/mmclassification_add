import numpy as np
from mmcls.datasets.builder import DATASETS
from mmcls.datasets import ImageNet


@DATASETS.register_module()
class ImageNetAdd(ImageNet):
    # CLASSES = ["0", "1"]
    # CLASSES = ['简欧', '新古典', '欧式古典', '新中式', '中式', '日式', '美式', '北欧', '地中海', '工业风', '东南亚', '现代', '后现代', '法式', '侘寂', '其它']
    # CLASSES = ["北欧", "侘寂", "地中海", "东南亚", "法式", "港式", "工业风", "后现代", "混搭", "简欧", "美式",
    #            "欧式古典", "日式", "现代", "新古典", "新中式", "中式", "其它"]
    CLASSES = ["北欧", "侘寂", "地中海", "东南亚", "法式", "工业风", "后现代", "简欧", "美式",
               "欧式古典", "轻奢", "日式", "现代", "新古典", "新中式", "中式", "其它"]

    def __getitem__(self, idx):
        try:
            return self.prepare_data(idx)
        except:
            return self.__getitem__(np.random.randint(self.__len__()))
