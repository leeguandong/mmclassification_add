'''
@Time    : 2022/10/9 9:39
@Author  : leeguandon@gmail.com
'''
import os.path as osp
import cv2
import mmcv
import numpy as np

from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMaskImageFromFile(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        height, width = img.shape[:2]
        x1 = width * 0.1
        y1 = height * 0.1
        x2 = width * 0.9
        y2 = height * 0.1
        x3 = width * 0.9
        y3 = height * 0.9
        x4 = width * 0.1
        y4 = height * 0.9

        coordinates = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        img = cv2.fillPoly(img, [coordinates], (255, 255, 255))

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
