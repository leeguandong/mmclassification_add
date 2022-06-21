'''
@Time    : 2022/6/21 14:35
@Author  : leeguandon@gmail.com
'''
# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append("/home/ivms/local_disk/mmclassification_add")

import os.path as osp
from pathlib import Path
from argparse import ArgumentParser
from mmcls_add.utils import list_from_file, list_to_file
from mmcls.apis import inference_model, init_model, show_result_pyplot
from mmcv.utils import ProgressBar


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/ivms/local_disk/phase2/valset2_nolabel.txt', help='Image file')
    parser.add_argument('--img_root', default='/home/ivms/local_disk/phase2/testset1', type=str, help='Image root path')
    parser.add_argument('--config', default='/home/ivms/net_disk_project/19045845/dataclean/cvpr2022/res2net101-w26-s4_8xb32_in1k_orign.py',
                        help='test config file path')
    parser.add_argument('--checkpoint', default='/home/ivms/net_disk_project/19045845/dataclean/cvpr2022/res2net101_epoch100.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    lines = list_from_file(args.img)
    progressbar = ProgressBar(task_num=len(lines))
    f = open("valset2_nolabel.txt", 'w+')
    for line in lines:
        progressbar.update()
        img_path = osp.join(args.img_root, line.strip())
        # test a single image
        result = inference_model(model, img_path)
        if result['pred_label'] == 0:
            result['pred_score'] = 1 - result['pred_score']
        img_name = osp.basename(img_path)
        f.write(img_name)
        f.write(" ")
        f.write(str(result['pred_score']))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
