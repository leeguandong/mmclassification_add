'''
@Time    : 2022/7/19 17:23
@Author  : leeguandon@gmail.com
'''
import sys
sys.path.append("/home/ivms/local_disk/mmclassification_add")

import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot

label_dict = {0: 'other', 1: 'furniter_bg'}


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/ivms/local_disk/img_download', help='Image file')
    parser.add_argument('--config', default='/home/ivms/local_disk/mmclassification_add/results/result_resnet18_8xb32/resnet18_8xb32.py', help='Config file')
    parser.add_argument('--checkpoint', default='/home/ivms/local_disk/mmclassification_add/results/result_resnet18_8xb32/latest.pth', help='Checkpoint file')
    parser.add_argument('--results', default='/home/ivms/local_disk')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    for img in Path(args.img).rglob("*.jpg"):
        # test a single image
        result = inference_model(model, str(img))

        Path(os.path.join(args.results, label_dict[0])).mkdir(parents=True, exist_ok=True)
        # Path(os.path.join(args.results, label_dict[1])).mkdir(parents=True, exist_ok=True)
        if label_dict[result['pred_label']] == 'furniter_bg':
            shutil.copy(str(img), os.path.join(args.results, label_dict[0]))


if __name__ == '__main__':
    main()
