'''
@Time    : 2022/6/21 17:23
@Author  : leeguandon@gmail.com
'''
import numpy as np
import pandas as pd
import os.path as osp
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
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)

    lines = list_from_file(args.img)
    progressbar = ProgressBar(task_num=len(lines))

    led_csv = pd.DataFrame(np.zeros((len(lines), 2)), columns=['image', 'label']).astype(str)

    for index, line in enumerate(lines):
        progressbar.update()
        img_path = osp.join(args.img_root, line.strip())
        # test a single image
        result = inference_model(model, img_path)
        led_csv.loc[index]['image'] = osp.basename(img_path)
        led_csv.loc[index]['label'] = str(result['pred_label'])

    led_csv.to_csv('led_csv.csv', index=0)


if __name__ == "__main__":
    main()
