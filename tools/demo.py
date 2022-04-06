'''
@Time    : 2021/8/27 10:12
@Author  : leeguandon@gmail.com
'''
from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=r"D:\smart_banner\style_3\val\cuxiao-renao\197.jpg", help='Image file')
    parser.add_argument('--config', default=r"E:\comprehensive_library\mmclassification\configs\resnet\resnet18_b32x8_imagenet.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default=r"E:\comprehensive_library\mmclassification\weights\resnet18_batch256_imagenet_20200708-34ab8f90.pth",
                        help='Checkpoint file')

    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    print(result)
    # show the results
    show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
