'''
@Time    : 2021/8/19 19:52
@Author  : 19045845
'''
import json
import torch
import argparse
from collections import OrderedDict
from .utils import save_json


def show(src, dst):
    # blobs = torch.load(src, map_location="cpu")["state_dict"]
    blobs = torch.load(src, map_location="cpu")

    f = open(dst, "w")
    for key, weight in blobs.items():
        print(key)
        f.write(key)
        f.write("\n")


def map(dst_mm, dst, map_dir):
    state_dict = OrderedDict()
    f1 = open(dst_mm)
    f2 = open(dst)

    line_mm = f1.readlines()
    line = f2.readlines()

    for key_mm, key in zip(line_mm, line):
        key_mm = key_mm.replace('\n', '').replace('\r', '')
        key = key.replace('\n', '').replace('\r', '')
        state_dict[key] = key_mm

    save_json(state_dict, map_dir)


def convert(src, map, dst):
    blobs = torch.load(src, map_location="cpu")
    state_dict = OrderedDict()
    converted_names = set()
    map = json.load(open(map))

    for key, weight in blobs.items():
        state_dict[map[key]] = weight
        converted_names.add(key)

    for key in blobs:
        if key not in converted_names:
            print(f"not converted:{key}")

    checkpoint = dict()
    checkpoint["state_dict"] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('--src_mm', default=r"E:\comprehensive_library\mmclassification\output\res2net50_ps_test\epoch_1.pth")
    parser.add_argument('--dst_mm_txt', default="res2net50_mmcls.txt", help='save path')
    parser.add_argument("--src", default=r"E:\comprehensive_library\mmclassification\weights\res2net50_v1b_26w_4s-3cf99910.pth")
    parser.add_argument("--dst_txt", default="res2net50_v1b_26w_4s.txt")
    parser.add_argument("--map", default="res2net50_map.json")
    parser.add_argument("--dst", default="res2net50_v1b_26w_4s-3cf99910_mm.pth")
    args = parser.parse_args()
    # show(args.src, args.dst_txt)
    # map(args.dst_mm_txt, args.dst_txt, args.map)
    convert(args.src, args.map, args.dst)


if __name__ == '__main__':
    main()
