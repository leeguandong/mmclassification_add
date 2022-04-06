'''
@Time    : 2021/8/18 19:48
@Author  : 19045845
'''
import os
import argparse
from .utils import split


def parse_args():
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument("--manner", type=str, default="split", choices=["split"])
    parser.add_argument("--dirpath", default=r"C:\标签上传数据\训练数据\layers",
                        help="first level directory, the subdirectories are each category")
    parser.add_argument("--destdir", default=r"C:\标签上传数据\训练数据")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.manner == "split":
        split(args)
