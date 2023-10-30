'''
@Time    : 2022/7/20 10:15
@Author  : leeguandon@gmail.com
'''
import os
import shutil
import random
import argparse
from tqdm import tqdm
from pathlib import Path


def split(args):
    subdirs = os.listdir(args.dirpath)

    destdir_test = args.destdir + "/val"
    destdir_train = args.destdir + "/train"

    Path(args.destdir).mkdir(parents=True, exist_ok=True)
    Path(destdir_train).mkdir(parents=True, exist_ok=True)
    Path(destdir_test).mkdir(parents=True, exist_ok=True)

    for dir in tqdm(subdirs):
        tempdir = args.dirpath + '/' + dir + '/'

        Path(destdir_test + '/' + dir + '/').mkdir(parents=True, exist_ok=True)
        Path(destdir_train + '/' + dir + '/').mkdir(parents=True, exist_ok=True)

        fs = os.listdir(tempdir)
        random.shuffle(fs)
        le = int(len(fs) * args.ratio)

        for f in fs:
            if f.split(".ipynb_")[-1] == "checkpoints":
                continue
            if f in fs[:le]:
                shutil.copy(tempdir + f, destdir_train + '/' + dir + '/')
            if f in fs[le:]:
                shutil.copy(tempdir + f, destdir_test + '/' + dir + '/')
            # if f in fs[:le]:
            #     shutil.move(tempdir + f, destdir_train + '/' + dir + '/')
            # if f in fs[le:]:
            #     shutil.move(tempdir + f, destdir_test + '/' + dir + "/")


def parse_args():
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument("--dirpath", default="/home/ivms/local_disk/zhimo_style",
                        help="first level directory, the subdirectories are each category")
    parser.add_argument("--destdir", default="/home/ivms/local_disk/zhimo_style_split")
    parser.add_argument("--ratio", default=0.8)
    args = parser.parse_args([])
    return args


if __name__ == "__main__":
    args = parse_args()
    split(args)
