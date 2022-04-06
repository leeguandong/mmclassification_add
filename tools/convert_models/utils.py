'''
@Time    : 2021/8/18 18:00
@Author  : 19045845
'''
import os
import json
import random
import shutil
import codecs
import numpy as np
from tqdm import tqdm


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Encoder, self).default(obj)


def save_json(dst, src):
    returnJson = json.dumps(obj=dst, cls=Encoder)

    file = codecs.open(src, "w", 'utf-8')
    file.write(returnJson)
    file.close()


def split(args):
    subdirs = os.listdir(args.dirpath)

    destdir_test = args.destdir + "/val"
    destdir_train = args.destdir + "/train"

    if not os.path.exists(destdir_test):
        os.mkdir(destdir_test)
    if not os.path.exists(destdir_train):
        os.mkdir(destdir_train)

    for dir in tqdm(subdirs):
        tempdir = args.dirpath + '/' + dir + '/'

        if not os.path.exists(destdir_test + '/' + dir + '/'):
            os.mkdir(destdir_test + '/' + dir + '/')
        if not os.path.exists(destdir_train + '/' + dir + '/'):
            os.mkdir(destdir_train + '/' + dir + '/')

        fs = os.listdir(tempdir)
        random.shuffle(fs)
        le = int(len(fs) * args.ratio)

        for f in fs:
            if f in fs[:le]:
                shutil.copy(tempdir + f, destdir_train + '/' + dir + '/')
            if f in fs[le:]:
                shutil.copy(tempdir + f, destdir_test + '/' + dir + '/')
