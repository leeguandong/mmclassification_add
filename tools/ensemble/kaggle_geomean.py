from __future__ import division
from collections import defaultdict
from glob import glob
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--glob_files', default='method*.csv')
    parser.add_argument('--loc_outfile', default='kaggle_vote.csv')
    args = parser.parse_args()
    return args


def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    if method == "average":
        scores = defaultdict(float)
    with open(loc_outfile, "w") as outfile:
        for i, glob_file in enumerate(glob(glob_files)):
            print("parsing: {}".format(glob_file))
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate(lines):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    if scores[(e, row[0])] == 0:
                        scores[(e, row[0])] = 1
                    scores[(e, row[0])] *= float(row[1])
        for j, k in sorted(scores):
            outfile.write("%s,%f\n" % (k, math.pow(scores[(j, k)], 1 / (i + 1))))
        print("wrote to {}".format(loc_outfile))


if __name__ == "__main__":
    args = parse_args()
    kaggle_bag(args.glob_files, args.loc_outfile)
