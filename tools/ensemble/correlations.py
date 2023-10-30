import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--first_file', default='')
    parser.add_argument('--second_file', default='')
    args = parser.parse_args()
    return args


def corr(first_file, second_file):
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    # assuming first column is `prediction_id` and second column is `prediction`
    prediction = first_df.columns[0]
    # correlation
    print("Finding correlation between: {} and {}".format(first_file, second_file))
    print("Column to be measured: {}".format(prediction))
    print("Pearson's correlation score: {}".format(first_df[prediction].corr(second_df[prediction], method='pearson')))
    print("Kendall's correlation score: {}".format(first_df[prediction].corr(second_df[prediction], method='kendall')))
    print("Spearman's correlation score: {}".format(first_df[prediction].corr(second_df[prediction], method='spearman')))


if __name__ == '__main__':
    args = parse_args()
    corr(args.first_file, args.second_file)
