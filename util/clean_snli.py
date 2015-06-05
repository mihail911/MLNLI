__author__ = 'mihaileric'

"""Script to clean up SNLI dataset."""

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import csv

path_to_data = '../nli-data'
file_name = ['snli_1.0rc3_dev.txt', 'snli_1.0rc3_train.txt', 'snli_1.0rc3_test.txt']

for f_name in file_name:
    with open(path_to_data + '/' + f_name, 'r') as f:
        with open(path_to_data + '/' + 'clean_' + f_name, 'w') as f2:
            reader = csv.DictReader(f, delimiter='\t')
            for line in reader:
                f2.write(line['gold_label'].upper() + '\t' + line['sentence1_binary_parse'] +
                    '\t' + line['sentence2_binary_parse'] + '\t' + line['sentence1'] + '\t' + line['sentence2'] + '\n')
