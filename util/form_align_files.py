__author__ = 'mihaileric'

"""Forms example data files of SICK data set appropriate
for input to Berkeley aligner."""

import os
import sys


"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import csv
import codecs
from utils import *


path_to_data = '../nli-data'
file_name = ['SICK_train_parsed.txt', 'SICK_dev_parsed.txt', 'SICK_test_parsed.txt']

for f_name in file_name:
    with open(path_to_data + '/' + f_name, 'r') as f:
        with codecs.open(path_to_data + '/' + f_name + ".e", 'w', encoding='utf8') as e_file: #Source file
            with codecs.open(path_to_data + '/' + f_name + ".f", 'w', encoding='utf8') as f_file: #Target file
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    if not line[0].startswith('%'):
                        ex_e = line[1]
                        ex_f = line[2]
                        e_file.write(' '.join(leaves(str2tree(ex_e))) + '\n')
                        f_file.write(' '.join(leaves(str2tree(ex_f))) + '\n')
