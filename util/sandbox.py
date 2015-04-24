__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from utils import *
from features.features import extract_nouns, extract_noun_synsets, synset_features

num_examples = 15
data_dir = '../nli-data/'
def sick_reader_modified(src_filename):
    count = 0
    for example in csv.reader(file(src_filename), delimiter="\t"):
        if count > num_examples:
            break
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.
            print (label, leaves(str2tree(t1)), leaves(str2tree(t2)))
        count += 1

#sick_reader_modified(data_dir+"SICK_train_parsed.txt")

x = 'dog is cat'
extract_noun_synsets(x)