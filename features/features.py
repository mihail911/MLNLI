__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from collections import Counter
import itertools
from util.utils import sick_train_reader, leaves

def word_overlap_features(t1, t2):
    overlap = [w1 for w1 in leaves(t1) if w1 in leaves(t2)]
    return Counter(overlap)


def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])

features_mapping = {'word_cross_product': word_cross_product_features,
            'word_overlap': word_overlap_features} #Mapping from feature to method that extract given features from sentences

def featurizer(reader=sick_train_reader, features=None):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector."""
    feats = []
    labels = []
    split_index = None
    for label, t1, t2 in reader():
        #print 'label: ', label, 't1: ', t1, 't2: ', t2
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features:
            d = features_mapping[feat](t1, t2)
            feat_dict.update(d)
        #print 'feat_dict: ', feat_dict
        feats.append(feat_dict)
        labels.append(label)
    return (feats, labels)