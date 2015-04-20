__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.nli import sick_train_reader, word_overlap_features, word_cross_product_features

features_mapping = {'word_cross_product': word_cross_product_features,
            'word_overlap': word_overlap_features} #Mapping from feature to method that extract given features from sentences

def featurizer(reader=sick_train_reader, features=None):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector."""
    feats = []
    labels = []
    split_index = None
    for label, t1, t2 in reader():
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features:
            d = features_mapping[feat](t1, t2)
            feat_dict.update(d)
        feats.append(feat_dict)
        labels.append(label)
    return (feats, labels)