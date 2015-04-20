__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from util.nli import word_cross_product_features, word_overlap_features



def build_log_regression_model(train_data = None, hyper_search = False):
    """Builds and returns an instance of a logistic regression model along with necessary.
    Also performs hyperparameter search of appropriate parameters if flag set."""
    pass

