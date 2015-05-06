__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from features.features import word_cross_product_features, word_overlap_features, hypernym_features
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

def build_log_regression_model(train_data = None, feature_vectorizer = DictVectorizer(sparse=False), features = None, feature_selector = SelectKBest(chi2, k=300)):
    """Builds and returns an instance of a logistic regression model along with necessary.
    Also performs hyperparameter search of appropriate parameters if flag set."""
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer, 'feature_selector', feature_selector)])
