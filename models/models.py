__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from features.features import word_cross_product_features, word_overlap_features, hypernym_features, featurizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from util.utils import sick_train_reader
from util.utils import sick_train_reader, sick_dev_reader
from sklearn import metrics

def build_log_regression_model(train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse=False), features = None, feature_selector = SelectKBest(chi2, k=300)):
    """Builds (trains) and returns an instance of a logistic regression model along with necessary.
    Also performs hyperparameter search of appropriate parameters if flag set.
    Features is a list of feature names to be extracted from the data."""
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), ('clf', LogisticRegression())])
    feat_vec = featurizer(sick_train_reader, features)
    clf_pipe.fit(feat_vec[0], feat_vec[1])
    return clf_pipe



def parameter_tune_log_reg():
    """Does hyperparameter tuning of logistic regression model."""
    pass

def evaluate_model(pipeline = None, dev_reader = sick_dev_reader, features = None):
    """Evaluates the given model on the test data and outputs statistics."""
    feat_vec, gold_labels = featurizer(sick_dev_reader, features)
    predicted_labels = pipeline.predict(feat_vec)
    print metrics.classification_report(predicted_labels, gold_labels)
