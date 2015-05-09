__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import numpy as np
import logging
from features.features import word_cross_product_features, word_overlap_features, hypernym_features, featurizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from util.utils import sick_train_reader
from util.utils import sick_train_reader, sick_dev_reader
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

logging.basicConfig(level = logging.INFO)

def build_log_regression_model(train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse=False), features = None, feature_selector = SelectKBest(chi2, k=300)):
    """Builds (trains) and returns an instance of a logistic regression model along with necessary.
    Also performs hyperparameter search of appropriate parameters if flag set.
    Features is a list of feature names to be extracted from the data."""
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), ('clf', LogisticRegression())])
    feat_vec, labels = featurizer(sick_train_reader, features)
    clf_pipe.fit(feat_vec, labels)
    return clf_pipe, feat_vec, labels



def parameter_tune_log_reg(pipeline = None, feat_vec = None, labels = None):
    """Does hyperparameter tuning of logistic regression model."""
    parameters = {'clf__C': np.arange(.1, 3.1, .1), 'clf__penalty': ['l1', 'l2'], 'feature_selector__k': np.arange(100,300,100)}

    print "Pipeline steps: ", [step for step, _ in pipeline.steps]
    print "Pipeline parameter grid: ", parameters

    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 10)
    grid_search.fit(feat_vec, labels)

    print "Best score: ", grid_search.best_score_
    best_params = grid_search.best_params_

    print "Best params: ", best_params

    return grid_search.best_estimator_

def evaluate_model(pipeline = None, reader = sick_dev_reader, features = None):
    """Evaluates the given model on the test data and outputs statistics."""
    feat_vec, gold_labels = featurizer(reader, features)
    predicted_labels = pipeline.predict(feat_vec)
    return metrics.classification_report(gold_labels, predicted_labels)
