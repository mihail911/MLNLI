__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import numpy as np
import logging
import cPickle as pickle
import time

from features.features import word_cross_product_features, word_overlap_features, hypernym_features, featurizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from util.utils import sick_train_reader, sick_dev_reader
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

def build_log_regression_model(train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse=False), features = None, feature_selector = SelectFpr(chi2, alpha = 0.05), file_name = None):

    """Builds (trains) and returns an instance of a logistic regression model along with necessary.
    Features is a list of feature names to be extracted from the data."""
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), ('clf', LogisticRegression(C=2.4))])

    feat_vec, labels = featurizer(sick_train_reader, features)

    feature_file_name = 'output/' + file_name + '.train.feature'
    label_file_name = 'output/' + file_name + '.train.label'

    print 'Saving feature vector file: ', feature_file_name
    print 'Saving labels file: ', label_file_name

    #Save feature vector to disk
    with open(feature_file_name, 'w') as f:
        pickle.dump(feat_vec, f)
    #Save label file
    with open(label_file_name, 'w') as f:
        pickle.dump(labels, f)

    clf_pipe.fit(feat_vec, labels)
    return clf_pipe, feat_vec, labels

def build_svm_model(train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse=False), features = None, feature_selector = SelectKBest(chi2, k=300), feature_file_name = None):
    """Builds (trains) and returns an instance of a SVM model along with necessary.
    Features is a list of feature names to be extracted from the data."""
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), ('clf', SVC())])
    feat_vec, labels = featurizer(sick_train_reader, features)
    clf_pipe.fit(feat_vec, labels)
    return clf_pipe, feat_vec, labels

def parameter_tune_svm(pipeline = None, feat_vec = None, labels = None):
    """Does hyperparameter tuning of SVM model."""
    parameters = {'clf__C': np.arange(.1, 3.1, .3), 'clf__penalty': ['l1', 'l2'], 'feature_selector__k': np.arange(300,400,100)}

    print "Pipeline steps: ", [step for step, _ in pipeline.steps]
    print "Pipeline parameter grid: ", parameters

    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 10)
    grid_search.fit(feat_vec, labels)

    print "Best score: ", grid_search.best_score_
    best_params = grid_search.best_params_

    print "Best params: ", best_params

    return grid_search.best_estimator_

def parameter_tune_log_reg(pipeline = None, feat_vec = None, labels = None):
    """Does hyperparameter tuning of logistic regression model."""

    parameters = {'clf__penalty':['l1','l2'], 'clf__C': np.arange(2.7,2.0, -0.1)} #'feature_selector__k': np.arange(300,400,100)}

    print "Pipeline steps: ", [step for step, _ in pipeline.steps]
    print "Pipeline parameter grid: ", parameters

    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 10)
    grid_search.fit(feat_vec, labels)

    print "Best score: ", grid_search.best_score_
    best_params = grid_search.best_params_

    print "Best params: ", best_params

    return grid_search.best_estimator_

def evaluate_model(pipeline = None, features = None, file_name = None):
    """Evaluates the given model on data and outputs statistics."""

    for reader_name, reader_func in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        feat_vec, gold_labels = featurizer(reader_func, features)

        dict_vec = pipeline.steps[0][1] #Extracts the dictVectorizer from the pipeline object (assumes vectorizer is first transform applied)
        print reader_name + ' Feature Set Size: ', len(dict_vec.feature_names_)

        feature_file_name = 'output/' + file_name + '.' + reader_name + '.feature'
        label_file_name = 'output/' + file_name + '.' + reader_name + '.label'

        print 'Saving feature vector file: ', feature_file_name
        print 'Saving labels file: ', label_file_name

        #Save feature vector to disk
        with open(feature_file_name + '.' + reader_name + '.feature', 'w') as f:
            pickle.dump(feat_vec, f)
        #Save label file
        with open(feature_file_name + '.dev' + reader_name + '.label', 'w') as f:
            pickle.dump(gold_labels, f)

        predicted_labels = pipeline.predict(feat_vec)
        print metrics.classification_report(gold_labels, predicted_labels)
