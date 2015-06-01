# -*- coding: utf-8 -*-

__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import numpy as np
import cPickle as pickle
import time

from features.features import word_cross_product_features, word_overlap_features, hypernym_features, featurizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB as MultNB
from sklearn.svm import SVC
from util.utils import sick_train_reader, sick_dev_reader
from util.colors import color, prettyPrint
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

_models = {"forest" : RandomForestClassifier(n_estimators = 17, criterion = 'entropy'),
           "log_reg" : LogisticRegression(), 
           "svm" : SVC(kernel='linear'),
           "naive_bayes" : MultNB(alpha = 1.0, fit_prior = True),
           "linear" : SGDClassifier(loss = 'hinge', alpha = 0.001, n_jobs= -1)
           # n_jobs = -1 is necessary for an sgdclassifier

           }

_param_grid = { "forest" : {},
                "log_reg" : {'clf__C': np.arange(2.0 ,0.5, -0.5)}, #'feature_selector__k': np.arange(1500, 3000, 300)}, 

               "svm" : {'clf__C': np.arange(.8, 2.1, .3), 'feature_selector__k': np.arange(300,400,200)},

               "naive_bayes" : {'clf__alpha': [1.0, 1.25, 1.5, 1.7, 2.0], 
                                'feature_selector__alpha' : [0.03, 0.04,
                                0.05]},
                "linear" : {'clf__alpha' : [0.01, 0.02, 0.03, 0.04, 0.05],
                                'feature_selector__k' : [300, 500, 750,
                                1000, 1500, 2000, 4000]}
                                                                
               }

def save_vectors (feat_vec = None, labels = None, file_extension = None):
    """ Saves the feature vectors and classification labels under the given file extension.  """

    feat_file_name = 'output/' + file_extension + '.feature'
    label_file_name = 'output/' + file_extension + '.label'

    prettyPrint('Saving feature vector file: {0} ... \n'
                'Saving Labels file: {1} ... '.format(feat_file_name, label_file_name), color.CYAN)

    #Save feature vector to disk
    with open(feat_file_name, 'w') as f:
        pickle.dump(feat_vec, f)
    #Save label file
    with open(label_file_name, 'w') as f:
        pickle.dump(labels, f)

def load_vectors (file_extension = None):
    """ Loads the feature vector and classification labels from the canonical output files in output/ """

    feat_file_name = 'output/' + file_extension + '.feature'
    label_file_name = 'output/' + file_extension + '.label'

    prettyPrint( "Loading feature vectors and labels from disk ... ", color.CYAN)
    with open(feat_file_name, 'r') as f:
        feat_vec = pickle.load(f)
    with open(label_file_name, 'r') as f:
        labels = pickle.load(f)
    prettyPrint ("Done loading feature vectors.", color.CYAN)
    return feat_vec, labels


def build_model(clf = "log_reg", train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse = True), 
                             features = None, feature_selector = SelectFpr(chi2, alpha = 0.05), file_name = None, load_vec = None):
    ''' Builds the model of choice. ''' 
    global _models

    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), 
                        ('clf', _models[clf])])

    feat_vec, labels = load_vectors (file_name) if load_vec else featurizer(train_reader, features)
    if not load_vec:    
        save_vectors(feat_vec, labels, file_extension = file_name)

    clf_pipe.fit(feat_vec, labels)
    return clf_pipe, feat_vec, labels

def parameter_tune (model = 'log_reg', pipeline = None, feat_vec = None, labels = None, grid = []):
    ''' Performs parameter tuning for a generalized pipeline model '''

    parameters = grid if grid else _param_grid[model]
    prettyPrint("Tune on grid parameters: {0}".format(parameters), color.GREEN)
    prettyPrint("Pipeline steps: {0}\nPipeline parameter grid: {1}".format([s1 for s1, _ in pipeline.steps],
                                                                        parameters), color.GREEN)

    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 10, n_jobs = -1)
    grid_search.fit(feat_vec, labels)
    prettyPrint( "Best score: {0} \nBest params: {1}".format(grid_search.best_score_,
                                                              grid_search.best_params_) , color.RED)

    return grid_search.best_estimator_

def evaluate_model(pipeline = None, reader = sick_dev_reader, features = None, file_name = "", load_vec = None):
    """Evaluates the given model on the test data and outputs statistics."""
    if reader == sick_dev_reader:
        reader_name = 'Dev'
    else:
        reader_name = 'Train'

    dict_vec = pipeline.steps[0][1] #Extracts the dictVectorizer from the pipeline object (assumes vectorizer is first transform applied)

    #Note this is only the actual feature set size if no feature selection/reduction happens!
    print reader_name + ' Feature Set Size: ', len(dict_vec.feature_names_)

    prettyColor = color.RED
    if reader == 'sick_dev_reader':
        reader = sick_dev_reader
        file_name += ".dev"
    elif reader == 'sick_train_reader':
        reader = sick_train_reader
        file_name += ".train"
        prettyColor = color.CYAN

    feat_vec, gold_labels = load_vectors (file_name) if load_vec else featurizer(reader, features)
    if not load_vec:
        save_vectors (feat_vec, gold_labels, file_name)

    predicted_labels = pipeline.predict(feat_vec)
    prettyPrint( metrics.classification_report(gold_labels, predicted_labels), prettyColor)

