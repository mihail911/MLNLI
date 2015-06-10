# -*- coding: utf-8 -*-

__author__ = 'mihaileric/chrisbillovits'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import numpy as np
import cPickle as pickle
import time

from features.features import word_cross_product_features, word_overlap_features, hypernym_features, featurizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest, RFE
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB as MultNB
from sklearn.svm import SVC
from util.utils import sick_train_reader, sick_dev_reader, sick_train_dev_reader, sick_test_reader, snli_test_reader, snli_train_reader, snli_dev_reader
from util.colors import color, prettyPrint
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

_models = {"forest" : RandomForestClassifier(n_estimators = 17, criterion = 'entropy', n_jobs = -1),
           "extra_tree" : ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy', n_jobs = -1, max_features = None,
                                              max_depth = 200), 
           "log_reg" : LogisticRegression(), 
           "svm" : SVC(kernel='linear'),
           "naive_bayes" : MultNB(alpha = 1.0, fit_prior = True),
           "linear" : SGDClassifier(loss = 'hinge', alpha = 0.001, n_jobs= -1)
           

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
    """ Loads the feature vector and classification labels from the
        canonical output files in output.  If the file does not exist,
        the load is aborted. 
    """
    
    feat_file_name = 'output/' + file_extension + '.feature'
    label_file_name = 'output/' + file_extension + '.label'
    
    prettyPrint( "Loading feature vectors and labels from disk ... ", color.CYAN)
    if not os.path.isfile(feat_file_name) or not os.path.isfile(label_file_name):
        prettyPrint("Feature vector files {0} could not be found.  Generating from scratch instead ...".format(feat_file_name), color.CYAN)
        return None, None
    with open(feat_file_name, 'r') as f:
        feat_vec = pickle.load(f)
    with open(label_file_name, 'r') as f:
        labels = pickle.load(f)

    prettyPrint ("Done loading feature vectors.", color.CYAN)
    return feat_vec, labels

def obtain_vectors(file_extension = None, load_vec = True, reader = None, features = None):
    ''' Loads feature vectors either from file, or generates them anew.
        If the path doesn't exist, the load_vec flag is ignored. '''
    feat_vec, labels = None, None
    if load_vec:
        feat_vec, labels = load_vectors(file_extension)
    if not load_vec or not feat_vec:
        feat_vec, labels = featurizer(reader, features)
        save_vectors(feat_vec, labels, file_extension)
    return feat_vec, labels

def build_model(clf = "log_reg", train_reader = sick_train_reader, feature_vectorizer = DictVectorizer(sparse = True), 
                             features = None, feature_selector = SelectFpr(chi2, alpha = 0.05), file_name = None, load_vec = None):
    ''' Builds the model of choice. ''' 
    global _models
    ''' Putting RFE in the pipeline '''
    #feature_selector = RFE( LogisticRegression(solver='lbfgs'),
    #                         n_features_to_select = 5000,
    #                         step = 0.05)
    
    clf_pipe = Pipeline([('dict_vector', feature_vectorizer), ('feature_selector', feature_selector), 
                        ('clf', _models[clf])])
    feat_vec, labels = obtain_vectors(file_name, load_vec, train_reader, features)
    
    return clf_pipe, feat_vec, labels

def parameter_tune (model = 'log_reg', pipeline = None, feat_vec = None, labels = None, grid = []):
    ''' Performs parameter tuning for a generalized pipeline model '''

    parameters = grid if grid else _param_grid[model]
    prettyPrint("Tune on grid parameters: {0}".format(parameters), color.GREEN)
    prettyPrint("Pipeline steps: {0}\nPipeline parameter grid: {1}".format([s1 for s1, _ in pipeline.steps],
                                                                        parameters), color.GREEN)
    gridSize = 1
    for key in parameters.keys():
        gridSize *= len(parameters[key])

    if gridSize == 1: # Save time by not cross-validating
        prettyPrint(" *** Skipping cross-validation: only one grid option ***", color.GREEN)
        parameters = {key : val[0] for key, val in parameters.iteritems() }
        pipeline.set_params(**parameters)
        pipeline.fit(feat_vec, labels)
        return pipeline

    grid_search = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 10, n_jobs = -1)
    grid_search.fit(feat_vec, labels)
    prettyPrint( "Best score: {0} \nBest params: {1}".format(grid_search.best_score_,
                                                              grid_search.best_params_) , color.RED)

    return grid_search.best_estimator_

def evaluate_model(pipeline = None, reader = sick_dev_reader, features = None, file_name = "", load_vec = None):
    """Evaluates the given model on the test data and outputs statistics."""
    if reader == sick_dev_reader or reader == snli_dev_reader:
        reader_name = 'Dev'
    elif reader == sick_train_dev_reader:
        reader_name = 'Train + Dev'
    elif reader == sick_test_reader or reader == snli_test_reader:
        reader_name = 'Test'
    elif reader == sick_train_reader or reader == snli_train_reader:
        reader_name = 'Train'

    if len(pipeline.steps) == 2: #Only have a vectorizer and a classifier step in pipeline
        dict_vectorizer = pipeline.steps[0][1]
        print reader_name + ' Feature Set Size: ', len(dict_vectorizer.feature_names_)
    else:
        feature_selector = pipeline.steps[1][1] #Extracts the dictVectorizer from the pipeline object (assumes feature vectorizer is first transform applied)
        print reader_name + ' Feature Set Size: ', len(feature_selector.get_support(True))

    prettyColor = color.RED
    if reader == 'sick_dev_reader':
        reader = sick_dev_reader
        file_name += '.dev'
    elif reader == 'sick_train_dev_reader':
        reader = sick_train_dev_reader
        file_name += '.train_dev'
    elif reader == 'sick_train_reader':
        reader = sick_train_reader
        file_name += '.train'
        prettyColor = color.CYAN
    elif reader == 'snli_train_reader':
        reader = sick_train_reader
        prettyColor = color.CYAN
        file_name += '.train'
    elif reader == 'snli_dev_reader':
        reader = snli_dev_reader
        file_name += '.dev'
    elif reader == 'snli_test_reader':
        reader = snli_test_reader
        file_name += '.test'
    else:
        reader = sick_test_reader
        file_name += '.test'
    feat_vec, gold_labels = obtain_vectors(file_name, load_vec, reader, features)
    
    predicted_labels = pipeline.predict(feat_vec)
    prettyPrint( metrics.classification_report(gold_labels, predicted_labels, digits = 5), prettyColor)

