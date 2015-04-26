# -*- coding: utf-8 -*-

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import re
import csv
import numpy as np
import itertools
from collections import Counter
from features.features import featurizer
from util.utils import str2tree, leaves, sick_train_reader, sick_dev_reader, sick_test_reader
import time
import logging
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
try:
    import sklearn
except ImportError:
    sys.stderr.write("scikit-learn version 0.16.* is required\n")
    sys.exit(2)
if sklearn.__version__[:4] != '0.16':
    sys.stderr.write("scikit-learn version 0.16.* is required. You're at %s.\n" % sklearn.__version__)
    sys.exit(2)

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from util.distributedwordreps import build, ShallowNeuralNetwork

LABELS = ['ENTAILMENT', 'CONTRADICTION', 'NEUTRAL']

#
#
# if __name__ == '__main__': # Prevent this example from loading on import of this module.
#
#     print str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")

# if __name__ == '__main__': # Prevent this example from loading on import of this module.
#
#     t = str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")


#TODO: Perform some sort of weighting to feature vector (TF-IDF, PMI,etc.)

def train_classifier(
        reader=sick_train_reader,
        features=None,
        feature_selector=SelectKBest(chi2, k=300), # Use None to stop feature selection
        cv=10, # Number of folds used in cross-validation
        priorlims=np.arange(.1, 3.1, .1)): # regularization priors to explore (we expect something around 1)
    # Featurize the data:
    feats, labels = featurizer(reader=reader, features=features)
    
    # Map the count dictionaries to a sparse feature matrix:
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(feats)

    #TODO: Rewrite below using Pipeline framework

    searchmod_1 = LogisticRegression(fit_intercept=True, intercept_scaling=1)
    kbest_params = [{'k': range(40, 200, 10)}]

    #Feature Selection
    feat_matrix = None
    if feature_selector:
        feat_matrix = feature_selector.fit_transform(X, labels)
    else:
        feat_matrix = X

    classifier_pipe = Pipeline([('dict_vector', vectorizer, 'feature_selector', feature_selector),
        ('clf', searchmod_1)]) #Pipeline for transforming data and training a model
    
    ##### HYPER-PARAMETER SEARCH
    # Define the basic model to use for parameter search:
    searchmod_2 = LogisticRegression(fit_intercept=True, intercept_scaling=1)
    # Parameters to grid-search over:
    parameters = {'C':priorlims, 'penalty':['l1','l2']}  
    # Cross-validation grid search to find the best hyper-parameters:     
    clf = GridSearchCV(searchmod_2, parameters, cv=cv)
    clf.fit(feat_matrix, labels)
    params = clf.best_params_

    # Establish the model we want using the parameters obtained from the search:
    mod = LogisticRegression(fit_intercept=True, intercept_scaling=1, C=params['C'], penalty=params['penalty'])

    ##### ASSESSMENT              
    # Cross-validation of our favored model; for other summaries, use different
    # values for scoring: http://scikit-learn.org/dev/modules/model_evaluation.html
    scores = cross_val_score(mod, feat_matrix, labels, cv=cv, scoring="f1_macro")       
    print 'Best model', mod
    print '%s features selected out of %s total' % (feat_matrix.shape[1], X.shape[1])
    print 'F1 mean: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

    # TRAIN OUR MODEL:
    mod.fit(feat_matrix, labels)

    # Return the trained model along with the objects we need to
    # featurize test data in a way that aligns with our training
    # matrix:
    return (mod, vectorizer, feature_selector, features)





def evaluate_trained_classifier(model=None, reader=sick_dev_reader):
    """Evaluate model, the output of train_classifier, on the data in reader."""
    mod, vectorizer, feature_selector, features = model
    feats, labels = featurizer(reader=reader, features = features)
    feat_matrix = vectorizer.transform(feats)
    if feature_selector:
        feat_matrix = feature_selector.transform(feat_matrix)
    predictions = mod.predict(feat_matrix)
    return metrics.classification_report(labels, predictions)



# #Test Word Overlap
# if __name__ == '__main__': # Prevent this example from loading on import of this module.
#     start_train = time.time()
#     overlapmodel = train_classifier(features=['word_overlap'])
#     end_train = time.time()
#     logging.info("Time to Train Log Reg with Overlap: " + str(end_train - start_train))
#
#     for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
#         print "======================================================================"
#         print readername
#
#         start_test = time.time()
#         print evaluate_trained_classifier(model=overlapmodel, reader=reader)
#         end_test = time.time()
#         logging.info("Time to Evaluate Log Reg with Overlap: " + str(end_test - start_test))
#
#
#
# #Test Cross Product
# if __name__ == '__main__': # Prevent this example from loading on import of this module.
#     start_train = time.time()
#     crossmodel = train_classifier(features=['word_cross_product'])
#     end_train = time.time()
#     logging.info("Time to Train Log Reg with Cross Product: " + str(end_train - start_train))
#
#     for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
#         print "======================================================================"
#         print readername
#
#         start_test = time.time()
#         print evaluate_trained_classifier(model=crossmodel, reader=reader)
#         end_test = time.time()
#         logging.info("Time to Evaluate Log Reg with Cross Product: " + str(end_test - start_test))

#Test Word_Overlap + Cross_Product
start_train = time.time()
crossmodel = train_classifier(features=['word_cross_product'])
end_train = time.time()
logging.info("Time to Train Log Reg with Cross Product + Overlap: " + str(end_train - start_train))

for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
    print "======================================================================"
    print readername

    start_test = time.time()
    print evaluate_trained_classifier(model=crossmodel, reader=reader)
    end_test = time.time()
    logging.info("Time to Evaluate Log Reg with Cross Product + Overlap: " + str(end_test - start_test))



#
#
#
# GLOVE_MAT, GLOVE_VOCAB, _ = build('../distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)
#
#
#
# def glvvec(w):
#     """Return the GloVe vector for w."""
#     i = GLOVE_VOCAB.index(w)
#     return GLOVE_MAT[i]
#
# def glove_features(t):
#     """Return the mean glove vector of the leaves in tree t."""
#     return np.mean([glvvec(w) for w in leaves(t) if w in GLOVE_VOCAB], axis=0)
#
# def vec_concatenate(u, v):
#     return np.concatenate((u, v))
#
# def glove_featurizer(t1, t2):
#     """Combined input vector based on glove_features and concatenation."""
#     return vec_concatenate(glove_features(t1), glove_features(t2))
#
#
#
#
#
# def labelvec(label):
#     """Return output vectors like [1,-1,-1], where the unique 1 is the true label."""
#     vec = np.repeat(-1.0, 3)
#     vec[LABELS.index(label)] = 1.0
#     return vec
#
#
#
#
#
# def data_prep(reader=sick_train_reader, featurizer=glove_featurizer):
#     dataset = []
#     for label, t1, t2 in reader():
#         dataset.append([featurizer(t1, t2), labelvec(label)])
#     return dataset
#
#
#
#
# def train_network(
#         hidden_dim=100,
#         maxiter=1000,
#         reader=sick_train_reader,
#         featurizer=glove_featurizer,
#         display_progress=False):
#     dataset = data_prep(reader=reader, featurizer=featurizer)
#     net = ShallowNeuralNetwork(input_dim=len(dataset[0][0]), hidden_dim=hidden_dim, output_dim=len(LABELS))
#     net.train(dataset, maxiter=maxiter, display_progress=display_progress)
#     return (net, featurizer)
#
#
#
#
# def evaluate_trained_network(network=None, reader=sick_dev_reader):
#     """Evaluate network, the output of train_network, on the data in reader"""
#     net, featurizer = network
#     dataset = data_prep(reader=reader, featurizer=featurizer)
#     predictions = []
#     cats = []
#     for ex, cat in dataset:
#         # The raw prediction is a triple of real numbers:
#         prediction = net.predict(ex)
#         # Argmax dimension for the prediction; this could be done better with
#         # an explicit softmax objective:
#         prediction = LABELS[np.argmax(prediction)]
#         predictions.append(prediction)
#         # Store the gold label for the classification report:
#         cats.append(LABELS[np.argmax(cat)])
#     # Report:
#     print "======================================================================"
#     print metrics.classification_report(cats, predictions, target_names=LABELS)
#
#
#
# #
# # if __name__ == '__main__': # Prevent this example from loading on import of this module.
# #
# #     network = train_network(hidden_dim=10, maxiter=1000, reader=sick_train_reader, featurizer=glove_featurizer)
# #
# #     for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
# #         print "======================================================================"
# #         print readername
# #         evaluate_trained_network(network=network, reader=reader)
#
#
#
#

