__author__ = 'mihaileric'

import os,sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

import logging
from models.models import build_log_regression_model, evaluate_model, parameter_tune_log_reg
from argparse import ArgumentParser

parser = ArgumentParser('description = provide arguments for running model pipeline')
parser.add_argument('--conf_file', help = 'name of configuration file ')
arguments = parser.parse_args()

params = {}

#Loads the parameters set in the conf file and saves in a global dict.
with open(arguments.conf_file, 'r') as f:
    model_line = f.readline().split(':')
    params[model_line[0].strip()] = model_line[1].strip()

    features_line = f.readline().split(':')
    values = features_line[1].strip()
    params[features_line[0].strip()] = [feat.strip() for feat in values.split(',')]

    feature_name_line = f.readline().split(':')
    params[feature_name_line[0].strip()] = feature_name_line[1].strip()


print "Training model..."
print "Using features: ", params['features']

#Build model by specifying vectorizer, feature selector, features
model, feat_vec, labels = build_log_regression_model(features = params['features'], file_name = params['feature_file'])
#report = evaluate_model(pipeline = model, features = params['features'], feature_file_name = params['features']) #Returns a classification report
best_model = parameter_tune_log_reg(model, feat_vec, labels)
print evaluate_model(best_model, features = params['features'], file_name = params['feature_file'])

print "Finished training and evaluating model"

