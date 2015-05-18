__author__ = 'mihaileric'

import os,sys
import re
import collections
import time

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

import logging
from models.models import build_log_regression_model, evaluate_model, parameter_tune_log_reg, save_vectors
from argparse import ArgumentParser
from util.colors import color, prettyPrint

parser = ArgumentParser('description = provide arguments for running model pipeline')
parser.add_argument('--conf_file', help = 'name of configuration file ')
arguments = parser.parse_args()

params = collections.defaultdict(list)
# Loads the parameters set in the conf file and saves in a global dict.
# TODO: regex split a little hacky
with open(arguments.conf_file, 'r') as f:
	stream = f.readlines()
	for line in stream:	
		kv = re.split(r'[ ,:;]*', line.rstrip())
		val = kv[1:] if (len(kv) > 2 or kv[0] == 'features') else kv[1]
		params[kv[0]] = val
# Special-case parsing of arguments
params['load_vectors'] = True if params['load_vectors'].lower() == 'true' else False 

print params

prettyPrint("===" * 16 + "\nTraining model '{0}' ... ".format(params['model']), color.YELLOW)
prettyPrint("With features: {0}".format(params['features']), color.YELLOW)

# Calibrate params 

start_train = time.time()
# Build model by specifying vectorizer, feature selector, features

model, feat_vec, labels = build_log_regression_model(features = params['features'], file_name = params['feature_file'] + ".train", 
													 load_vec = params['load_vectors'])

best_model = parameter_tune_log_reg(model, feat_vec, labels)

end_train = time.time() 
prettyPrint ("Finished training.  Took {0:.2f} seconds".format(end_train - start_train), color.RED)

# Training set , Dev set evaluation
# Training set, we always want to load from disk, because we save to disk immediately beforehand. 
evaluate_model(best_model, reader = 'sick_train_reader', features = params['features'], file_name = params['feature_file'], 
			   load_vec = 'true')

prettyPrint ('===' * 16, color.YELLOW)
evaluate_model(best_model, features = params['features'], file_name = params['feature_file'] + ".dev", load_vec = params['load_vectors'])

prettyPrint("Finished training and evaluating model\n" + "===" * 16, color.YELLOW)

