__author__ = 'mihaileric'

import os,sys
import re
import collections
import time
import numpy as np

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

import logging
from models.models import *
from argparse import ArgumentParser
from util.colors import color, prettyPrint
from ast import literal_eval as str2dict

from sklearn.feature_selection import SelectFpr, chi2, SelectKBest

parser = ArgumentParser('description = provide arguments for running model pipeline')
parser.add_argument('--conf', help = 'name of configuration file ')
arguments = parser.parse_args()

params = collections.defaultdict(list)
# Loads the parameters set in the conf file and saves in a global dict.
# TODO: regex split a little hacky
with open(arguments.conf, 'r') as f:
	stream = f.readlines()
	for line in stream:	
		kv = re.split(r'[ ,:;]*', line.rstrip())
		val = kv[1:] if (len(kv) > 2 or kv[0] == 'features') else kv[1]

		if kv[0] == 'param_grid': # Need to re-parse the expression
			# Eval to allow for numpy definitions in the config file.
			val = eval( ':'.join( line.split(':')[1 :] ).strip() )  
		
		params[kv[0]] = val
# Special-case parsing of arguments
params['load_vectors'] = True if params['load_vectors'].lower() == 'true' else False 
print params
print 'Configuration file used: ' + arguments.conf

prettyPrint("-" * 80 + "\nTraining model '{0}' ... ".format(params['model']), color.YELLOW)
prettyPrint("With features: {0}".format(params['features']), color.YELLOW)

# Calibrate params 

start_train = time.time()
model, feat_vec, labels = build_model(clf = params['model'], features = params['features'], file_name = params['feature_file'] + ".train",
									  load_vec = params['load_vectors'], feature_selector = SelectKBest(chi2, k = 300))

best_model = parameter_tune(params['model'], model, feat_vec, labels, grid = params['param_grid'])
end_train = time.time() 

prettyPrint ("Finished training.  Took {0:.2f} seconds".format(end_train - start_train), color.RED)

# Training set , Dev set evaluation
evaluate_model(best_model, reader = 'sick_train_reader', features = params['features'], file_name = params['feature_file'], 
			   load_vec = 'true')

prettyPrint ('-' * 80, color.YELLOW)
evaluate_model(best_model, features = params['features'], file_name = params['feature_file'] + ".dev", load_vec = params['load_vectors'])
prettyPrint("Finished training and evaluating model\n" + "-" * 80, color.YELLOW)

