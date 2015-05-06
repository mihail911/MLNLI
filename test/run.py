__author__ = 'mihaileric'

import os,sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from models.models import build_log_regression_model
from argparse import ArgumentParser


parser = ArgumentParser('description = provide arguments for running model pipeline')
parser.add_argument('--model', help = 'choose model from among log_reg')



model = build_log_regression_model()
