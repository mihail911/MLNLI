import sys, os
from multiprocessing import Pool, Lock
from copy import deepcopy 
from run import *
from argparse import ArgumentParser

''' Set root directory ''' 
root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

from util.colors import prettyPrint

_console_lock = Lock()

def main(args):
    ''' Runs an ablation study on the features. '''
    params = set_config(args.conf)
    feature_list = [feature for feature in params['features']]
    

    subproc_args = [ deepcopy(params) for i in range(len(feature_list)) ]
    for i in range(len(subproc_args)):
        subproc_args[i]['index'] = i
    ''' -------------------- '''
    proc_pool = Pool(args.mp, initargs = (_console_lock,))
    proc_pool.map(single_ablation, subproc_args)
    prettyPrint("-" * 80, color.YELLOW)
    
    
def single_ablation(params):
    print params
    index = params['index']
    ''' Called by a subprocess '''
    
    feature = params['features'][index]
    params['feature_file'] = 'wo+' + feature
    params['features'] = ['word_overlap', feature]
    
    # params['param_grid']['feature_selector__k'] = ['all'] # Use all features

    _console_lock.acquire()
    prettyPrint("Starting job for word overlap + {0}".format(feature), color.YELLOW)
    _console_lock.release()
        
    old_stdout = pipe_stdout(params['feature_file'])
    results = train_model(params)
    test_model(params, 'dev', *results)

    sys.stdout = old_stdout

    _console_lock.acquire()
    prettyPrint("Done with job for feature set {0}".format(params['feature_file'], color.RED))
    _console_lock.release()
        
def pipe_stdout(file_name):
    ''' Pipes stdout to a file.  Returns the old file stream. '''
    old_stdout = sys.stdout
    sys.stdout = open('output/' + file_name, 'w+')
    return old_stdout


if __name__ == '__main__':
    parser = ArgumentParser('description = provide arguments for running model pipeline')
    parser.add_argument('--conf', help = 'name of configuration file ')
    parser.add_argument('--mp', help = 'Number of processes to spawn')
    arguments = parser.parse_args()
  
    if not arguments.mp or int(arguments.mp) < 1 or int(arguments.mp) > 8:
        prettyPrint("Valid multiprocessing argument not found: defaulting to single process", color.YELLOW) 
        arguments.mp = 1
    arguments.mp = int(arguments.mp)
    main(arguments)






