__author__ = 'rhyschris'

'''Utility to list possible features available, for writing in the runtime configuration file.'''

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)


import features.features as feat
from util.colors import color, prettyPrint


if __name__ == '__main__':
    counter = 0
    
    sys.stdout.write(color.RED + "Possible Features: \n" + '--' * 10 + '\n' + color.CYAN)
    for feature in sorted(feat.features_mapping.keys()):
        terminator = "" if counter % 3 else "\n"
        sys.stdout.write(feature + '\t' + terminator)
        counter += 1
    sys.stdout.write(color.END + '\n')
