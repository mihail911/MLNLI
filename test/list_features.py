__author__ = 'rhyschris'

'''Utility to list possible features available, for writing in the runtime configuration file.'''

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)

class color:
   RED = '\033[91m'
   CYAN = '\033[96m'
   END = '\033[0m'

import features.features as feat

if __name__ == '__main__':
    sys.stdout.write(color.RED + "Possible Features: \n" + '--' * 10 + '\n' + color.CYAN)
    for feature in sorted(feat.features_mapping.keys()):
        sys.stdout.write(feature + '\t')
    sys.stdout.write(color.END + '\n')
