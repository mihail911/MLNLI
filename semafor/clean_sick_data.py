__author__ = 'guthriec'

import csv
from nltk import word_tokenize
from util.utils import sick_train_reader, sick_test_reader, sick_dev_reader, leaves, str2tree

def strip():
    for example in sick_dev_reader():
        yield example[0], ' '.join(leaves(example[1])), ' '.join(leaves(example[2]))

f = open('semafor_dev', 'w')
for ex in strip():
    f.write(ex[1] + '\n')
    f.write(ex[2] + '\n')
f.close()