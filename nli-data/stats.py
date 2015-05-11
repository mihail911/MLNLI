__author__=

import os, sys, re

''' Prints the distribution of classification tags in a train / test set. ''' 

classes = ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]
filename = "SICK_train_parsed.txt"

if __name__ == '__main__':    
    if len(sys.argv) > 1:

        for i, arg in enumerate(sys.argv[1:]):
            if arg == '-f':
                filename = sys.argv[i + 2] # NOT robust
            elif arg == '-c':
                print sys.argv[i + 2:]
                classes = sys.argv[i + 2:]
            elif arg == '-h':
                print "USAGE: python stats.py [-f filename] [-c class_labels]"
                quit()


    print "Reading file: %s" % filename 

    patterns = [re.compile(cl) for cl in classes]
        
    with open(filename, 'r') as f:
        fileString = f.read()
        groupSize = [len(re.findall(pattern, fileString)) for pattern in patterns]

        print 'Classes:'
        for i in xrange(len(classes)):
            
            print "{0} : {1}".format(classes[i], groupSize[i])
    
