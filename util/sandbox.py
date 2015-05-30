__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from utils import *
#from features.features import extract_nouns, extract_noun_synsets, phrase_share_feature, subphrase_generator, noun_phrase_modifier_features, noun_phrase_word_vec_features
import nltk
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *

# def subphrases(t1):
#     all_subphrases = []
#     words = leaves(t1)
#     for w in words:
#         all_subphrases += [w]
#     subphrases_helper(all_subphrases, t1)
#     return all_subphrases
#
# def subphrases_helper(phrases, t1):
#     phrases += [t1]
#     for tup in t1:
#         if type(tup) == tuple:
#             subphrases_helper(phrases, tup)
#
num_examples = 100
data_dir = '../nli-data/'
def sick_reader_modified(src_filename):
    count = 0
    for example in csv.reader(file(src_filename), delimiter="\t"):
        if count > num_examples:
            break
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.
            if label == 'ENTAILMENT':
                print (label, str2tree(t1), str2tree(t2))
                print (label, str2tree(t1), str2tree(t2))
                #print subphrase_generator(str2tree(t1))
                #print subphrase_generator(str2tree(t2))
                # print subphrases(str2tree(t1))
                # print subphrases(str2tree(t2))
                print nltk.pos_tag(leaves(str2tree(t1)))
                print nltk.pos_tag(leaves(str2tree(t2)))
                print '=' * 15
                print '=' * 15

        count += 1

sick_reader_modified(data_dir+"SICK_train_parsed.txt")


text = ['A', 'motorcyclist', 'of', 'dog', 'is', 'dangerously', 'riding', 'motorbikes', 'along', 'a', 'roadway']
grammar = """ \
            NN-PHRASE: {<DT.*> <NN> <RB>}
                      {<DT.*> <JJ> <NN>}
                      {<NN> <IN> <NN>}
                      {<RB> <JJ> <NN>}
                      {<DT.*> <NNS> <RB>}
                      {<DT.*> <JJ> <NNS>}
                      {<NNS> <IN> <NNS>}
                      {<RB> <JJ> <NNS>}
                      {<DT.*> <NNP> <RB>}
                      {<DT.*> <JJ> <NNP>}
                      {<NNP> <IN> <NNP>}
                      {<RB> <JJ> <NNP>}
                      {<DT.*> <NNPS> <RB>}
                      {<DT.*> <JJ> <NNPS>}
                      {<NNPS> <IN> <NNP>}
                      {<RB> <JJ> <NNPS>}
            VB-PHRASE : {<RB> <VB>}
                        {<RB> <VBD>}
                        {<RB> <VBG>}
                        {<RB> <VBN>}
                        {<RB> <VBP>}
                        {<RB> <VBZ>}
                        {<RBR> <VB>}
                        {<RBR> <VBD>}
                        {<RBR> <VBG>}
                        {<RBR> <VBN>}
                        {<RBR> <VBP>}
                        {<RBR> <VBZ>}
                        {<RBS> <VB>}
                        {<RBS> <VBD>}
                        {<RBS> <VBG>}
                        {<RBS> <VBN>}
                        {<RBS> <VBP>}
                        {<RBS> <VBZ>}

          """
cp = nltk.RegexpParser(grammar)
tree = cp.parse(nltk.pos_tag(text))

for subtree in tree.subtrees():
    print 'Subtree: ', subtree
    if subtree.label() == 'NN-PHRASE' or subtree.label() == 'VB-PHRASE':
        print subtree.leaves()
        print '-'*10

#x1 = (('A', 'motorcyclist'), ('is', (('riding', (('a', 'motorbike'), 'dangerously')), ('along', ('a', 'roadway')))))
#x2 =  (('A', 'motorcyclist'), ('is', ('riding', (('a', 'motorbike'), ('along', ('a', 'roadway'))))))

# c = noun_phrase_modifier_features(x1, x2)
#noun_phrase_word_vec_features(x1,x2)