__author__ = 'mihaileric'

import os
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from collections import Counter
import itertools
from util.utils import sick_train_reader, leaves

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def word_overlap_features(t1, t2):
    overlap = [w1 for w1 in leaves(t1) if w1 in leaves(t2)]
    return Counter(overlap)

def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])

features_mapping = {'word_cross_product': word_cross_product_features,
            'word_overlap': word_overlap_features} #Mapping from feature to method that extracts  given features from sentences

def extract_nouns(sent):
    """Extracts nouns in a given sentence."""
    tokens = word_tokenize(sent)
    pos_tagged = pos_tag(tokens)
    return [word[0] for word in pos_tagged if word[1] == 'NN' or word[1] == 'NNS']

def extract_nouns_lemma(sent):
    """Extracts lemmatized nouns in a given sentence."""
    tokens = word_tokenize(sent)
    pos_tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word[0]) for word in pos_tagged if word[1] == 'NN' or word[1] == 'NNS']

def extract_noun_synsets(sent):
    """Return list of all noun synsets for a given sentence."""
    synsets = []
    all_nouns = extract_nouns(sent)
    for noun in all_nouns:
        synsets.extend(wn.synsets(noun))
    return synsets

def extract_nouns_and_synsets(sent):
    """Extracts pair of both nouns and synsets for those nouns in a given sent."""
    synsets = []
    all_nouns = extract_nouns(sent)
    for noun in all_nouns:
        synsets.extend(wn.synsets(noun))
    return (all_nouns, synsets)



def synset_features(sent1, sent2):
    """Returns counter for all mutual synsets between two sentences."""
    sent1_synsets = extract_noun_synsets(sent1)
    sent2_synsets = extract_noun_synsets(sent2)
    overlap_synsets = [syn for syn in sent1_synsets if syn in sent2_synsets]

    return Counter(overlap_synsets)

def hypernym_features(sent1, sent2):
    """ Calculate hypernyms of sent1 and check if synsets of sent2 contained in
    hypernyms of sent1. Trying to capture patterns of the form
    'A dog is jumping.' entails 'An animal is being active.'
    Returns an indicator feature of form 'contains_hypernyms: True/False'

    TODO: Fix missing simple entail inference such as dog -> dog
    """
    s1_nouns, s1_syns = extract_nouns_and_synsets(sent1)
    s2_syns = extract_noun_synsets(sent2)
    all_hyper_synsets = set([]) #Stores the hypernym synsets of the nouns in the first sentence
    for syn in s1_syns:
        all_hyper_synsets.update(set([i for i in syn.closure(lambda s:s.hypernyms())]))
    synset_overlap = all_hyper_synsets & set(s2_syns) #Stores intersection of sent2 synsets and hypernyms of sent1
    return Counter({'contains_hypernyms:': len(synset_overlap) >= 1})

def antonym_features(sent1, sent2):
    """Use antonyms between sentences to recognize contradiction patterns."""
    #TODO!

def featurizer(reader=sick_train_reader, features_funcs=None):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector."""
    feats = []
    labels = []
    split_index = None
    for label, t1, t2 in reader():
        #print 'label: ', label, 't1: ', t1, 't2: ', t2
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features_funcs:
            d = features_mapping[feat](t1, t2)
            feat_dict.update(d)
        #print 'feat_dict: ', feat_dict
        feats.append(feat_dict)
        labels.append(label)
    return (feats, labels)