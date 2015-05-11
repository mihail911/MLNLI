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
        synsets.extend(wn.synsets(noun, pos=wn.NOUN))
    return synsets

def extract_nouns_and_synsets(sent):
    """Extracts pair of both nouns and synsets for those nouns in a given sent."""
    synsets = []
    all_nouns = extract_nouns(sent)
    for noun in all_nouns:
        synsets.extend(wn.synsets(noun, pos=wn.NOUN))
    return (all_nouns, synsets)

def noun_synset_dict(sent):
    synsets = {}
    all_nouns = extract_nouns(sent)
    for noun in all_nouns:
        synsets[noun] = wn.synsets(noun, pos=wn.NOUN)
    return synsets

def extract_adj_lemmas(sent):
    """Extracts all adjectives in a given sentence"""
    lemmas = []
    tokens = word_tokenize(sent)
    pos_tagged = pos_tag(tokens)
    for word in pos_tagged:
        if word[1] in ['JJ', 'JJR', 'JJS']:
            lemmas.extend(wn.lemmas(word[0], wn.ADJ))
    return lemmas

def extract_adj_antonyms(sent):
    """Return list of all antonym lemmas for nouns in a given sentence"""
    antonyms = []
    all_adj_lemmas = extract_adj_lemmas(sent)
    for lemma in all_adj_lemmas:
        antonyms.extend(lemma.antonyms())
    return antonyms

def synset_overlap_features(sent1, sent2):
    """Returns counter for all mutual synsets between two sentences."""
    sent1_synsets = extract_noun_synsets(sent1)
    sent2_synsets = extract_noun_synsets(sent2)
    overlap_synsets = [syn for syn in sent1_synsets if syn in sent2_synsets]
    return Counter(overlap_synsets)

def synset_exclusive_first_features(sent1, sent2):
    """Returns counter for all nouns in first sentence with no possible synonyms in second"""
    sent1_synset_dict = noun_synset_dict(sent1)
    sent2_synsets = extract_noun_synsets(sent2)
    firstonly_nouns = [noun for noun in sent1_synset_dict if not len(set(sent1_synset_dict[noun]) & set(sent2_synsets))]
    return Counter(firstonly_nouns)

def synset_exclusive_second_features(sent1, sent2):
    """Returns counter for all nouns in second sentence with no possible synonyms in first"""
    sent1_synsets = extract_noun_synsets(sent1)
    sent2_synset_dict = noun_synset_dict(sent2)
    secondonly_nouns = [noun for noun in sent2_synset_dict if not len(set(sent2_synset_dict[noun]) & set(sent1_synsets))]
    return Counter(secondonly_nouns)


def hypernym_features(sent1, sent2):
    """ Calculate hypernyms of sent1 and check if synsets of sent2 contained in
    hypernyms of sent1. Trying to capture patterns of the form
    'A dog is jumping.' entails 'An animal is being active.'
    Returns an indicator feature of form 'contains_hypernyms: True/False'

    TODO: Fix missing simple entail inference such as dog -> dog
    """
    s1_nouns, s1_syns = extract_nouns_and_synsets(sent1)
    s2_syns = extract_noun_synsets(sent2)
    all_hyper_synsets = set(s1_syns) #Stores the hypernym synsets of the nouns in the first sentence
    for syn in s1_syns:
        all_hyper_synsets.update(set([i for i in syn.closure(lambda s:s.hypernyms())]))
    synset_overlap = all_hyper_synsets & set(s2_syns) #Stores intersection of sent2 synsets and hypernyms of sent1
    return Counter({'contains_hypernyms:': len(synset_overlap) >= 1})

def antonym_features(sent1, sent2):
    """Use antonyms between sentences to recognize contradiction patterns."""
    sent2_lemmas = extract_adj_lemmas(sent2)
    sent1_antonyms = extract_adj_antonyms(sent1)
    antonyms = [lem for lem in sent1_antonyms if lem in sent2_lemmas]
    return Counter(antonyms)

def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])

features_mapping = {'word_cross_product': word_cross_product_features,
            'word_overlap': word_overlap_features,
            'synset_overlap' : synset_overlap_features,
            'hypernyms' : hypernym_features,
            'antonyms' : antonym_features} #Mapping from feature to method that extracts  given features from sentences

def featurizer(reader=sick_train_reader, features_funcs=None):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector."""
    feats = []
    labels = []
    split_index = None
    for label, t1, t2 in reader():
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features_funcs:
            d = features_mapping[feat](t1, t2)
            feat_dict.update(d)
        feats.append(feat_dict)
        labels.append(label)
    return (feats, labels)
