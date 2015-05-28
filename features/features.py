__author__ = 'mihaileric'

import os
import sys


"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from collections import Counter
from framenet.fn_tools import is_super_frame
from framenet import frame
import itertools
import nltk
import csv
from util.utils import sick_train_reader, leaves
import numpy as np

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from util.colors import color, prettyPrint
from util.distributedwordreps import build, cosine

lemmatizer = WordNetLemmatizer()

# with open('..', 'r') as f:
#     count = 0
#     for line in f:
#         if count > 5: break
#         print line
#         count += 1

GLOVE_MAT, GLOVE_VOCAB, _ = build('../cs224u/distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

def glvvec(w):
    """Return the GloVe vector for w."""
    i = GLOVE_VOCAB.index(w)
    return GLOVE_MAT[i]

def word_overlap_features(t1, t2):
    overlap = [w1 for w1 in leaves(t1) if w1 in leaves(t2)]
    feat = Counter(overlap)
    feat['overlap_length'] = len(overlap)
    feat['one_not_two_length'] = len([w1 for w1 in leaves(t1) if w1 not in leaves(t2)])
    feat['two_not_one_length'] = len([w2 for w2 in leaves(t2) if w2 not in leaves(t1)])
    return feat

def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])

def tree2sent(t1, t2):
    return ' '.join(leaves(t1)), ' '.join(leaves(t2))

def length_features(t1, t2):
    feat = {}
    feat['length_1'] = len(leaves(t1))
    feat['length_2'] = len(leaves(t2))
    return feat

def tree_depth(t):
    curr_depth = 0
    max_depth = 0
    for ch in str(t):
        if ch == '(':
            curr_depth += 1
        if ch == ')':
            curr_depth -= 1
        if curr_depth > max_depth:
            max_depth = curr_depth
    return max_depth

def tree_depth_features(t1, t2):
    feat = {}
    feat['depth_1'] = tree_depth(t1)
    feat['depth_2'] = tree_depth(t2)
    feat['depth_similarity'] = 1 - abs((feat['depth_1'] - feat['depth_2'])/(feat['depth_1'] + feat['depth_2']))
    return feat

def penn2wn(tag):
    """ Given a Penn Treebank tag, returns the appropriate 
        WordNet tag, if possible.  Otherwise returns ''. """

    if tag[0] in 'JVNR':
        ind = 'JVNR'.index(tag[0]) # Starts with (AD)J, V(ERB), N(OUN), (ADVE)R(B)
        return 'avnr'[ind]
    return ''

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

def extract_synsets (sent):
    ''' Extracts the synsets of all the words in the sentence with matching
        POS tag.  '''
    synsets = []
    tagged = pos_tag(sent.split())
   # print tagged
    for word, pos in tagged:
        synsets.extend(wn.synsets(word, pos=penn2wn(pos)))
    return synsets

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

def synset_overlap_features(t1, t2):
    """Returns counter for all mutual synsets between two sentences."""
    sent1, sent2 = tree2sent(t1, t2)
    sent1_synsets = extract_noun_synsets(sent1)
    sent2_synsets = extract_noun_synsets(sent2)
    overlap_synsets = [str(syn) for syn in sent1_synsets if syn in sent2_synsets]
    return Counter(overlap_synsets)

def synset_exclusive_first_features(t1, t2):
    """Returns counter for all nouns in first sentence with no possible synonyms in second"""
    sent1, sent2 = tree2sent(t1, t2)
    sent1_synset_dict = noun_synset_dict(sent1)
    sent2_synsets = extract_noun_synsets(sent2)
    firstonly_nouns = [str(noun) for noun in sent1_synset_dict if not len(set(sent1_synset_dict[noun]) & set(sent2_synsets))]
    return Counter(firstonly_nouns)

def synset_exclusive_second_features(t1, t2):
    """Returns counter for all nouns in second sentence with no possible synonyms in first"""
    sent1, sent2 = tree2sent(t1, t2)
    sent1_synsets = extract_noun_synsets(sent1)
    sent2_synset_dict = noun_synset_dict(sent2)
    secondonly_nouns = [str(noun) for noun in sent2_synset_dict if not len(set(sent2_synset_dict[noun]) & set(sent1_synsets))]
    return Counter(secondonly_nouns)

def subphrase_generator(tree):
    ''' Given a tree, returns all of the subphrases '''
    phrases = [tree]
    def extract_subphrases(subtree):
        if isinstance(subtree, tuple):
            for sp in subtree:
                phrases.append(sp)
                extract_subphrases(sp)

    extract_subphrases(tree)
    return phrases

# When a phrase in t2 contains a phrase in t1, count it.
# TODO: Test this, and compare it against summing for v IN p2.
def phrase_share_feature(t1, t2):
    p1, p2 = subphrase_generator(t1), subphrase_generator(t2)
    shared = [str((v, w)) for v in p1 for w in p2 if v == w]
    return Counter(shared)

def general_hypernym(t1, t2):   
    ''' Calculates hypernyms of sentence 1 and sentence 2 using matching POS tags. 
        Also permits self-hypernyms '''
    s1_leaves, s2_leaves = leaves(t1), leaves(t2)
    sent1, sent2 = ' '.join(s1_leaves), ' '.join(s2_leaves)
    
    hsyns = extract_synsets(sent1)
    syns = extract_synsets(sent2)

    all_hyper_synsets = set(extract_synsets(sent1))

    # Counts the number of synsets of a word in the sentence with the 
    # same POS tag as parsed 
    s1_len = len(set(s for s in hsyns if s.name().partition('.')[0] in sent1))
    s2_len = len(set(s for s in syns if s.name().partition('.')[0] in sent2))
                            
    for h in hsyns:
        all_hyper_synsets.update(set([i for i in h.closure(lambda s:s.hypernyms())]))
    overlap = all_hyper_synsets & set(syns)

    feature_string = "hypernyms {0} {1} {2}".format(s1_len, s2_len, len(overlap))
    return Counter({feature_string : 1})
 
def hypernym_features(t1, t2):
    """ Calculate hypernyms of sent1 and check if synsets of sent2 contained in
    hypernyms of sent1. Trying to capture patterns of the form
    'A dog is jumping.' entails 'An animal is being active.'
    Returns an indicator feature of form 'contains_hypernyms: True/False'
	TODO: Change what this feature returns!
    """
    sent1 = ' '.join(leaves(t1))
    sent2 = ' '.join(leaves(t2))
    s1_nouns, s1_syns = extract_nouns_and_synsets(sent1)

    
    s2_nouns, s2_syns  = extract_nouns_and_synsets(sent2)
    all_hyper_synsets = set(s1_syns) #Stores the hypernym synsets of the nouns in the first sentence
    s1_len, s2_len = len(s1_nouns), len(s2_nouns)

    for syn in s1_syns:
        all_hyper_synsets.update(set([i for i in syn.closure(lambda s:s.hypernyms())]))
    synset_overlap = all_hyper_synsets & set(s2_syns) # Stores intersection of sent2 synsets and hypernyms of sent1
    
    # Discretize into smaller buckets based on the number of nouns in each sentence,
    # and the size of the synset overlap.
    feature_string = 'hypernyms {0} {1} {2}'.format(s1_len, s2_len, len(synset_overlap))

    return Counter({feature_string : 1})

def antonym_features(t1, t2):

    """Use antonyms between sentences to recognize contradiction patterns. TODO: Extract antonyms from nouns and other syntactic families as well!"""

    sent1 = ' '.join(leaves(t1))
    sent2 = ' '.join(leaves(t2))
    sent2_lemmas = extract_adj_lemmas(sent2)
    sent1_antonyms = extract_adj_antonyms(sent1)
    antonyms = [str(lem) for lem in sent1_antonyms if lem in sent2_lemmas]
    return Counter(antonyms)

def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])


def word_cross_product_nv(t1, t2):
    nv1 = [w[0] for w in pos_tag(leaves(t1)) if penn2wn(w[1]) in 'nv']
    nv2 = [w[0] for w in pos_tag(leaves(t2)) if penn2wn(w[1]) in 'nv']

def frame_overlap_features(t1, t2, sf1, sf2):
    frame_names1 = [f1.name for f1 in sf1]
    frame_names2 = [f2.name for f2 in sf2]
    overlap = ['frame_' + fn1 for fn1 in frame_names1 if fn1 in frame_names2]
    feat = Counter(overlap)
    feat['first_frames'] = len(sf1)
    feat['second_frames'] = len(sf2)
    feat['overlap_frames'] = len(overlap)
    return feat

def frame_cross_product_features(t1, t2, sf1, sf2):
    frame_names1 = [f1.name for f1 in sf1]
    frame_names2 = [f2.name for f2 in sf2]
    return Counter([(f1, f2) for f1, f2 in itertools.product(frame_names1, frame_names2)])

def super_overlap(sf1, sf2):
    return [(f1, f2) for f1 in sf1 for f2 in sf2 if is_super_frame(f1, f2)]

def frame_entailment_features(t1, t2, sf1, sf2):
    super_overlap_strings = ['entailed_frame_' + f1.name for (f1, f2) in super_overlap(sf1, sf2)]
    feat = Counter(super_overlap_strings)
    feat['first_frames'] = len(sf1)
    feat['second_frames'] = len(sf2)
    feat['entailed_frames'] = len(super_overlap_strings)
    return feat


def frame_similarity_features(t1, t2, sf1, sf2):
    overlap = [(f1, f2) for f1 in sf1 for f2 in sf2 if f1.name == f2.name]
    overlap.extend(super_overlap(sf1, sf2))
    feat = {}
    total_sim = 0.0
    worst_sim = 1.0
    for f1, f2 in overlap:
        sim = frame.frame_similarity(f1, f2)
        total_sim += sim
        if sim < worst_sim:
            worst_sim = sim
        feat['frame_similarity_' + f1.name] = sim
    if len(overlap):
        feat['average_frame_similarity'] = total_sim/len(overlap)
        feat['worst_frame_similarity'] = worst_sim
    return feat

def negation_features(t1, t2):
    feat = {}
    s1, s2 = leaves(t1), leaves(t2)
    for word in ['no', 'not', 'none', "n't", 'nobody']:
        if (word in s1 and word not in s2) or (word in s2 and word not in s1):
            feat['{0}_negation'.format(word)] = 1.0

    return feat

grammar = """ \
            NN-PHRASE: {<DT.*> <NN> <RB>}
                      {<DT.*> <NN>}
                      {<DT.*> <JJ> <NN>}
          """
cp = nltk.RegexpParser(grammar)

def get_noun_phrase_labeled(t1,t2):
    """Gets noun phrases for given trees as lists with
    labeled POS tags."""
    sent1 = leaves(t1)
    sent2 = leaves(t2)

    tree1 = cp.parse(nltk.pos_tag(sent1))
    tree2 = cp.parse(nltk.pos_tag(sent2))

    #List of noun phrases with tokens and POS tags
    np1 = [subtree1.leaves() for subtree1 in tree1.subtrees() if subtree1.label() == 'NN-PHRASE']
    np2 = [subtree2.leaves() for subtree2 in tree2.subtrees() if subtree2.label() == 'NN-PHRASE']

    return np1, np2

def get_noun_phrase_words(nps_labeled):
    """Gets the words of a noun phrase."""
    all_np_words = []
    for np in nps_labeled:
        all_np_words += [" ".join([pair[0].lower() for pair in np])]
    return all_np_words

def get_noun_phrase_mapping(np1, np2):
    """Gets noun phrase mapping from noun phrase text to labeled list."""
    np1_token_mapping = {" ".join([pair[0].lower() for pair in np]): np for np in np1}
    np2_token_mapping = {" ".join([pair[0].lower() for pair in np]): np for np in np2}

    return np1_token_mapping, np2_token_mapping

def noun_phrase_modifier_features(t1, t2):
    """Extracts noun phrases within sentences and identifies
    whether a noun phrase in second sentence is subsumed by first."""
    feat = []
    np1, np2 = get_noun_phrase_labeled(t1, t2)

    np1_token_mapping, np2_token_mapping = get_noun_phrase_mapping(np1, np2)

    for tok1 in np1_token_mapping.keys():
        for tok2 in np2_token_mapping.keys():
            if set(tok2).issubset(set(tok1)):
                sent1_entities = np1_token_mapping[tok1]
                sent2_entities = np2_token_mapping[tok2]
                np1_pos = [tok[1] for tok in sent1_entities]
                np2_pos = [tok[1] for tok in sent2_entities]
                feature_key = ",".join(np1_pos) + ":" + ",".join(np2_pos)
                feat += [feature_key]

    return Counter(feat)

def get_noun_phrase_vector(noun_phrase):
    """Generate an aggregate distributed word vector
    for a given noun phrase."""
    all_word_vecs = [glvvec(w) for w in noun_phrase if w in GLOVE_VOCAB] #How often are the words not in glove vocab?
    return np.sum(all_word_vecs, axis=0)

def noun_phrase_word_vec_features(t1, t2):
    """Produces features for the similarity between
    pairwise noun phrase word vectors across the two sentences."""
    feat = {}
    np1, np2 = get_noun_phrase_labeled(t1, t2)

    np1_token_mapping, np2_token_mapping = get_noun_phrase_mapping(np1, np2)

    np1_words = get_noun_phrase_words(np1)
    np2_words = get_noun_phrase_words(np2)

    for n1 in np1_words:
        for n2 in np2_words:
            sent1_entities = np1_token_mapping[n1]
            sent2_entities = np2_token_mapping[n2]
            np1_pos = [tok[1] for tok in sent1_entities]
            np2_pos = [tok[1] for tok in sent2_entities]
            feature_key = ",".join(np1_pos) + ":" + ",".join(np2_pos)

            #Make feature weight equal to cosine similarity
            n1_vec = get_noun_phrase_vector(list(n1))
            n2_vec = get_noun_phrase_vector(list(n2))
            sim = 1 - cosine(n1_vec, n2_vec)
            feat[feature_key] = sim
    return feat


features_mapping = {'word_cross_product': word_cross_product_features,
            'word_overlap': word_overlap_features,
            'synset_overlap' : synset_overlap_features,
            'hypernyms' : hypernym_features,
            'new_hyp' : general_hypernym,
            'antonyms' : antonym_features,
            'first_not_second' : synset_exclusive_first_features,
            'second_not_first' : synset_exclusive_second_features,
            'frame_overlap' : frame_overlap_features,
            'frame_entailment' : frame_entailment_features,
            'frame_similarity' : frame_similarity_features,
            'frame_cross_product' : frame_cross_product_features,
            'negation' : negation_features,
            'length' : length_features,
            'tree_depth' : tree_depth_features,
            'noun_phrase_modifier' : noun_phrase_modifier_features,
            'noun_phrase_word_vec' : noun_phrase_word_vec_features} #Mapping from feature to method that extracts  given features from sentences

def featurizer(reader=sick_train_reader, features_funcs=None):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector.

    Valid feature_funcs return a dict of string : int key-value pairs.  """
    feats = []
    labels = []
    split_index = None

    for label, t1, t2, sf1, sf2 in reader():
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features_funcs:
            if feat.startswith('frame'):
                d = features_mapping[feat](t1, t2, sf1, sf2)
            else:
                d = features_mapping[feat](t1, t2)
            feat_dict.update(d)

        feats.append(feat_dict)
        labels.append(label)
    return (feats, labels)
