__author__ = 'guthriec'

import features.features as features
from nltk.corpus import wordnet as wn

def test_antonyms():
    hot_sent = "Of the hot soups, it is the best."
    cold_sent = "It is the worst cold soup"
    if features.antonym_features(cold_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        return False
    if not features.antonym_features(hot_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        return False
    return True

print test_antonyms()
