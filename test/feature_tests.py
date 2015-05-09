__author__ = 'guthriec'

import features.features as features
from nltk.corpus import wordnet as wn

def add_failure(return_val, message):
    return_val['passed'] = False
    return_val['messages'].append(message)

def test_hypernyms():
    table_sent1 = "I hate when the table is too short."
    table_sent2 = "What a cute table!"
    furniture_sent = "That furniture is ugly."
    hot_sent = "Of the hot soups, it is the best."
    return_val = {'passed': True, 'messages': []}
    if not features.hypernym_features(table_sent1, furniture_sent)['contains_hypernyms:']:
        add_failure(return_val, "Basic hypernym not captured")
    if not features.hypernym_features(table_sent1, table_sent2)['contains_hypernyms:']:
        add_failure(return_val, "Reflexive hypernym (i.e. dog -> dog) not captured")
    if features.hypernym_features(table_sent1, hot_sent)['contains_hypernyms:']:
        add_failure(return_val, "Hypernym false positive")
    if return_val['passed']:
        return_val['messages'] = ["Hypernym tests successful"]
    return return_val

def test_synsets():
    return False

def test_antonyms():
    hot_sent = "Of the hot soups, it is the best."
    cold_sent = "It is the worst cold soup"
    return_val = {'passed': True, 'messages': []}
    if features.antonym_features(cold_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        add_failure(return_val, "Antonym falsely detected")
    if not features.antonym_features(hot_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        add_failure(return_val, "Antonym failed to be detected")
    if return_val['passed']:
        return_val['messages'] = ["Antonym tests successful"]
    return return_val

def run_feature_tests(print_results=True):
    results = []
    results.append(test_hypernyms())
    results.append(test_antonyms())
    success = True
    for result in results:
        if print_results:
            print result['messages']
        success = success and result['passed']
    if print_results and success:
        print "All tests successful"
    if print_results and not success:
        print "Test failure, see above for details"
    return success

run_feature_tests()