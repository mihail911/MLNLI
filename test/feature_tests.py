__author__ = 'guthriec'

import features.features as features
from nltk.corpus import wordnet as wn

class TestResult:
    def __init__(self, test_name):
        self.passed = True
        self.err_messages = []
        self.test_name = test_name

    def add_failure(self, message = ""):
        self.passed = False
        self.err_messages.append(message)

    def full_messages(self):
        if self.passed:
            return [self.test_name + " passed"]
        else:
            out = [self.test_name + " failed"]
            out.extend(self.err_messages)
            return out


def test_hypernyms():
    table_sent1 = "I hate when the table is too short."
    table_sent2 = "What a cute table!"
    furniture_sent = "That furniture is ugly."
    hot_sent = "Of the hot soups, it is the best."
    result = TestResult("Hypernyms")
    if not features.hypernym_features(table_sent1, furniture_sent)['contains_hypernyms:']:
        result.add_failure("Basic hypernym not captured")
    if not features.hypernym_features(table_sent1, table_sent2)['contains_hypernyms:']:
        result.add_failure("Reflexive hypernym (i.e. dog -> dog) not captured")
    if features.hypernym_features(table_sent1, hot_sent)['contains_hypernyms:']:
        result.add_failure("Hypernym false positive")
    return result

def test_synsets():
    return False

def test_antonyms():
    hot_sent = "Of the hot soups, it is the best."
    cold_sent = "It is the worst cold soup"
    result = TestResult('Antonyms')
    if features.antonym_features(cold_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        result.add_failure("Antonym falsely detected")
    if not features.antonym_features(hot_sent, cold_sent)[wn.lemma('cold.a.01.cold')]:
        result.add_failure("Antonym failed to be detected")
    return result

def run_feature_tests(print_results=True):
    results = []
    results.append(test_hypernyms())
    results.append(test_antonyms())
    success = True
    for result in results:
        if print_results:
            print result.full_messages()
        success = success and result.passed
    if print_results and success:
        print "All tests successful"
    if print_results and not success:
        print "Test failure, see above for details"
    return success

print run_feature_tests()