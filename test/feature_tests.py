__author__ = 'guthriec'


import os, sys
"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)


from util.utils import str2tree
from nltk.corpus import wordnet as wn
import features.features as features

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
    dog_sent1 = "The dog is jumping on the bed."
    dog_sent2 = "I look at the dog being active."

    result = TestResult("Hypernyms")
    if not features.hypernym_features(table_sent1, furniture_sent)['contains_hypernyms:']:
        result.add_failure("Basic hypernym not captured")
    if not features.hypernym_features(table_sent1, table_sent2)['contains_hypernyms:']:
        result.add_failure("Self-hypernym (i.e. dog -> dog) not captured")
    if features.hypernym_features(table_sent1, hot_sent)['contains_hypernyms:']:
        result.add_failure("Hypernym false positive")
    return result

def test_general_hyp():
    t1 = str2tree("dog (smart)")
    t2 = str2tree("cat (dumb)")

    features.general_hypernym(t1, t2)


def test_synset_overlap():
    way_sent1 = str2tree("There's no way I can do that")
    way_sent2 = str2tree("I don't have the means to help")
    result = TestResult("Synset overlap")
    if not features.synset_overlap_features(way_sent1, way_sent2)[wn.synset('means.n.01')]:
        result.add_failure("Basic synonym not captured")
    return result

def test_synset_exclusive():
    result = TestResult("Exclusive synset")
    sent1 = str2tree("The dog ate the meal, then peed on the tree")
    sent2 = str2tree("The dog ate the meal, then went to bed")
    if not features.synset_exclusive_first_features(sent1, sent2)['tree']:
        result.add_failure("Exclusive noun not included")
    if 'meal' in features.synset_exclusive_first_features(sent1, sent2):
        result.add_failure["Included noun not actually exclusive to first sentence"]
    return result

def test_antonyms():
    hot_sent = str2tree("Of the hot soups, it is the best.")
    cold_sent = str2tree("It is the worst cold soup")
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
    results.append(test_synset_overlap())
    results.append(test_synset_exclusive())
    success = True
    for result in results:
        if print_results:
            for message in result.full_messages():
                print message
        success = success and result.passed
    if print_results and success:
        print "All tests successful"
    if print_results and not success:
        print "Test failure, see above for details"
    return success

if __name__ == "__main__":
    run_feature_tests()
