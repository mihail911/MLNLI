__author__ = 'guthriec'

import features.features as features
from util.utils import str2tree, sick_dev_reader
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

    table_sent1 = str2tree("I hate when the table is too short.")
    table_sent2 = str2tree("What a cute table!")
    furniture_sent = str2tree("That furniture is ugly.")
    hot_sent = str2tree("Of the hot soups, it is the best.")
    dog_sent1 = str2tree("The dog is jumping on the bed.")
    dog_sent2 = str2tree("I look at the dog being active.")

    result = TestResult("Hypernyms")
    if not features.hypernym_features(table_sent1, furniture_sent)['contains_hypernyms:']:
        result.add_failure("Basic hypernym not captured")
    if not features.hypernym_features(table_sent1, table_sent2)['contains_hypernyms:']:
        result.add_failure("Self-hypernym (i.e. dog -> dog) not captured")
    if features.hypernym_features(table_sent1, hot_sent)['contains_hypernyms:']:
        result.add_failure("Hypernym false positive")
    return result

def test_synset_overlap():
    way_sent1 = str2tree("There's no way I can do that")
    way_sent2 = str2tree("I don't have the means to help")
    result = TestResult("Synset overlap")
    if not features.synset_overlap_features(way_sent1, way_sent2)['means.n.01']:
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
    if features.antonym_features(cold_sent, cold_sent)['cold.a.01.cold']:
        result.add_failure("Antonym falsely detected")
    if not features.antonym_features(hot_sent, cold_sent)['cold.a.01.cold']:
        result.add_failure("Antonym failed to be detected")
    return result

def test_frame_overlap():
    result = TestResult('Frame overlap')
    curr_dev_el = 0
    for label, t1, t2, sf1, sf2 in sick_dev_reader():
        print t1, t2, features.frame_overlap(t1, t2, sf1, sf2)
        curr_dev_el += 1
        if curr_dev_el == 5:
            break
    return result

def test_frame_entailment():
    result = TestResult('Frame entailment')
    curr_dev_el = 0
    for label, t1, t2, sf1, sf2 in sick_dev_reader():
        print t1, t2, features.frame_entailment(t1, t2, sf1, sf2)
        curr_dev_el += 1
        if curr_dev_el == 5:
            break
    return result

def run_feature_tests(print_results=True):
    results = []
    results.append(test_hypernyms())
    results.append(test_antonyms())
    results.append(test_synset_overlap())
    results.append(test_synset_exclusive())
    results.append(test_frame_overlap())
    results.append(test_frame_entailment())
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
