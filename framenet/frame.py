__author__ = 'guthriec'

from nltk.stem.snowball import *
stemmer = SnowballStemmer("english")

class Frame:
    def __init__(self, text, annotation_xml):
        if annotation_xml.tag != 'annotationSet':
            raise ValueError('annotation_xml must be an annotationSet')
        self.name = annotation_xml.get('frameName')
        self.text = text
        self.labels = {}
        for label in annotation_xml.iter('label'):
            start = int(label.get('start'))
            end = int(label.get('end'))
            label_name = label.get('name')
            label_text = text[start:(end+1)]
            self.labels[label_name] = label_text

    def __str__(self):
        return self.text + ' ' + self.name + ' ' + ' '.join([label + ': ' + stemmer.stem(word) for (label, word) in self.labels.iteritems()])

def frame_similarity(f1, f2):
    matches = 0.0
    mismatches = 0.0
    label_mismatches = 0.0
    for label1, text1 in f1.labels.iteritems():
        try:
            text2 = f2.labels[label1]
            if stemmer.stem(text1) == stemmer.stem(text2):
                matches += 1
            else:
                mismatches += 1
        except KeyError:
            label_mismatches += 1
    for label2 in f2.labels:
        if label2 not in f1.labels:
            label_mismatches += 1

    return matches / (matches + mismatches + label_mismatches)