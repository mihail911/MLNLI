__author__ = 'guthriec'

import xml.etree.ElementTree as ET

def frametuples(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    sentences = []
    next_s_id = 0
    for sentence in root.iter('sentence'):
        s_id = int(sentence.get('ID'))
        if not s_id == next_s_id:
            raise RuntimeError('sentences out of order')
        next_s_id += 1
        frames = sentence.iter('annotationSet')
        sentences.append([frame.get('frameName') for frame in frames])
    tuples = []
    for i in range(0, len(sentences)-1, 2):
        tuples.append((sentences[i], sentences[i+1]))
    return tuples