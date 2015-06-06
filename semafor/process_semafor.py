__author__ = 'guthriec'

from framenet.frame import Frame
import xml.etree.ElementTree as ET

def frametuples(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    sentences = []
    next_s_id = 0
    for sentence in root.iter('sentence'):
        s_id = int(sentence.get('ID'))
        next_s_id += 1
        annotation_sets = sentence.iter('annotationSet')
        text = sentence.find('text').text
        frame_list = []
        for annotation_set in annotation_sets:
            full_frame = Frame(text, annotation_set)
            frame_list.append(full_frame)
        sentences.append(frame_list)
    tuples = []
    for i in range(0, len(sentences)-1, 2):
        tuples.append((sentences[i], sentences[i+1]))
    return tuples
