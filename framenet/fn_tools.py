__author__ = 'guthriec'

from framenet.frame import Frame
import os
import sys
import xml.etree.ElementTree as ET

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)
print root_dir

index_tree = ET.parse('framenet/frameIndex.xml')
relation_tree = ET.parse('framenet/frRelation.xml')

name_id_dict = {}
index_root = index_tree.getroot()
for frame in index_root.iter('{http://framenet.icsi.berkeley.edu}frame'):
    name_id_dict[frame.get('name')] = frame.get('ID')

super_frame_dict = {}
relation_root = relation_tree.getroot()

for relation in relation_root.iter('{http://framenet.icsi.berkeley.edu}frameRelation'):
    sub_frame = relation.get('subFrameName')
    super_frame = relation.get('superFrameName')
    if sub_frame in super_frame_dict:
        super_frame_dict[sub_frame].append(super_frame)
    else:
        super_frame_dict[sub_frame] = [super_frame]

for name in name_id_dict:
    if name in super_frame_dict:
        super_frame_dict[name].append(name)
    else:
        super_frame_dict[name] = [name]

def name_to_id(frame_name):
    return name_id_dict[frame_name]

def super_frame_names(frame):
    return super_frame_dict[frame.name]

def is_super_frame(f1, f2):
    # is f2 a super frame of f1?
    if f2.name in super_frame_dict[f1.name]:
        return True
    else:
        return False