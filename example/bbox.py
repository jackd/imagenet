#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imagenet.bbox import get_tarred_xml_file, get_xml_subpath
from imagenet.bbox import parse_xml_bbox

mode = 'val'

with get_tarred_xml_file(mode) as tar:
    members = tar.getmembers()
    print(len(members))
    for member in members:
        fn = member.name.split('/')[-1]
        if fn.endswith('.xml'):
            subpath = get_xml_subpath(fn)
            fp = tar.extractfile(member)
            print('Starting %s' % member.name)
            bboxes = parse_xml_bbox(fp)
            print(bboxes)
            exit()
