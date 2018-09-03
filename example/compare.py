#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imagenet.meta import meta
import imagenet.hdf5 as h

meta_ids = set(meta.get_wordnet_ids())
h_ids = set(h.get_wordnet_ids())
print(len(meta_ids))
print(len(h_ids))
for i in meta_ids:
    if i not in h_ids:
        print(i)
