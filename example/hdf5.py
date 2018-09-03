#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from imagenet.meta import meta, load_class_names
# from progress.bar import IncrementalBar
# from imagenet.hdf5 import get_train_file, as_image_bytes
from imagenet.hdf5 import get_train_dataset

class_names = load_class_names()
wn_indices = meta.get_wordnet_indices()

with get_train_dataset('r') as ds:
    print('Getting keys')
    keys = list(ds.keys())
    print('Shuffling...')
    random.shuffle(keys)
    print('keys: %d' % len(keys))
    for k in keys:
        k0, k1 = k
        index = wn_indices[k0]
        name = class_names[index]
        print(k0, name, k1)
        data = np.array(Image.open(ds[k]))
        plt.imshow(data)
        plt.show()

# with get_train_file() as fp:
#     print('Getting keys')
#     keys = []
#     bar = IncrementalBar(max=len(fp))
#     for k in fp:
#         bar.next()
#         keys.extend((k, k1) for k1 in fp[k].keys())
#     bar.finish()
#     print('Shuffling...')
#     random.shuffle(keys)
#     print('keys: %d' % len(keys))
#     for k1, k2 in keys:
#         print(k1, k2)
#         data = np.array(Image.open(as_image_bytes(fp[k1][k2])))
#         plt.imshow(data)
#         plt.show()
