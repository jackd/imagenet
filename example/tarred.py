#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from imagenet.tarred import get_class_dataset
from imagenet.meta import meta

ids = meta.get_wordnet_ids()

id = random.sample(ids, 1)[0]

with get_class_dataset(id) as ds:
    print('keys: %d' % len(tuple(ds.keys())))
    for k in ds.keys():
        print(k)
        data = np.array(Image.open(ds[k]))
        plt.imshow(data)
        plt.show()
