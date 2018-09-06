#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import random

from imagenet.hdf5 import get_file
from imagenet.meta import load_class_names
from imagenet.mode import get_mode
names = load_class_names()

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', default='train', help='data mode')


def vis(encoded_image, filename, target, bbox=None):
    import matplotlib.pyplot as plt
    from PIL import Image
    from imagenet.hdf5 import to_buffer
    import numpy as np
    import matplotlib.patches as patches
    print(filename)
    print(names[target-1])
    image = np.array(Image.open(to_buffer(encoded_image)))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if bbox is not None:
        bbs = np.array(bbox).reshape(-1, 4)
        for bb in bbs:
            ymin, xmin, ymax, xmax = bb
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none',
                linewidth=1, edgecolor='r')
            ax.add_patch(rect)
            print(bb)
    plt.show()


def main(_):
    mode = FLAGS.mode
    with get_file(get_mode(mode), 'r') as fp:
        images = fp['encoded_images']
        filenames = fp['filenames']
        targets = fp['targets']
        if 'bounding_boxes' in fp:
            bbs = fp['bounding_boxes']
        else:
            bbs = None
        n = len(images)
        print('number of examples: %d' % n)
        indices = list(range(n))
        print('shuffling...')
        random.shuffle(indices)
        print('Done!')
        for i in indices:
            vis(images[i], filenames[i], targets[i],
                None if bbs is None else bbs[i])


app.run(main)
