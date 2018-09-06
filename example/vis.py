#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'filename', default='ILSVRC2012_val_00000838.JPEG',
    help='example filename')


def show(filename):
    import matplotlib.pyplot as plt
    from imagenet.hdf5 import IndexedLoader
    from imagenet.vis import vis_image
    if filename.startswith('ILSVRC2012'):
        mode = filename.split('_')[1]
    else:
        mode = 'train'
    with IndexedLoader(mode) as loader:
        image, bboxes = loader.load_data(filename)
        vis_image(image, bboxes)
        plt.show()


def main(_):
    show(FLAGS.filename)


app.run(main)
