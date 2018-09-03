#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app
import numpy as np
import h5py

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'delete_after_copy', default=False,
    help='delete `.zip` files after creating equivalent hdf5 `group`')


dt = h5py.special_dtype(vlen=np.dtype('uint8'))


def copy(dst, wordnet_id, delete_after_copy):
    import imagenet.zipped as zipped
    import numpy as np
    # from PIL import Image
    from progress.bar import IncrementalBar
    import os
    with zipped.get_class_dataset(wordnet_id, 'r') as src:
        if wordnet_id in dst:
            del dst[wordnet_id]

        group = dst.create_group(wordnet_id)
        bar = IncrementalBar(max=len(src))
        for k in src:
            bar.next()
            group.create_dataset(
                # k, data=np.array(Image.open(src[k])), compression='jpeg')
                k, data=np.fromstring(src[k].read(), dtype=np.uint8))
        bar.finish()
    if delete_after_copy:
        os.remove(zipped.get_train_path(wordnet_id))


def copy_all(delete_after_copy):
    from imagenet.zipped import get_wordnet_ids
    from imagenet.hdf5 import get_train_file
    ids = get_wordnet_ids()
    n = len(ids)

    with get_train_file('a') as dst:
        for i, wordnet_id in enumerate(ids):
            print('Copying %s, %d / %d' % (wordnet_id, i+1, n))
            copy(dst, wordnet_id, delete_after_copy)


def main(_):
    copy_all(delete_after_copy=FLAGS.delete_after_copy)


app.run(main)
