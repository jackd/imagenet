#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'delete_after_copy', default=False,
    help='delete `.tar` files after creating equivalent `.zip`')


def copy(wordnet_id, delete_after_copy):
    import imagenet.tarred as tarred
    import imagenet.zipped as zipped
    import os
    with tarred.get_class_dataset(wordnet_id) as src:
        with zipped.get_class_dataset(wordnet_id, 'w') as dst:
            dst.save_dataset(src, show_progress=True)
    if delete_after_copy:
        os.remove(tarred.get_train_path(wordnet_id))


def copy_all(delete_after_copy):
    from imagenet.tarred import get_wordnet_ids
    ids = get_wordnet_ids()
    n = len(ids)
    for i, wordnet_id in enumerate(ids):
        print('Copying %s, %d / %d' % (wordnet_id, i+1, n))
        copy(wordnet_id, delete_after_copy)


def main(_):
    copy_all(delete_after_copy=FLAGS.delete_after_copy)


app.run(main)
