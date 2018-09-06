#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import logging
import os
import random
from progress.bar import IncrementalBar
import h5py
import numpy as np
import tarfile

from imagenet.tarred import get_tar_path
from imagenet.hdf5 import from_buffer
from imagenet.hdf5 import get_file, get_file_path
from imagenet.meta import meta
from imagenet.mode import get_mode


ZERO_BASED = False

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)

flags.DEFINE_boolean(
    'delete_tar', default=False,
    help='delete `.tar` files after creating conversion')

flags.DEFINE_boolean(
    'overwrite', default=False,
    help='if the target hdf5 file already exists and this is True, '
         'overwrites it, otherwise throws.')

flags.DEFINE_string('mode', default='val', help='data mode')

flags.register_validator(
    'mode',
    lambda mode: mode in (
        'val', 'train', 'test', 'eval', 'infer', 'predict', 'validation'),
    message="--mode must be in "
            "('val', 'train', 'test', 'eval', 'infer', "
            "'predict', 'validation').")


def prep_group(fp, n_examples, include_targets):
    vlen_dtype = h5py.special_dtype(vlen=np.dtype(np.uint8))
    images = fp.create_dataset(
        'encoded_images', shape=(n_examples,), dtype=vlen_dtype)
    filenames = fp.create_dataset(
        'filenames', shape=(n_examples,), dtype='S32')
    if include_targets:
        targets = fp.create_dataset(
            'targets', shape=(n_examples,), dtype=np.int32)
        return images, filenames, targets
    else:
        return images, filenames


def write_examples(fp, n_examples, include_targets, records, shuffle):
    groups = prep_group(fp, n_examples, include_targets)
    if include_targets:
        images, filenames, targets = groups
    else:
        images, filenames = groups
    indices = list(range(n_examples))
    if shuffle:
        random.shuffle(indices)
    bar = IncrementalBar(max=n_examples)
    i = 0
    for record in records:
        index = indices[i]
        i += 1
        if include_targets:
            image, filename, target = record
            targets[index] = target
        else:
            image, filename = record
        images[index] = from_buffer(image)
        filenames[index] = filename
        bar.next()
    bar.finish()


def maybe_delete(mode, wordnet_id, delete_tar):
    if delete_tar:
        tar_path = get_tar_path(mode, wordnet_id)
        logger.info('Removing tar file "%s"' % tar_path)
        os.remove(tar_path)


def get_train_length(wordnet_ids):
    mode = get_mode('train')
    n = 0
    logger.info('Getting number of train examples...')
    bar = IncrementalBar(max=len(wordnet_ids))
    for wordnet_id in wordnet_ids:
        n += get_tar_length(get_tar_path(mode, wordnet_id))
        bar.next()
    bar.finish()
    return n


def get_tar_length(tar_path):
    with tarfile.TarFile(tar_path, 'r') as fp:
        return len(fp.getmembers())


def get_train_records(wordnet_ids, delete_tar=False):
    class_indices = meta.get_wordnet_indices(zero_based=ZERO_BASED)
    mode = get_mode('train')
    for wordnet_id in wordnet_ids:
        target = class_indices[wordnet_id]

        with tarfile.TarFile(get_tar_path(mode, wordnet_id)) as tar:
            for member in tar.getmembers():
                filename = member.name
                image = tar.extractfile(member)
                yield image, filename, target
                image.close()

        maybe_delete(mode, wordnet_id, delete_tar)


def get_val_records(delete_tar=False):
    from imagenet.meta import load_val_labels
    mode = get_mode('val')
    labels = load_val_labels(zero_based=ZERO_BASED)

    with tarfile.TarFile(get_tar_path(mode)) as tar:
        for member in tar.getmembers():
            filename = member.name
            index = filename.split('_')[-1][:-5]
            target = labels[int(index)-1]
            image = tar.extractfile(member)
            yield image, filename, target
            image.close()

    maybe_delete(mode, None, delete_tar)


def get_test_records(delete_tar):
    mode = get_mode('test')

    with tarfile.TarFile(get_tar_path(mode)) as tar:
        for member in tar.getmembers():
            filename = member.name
            image = tar.extractfile(member)
            yield image, filename
            image.close()

    maybe_delete(mode, None, delete_tar)


def check_overwrite(mode, overwrite):
    path = get_file_path(mode)
    if not overwrite and os.path.isfile(path):
        raise RuntimeError(
            'hdf5 file already exists at path "%s". '
            '\nUse --overwrite if sure.' % path)


def convert_other(mode, delete_tar, overwrite=False):
    mode = get_mode(mode)
    check_overwrite(mode, overwrite)
    n_examples = get_tar_length(get_tar_path(mode))

    with get_file(mode, 'w') as fp:
        if mode == 'val':
            write_examples(
                fp, n_examples, include_targets=True,
                records=get_val_records(delete_tar), shuffle=False)
        elif mode == 'test':
            write_examples(
                fp, n_examples, include_targets=False,
                records=get_test_records(delete_tar), shuffle=False)
        else:
            raise ValueError('Invalid mode: "%s"' % mode)


def convert_train(delete_tar=False, overwrite=False):
    from imagenet.tarred import get_wordnet_ids
    mode = 'train'
    check_overwrite(mode, overwrite)
    wordnet_ids = get_wordnet_ids()

    print('Counting examples...')
    n_examples = get_train_length(wordnet_ids)
    print('n examples: %d' % n_examples)

    with get_file(mode, 'w') as fp:
        records = get_train_records(
            wordnet_ids, delete_tar=delete_tar)
        write_examples(
            fp, n_examples, include_targets=True, records=records,
            shuffle=True)


def _main(mode, delete_tar, overwrite):
    mode = get_mode(mode)
    if mode == 'train':
        convert_train(delete_tar=delete_tar, overwrite=overwrite)
    else:
        convert_other(mode, delete_tar=delete_tar, overwrite=overwrite)


def main(_):
    _main(FLAGS.mode, FLAGS.delete_tar, FLAGS.overwrite)


app.run(main)
