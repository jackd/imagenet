from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .path import get_imagenet_dir


def get_tar_dir():
    return os.path.join(get_imagenet_dir(), 'tarred')


def get_train_dir():
    return os.path.join(get_tar_dir(), 'ILSVRC2012_img_train')


def get_train_path(wordnet_id):
    return os.path.join(get_train_dir(), '%s.tar' % wordnet_id)


def get_eval_path():
    return os.path.join(get_tar_dir(), 'ILSVRC2012_img_val.tar')


def get_wordnet_ids():
    fns = os.listdir(get_train_dir())
    return tuple(fn[:-4] for fn in fns)


def get_class_dataset(wordnet_id):
    from dids.file_io.tar_file_dataset import TarFileDataset
    dataset = TarFileDataset(get_train_path(wordnet_id), mode='r')
    n = len(wordnet_id) + 1
    return dataset.map_keys(
        lambda k: '%s_%s.JPEG' % (wordnet_id, k), lambda k: k[n:-5])


def get_train_dataset():
    from dids.core import BiKeyDataset
    datasets = {k: get_class_dataset(k) for k in get_wordnet_ids()}
    return BiKeyDataset(datasets)


def get_eval_dataset():
    from dids.file_io.tar_file_dataset import TarFileDataset
    dataset = TarFileDataset(get_eval_path(), mode='r')
    return dataset.map_keys(
        lambda k: '%s.JPEG' % k, lambda k: k[:-5])
