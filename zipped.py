from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .path import get_imagenet_dir


def get_zip_dir():
    dir = os.path.join(get_imagenet_dir(), 'zipped')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_train_dir():
    folder = os.path.join(get_zip_dir(), 'ILSVRC2012_img_train')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_wordnet_ids():
    fns = os.listdir(get_train_dir())
    return tuple(fn[:-4] for fn in fns)


def get_train_path(wordnet_id):
    return os.path.join(get_train_dir(), '%s.zip' % wordnet_id)


def get_eval_path():
    return os.path.join(get_zip_dir(), 'ILSVRC2012_img_val.tar')


def get_class_dataset(wordnet_id, mode='r'):
    from dids.file_io.zip_file_dataset import ZipFileDataset
    dataset = ZipFileDataset(get_train_path(wordnet_id), mode=mode)
    n = len(wordnet_id) + 1
    return dataset.map_keys(
        lambda k: '%s_%s.JPEG' % (wordnet_id, k), lambda k: k[n:-5])


def get_train_dataset():
    from dids.core import BiKeyDataset
    datasets = {k: get_class_dataset(k) for k in get_wordnet_ids()}
    return BiKeyDataset(datasets)


def get_eval_dataset(mode='r'):
    from dids.file_io.zip_file_dataset import ZipFileDataset
    dataset = ZipFileDataset(get_eval_path(), mode=mode)
    return dataset.map_keys(
        lambda k: '%s.JPEG' % k, lambda k: k[:-5])
