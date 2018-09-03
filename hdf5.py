from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import h5py


def get_hdf5_dir():
    from .path import get_imagenet_dir
    dir = os.path.join(get_imagenet_dir(), 'hdf5')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_train_path():
    return os.path.join(get_hdf5_dir(), 'ILSVRC2012_img_train.hdf5')


def get_train_file(mode='r'):
    return h5py.File(get_train_path(), mode=mode)


def as_image_bytes(dataset):
    import numpy as np
    return io.BytesIO(np.array(dataset, dtype=np.uint8))


def get_train_dataset(mode='r'):
    from dids.file_io.hdf5 import NestedHdf5Dataset
    return NestedHdf5Dataset(2, get_train_path(), mode=mode).map(
        as_image_bytes)


def get_wordnet_ids():
    with get_train_file('r') as fp:
        return tuple(fp.keys())
