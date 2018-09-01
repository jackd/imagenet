from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def get_imagenet_dir():
    key = 'IMAGENET_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


def get_validation_labels_path():
    return os.path.join(
        get_imagenet_dir(), 'ILSVRC2012_validation_ground_truth.txt')


def get_meta_path():
    return os.path.join(get_imagenet_dir(), 'meta.mat')
