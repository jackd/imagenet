from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from .path import get_imagenet_dir
from .mode import get_mode

logger = logging.getLogger(__name__)


def get_tar_dir():
    return os.path.join(get_imagenet_dir(), 'tarred')


def get_train_dir():
    return os.path.join(get_tar_dir(), 'ILSVRC2012_img_train')


def get_train_path(wordnet_id):
    return os.path.join(get_train_dir(), '%s.tar' % wordnet_id)


def get_wordnet_ids():
    return os.listdir(get_tar_path('train')[:-4])


def get_tar_path(mode, wordnet_id=None):
    mode = get_mode(mode)
    if wordnet_id is None:
        return os.path.join(
            get_tar_dir(), 'ILSVRC2012_img_%s.tar' % mode)
    else:
        if mode != 'train':
            raise ValueError('wordnet_id must be None for mode "%s"' % mode)
        return os.path.join(
            get_tar_dir(), 'ILSVRC2012_img_%s' % mode, '%s.tar' % wordnet_id)
