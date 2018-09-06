from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from .mode import get_mode

logger = logging.getLogger(__name__)


def get_extracted_dir():
    from .path import get_imagenet_dir
    return os.path.join(get_imagenet_dir(), 'extracted')


def get_images_dir(mode='train', wordnet_id=None):
    mode = get_mode(mode)
    folder = os.path.join(get_extracted_dir(), 'ILSVRC2012_img_%s' % mode)
    if wordnet_id is not None:
        if mode != 'train':
            raise ValueError('wordnet_id must be None for mode "%s"' % mode)
        folder = os.path.join(folder, wordnet_id)
    return folder


def remove_images_dir(mode, wordnet_id=None):
    import shutil
    images_dir = get_images_dir(mode, wordnet_id)
    logger.info('Removing %s' % images_dir)
    shutil.rmtree(images_dir)


def extract(mode, wordnet_id=None):
    import tarfile
    from .tarred import get_tar_path
    folder = get_images_dir(mode, wordnet_id)
    path = get_tar_path(mode, wordnet_id)
    with tarfile.TarFile(path, 'r') as tar:
        tar.extractall(folder)
    return folder


class TemporaryExtraction(object):
    def __init__(self, mode, wordnet_id=None):
        self._mode = mode
        self._wordnet_id = wordnet_id
        self._folder = None

    def __enter__(self):
        logger.info('Performing temporary extraction')
        self._folder = extract(self._mode, self._wordnet_id)
        return self._folder

    def __exit__(self):
        import shutil
        logger.info('Removing temporary extraction')
        shutil.rmtree(self._folder)
        self._folder = None

    @property
    def mode(self):
        return self._mode

    @property
    def wordnet_id(self):
        return self._wordnet_id

    @property
    def folder(self):
        return self._folder
