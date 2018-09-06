"""Wrapper for loading/manipulating matlab meta data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .path import get_meta_path
from .path import get_validation_labels_path


class _ImagenetMeta(object):
    def __init__(self):
        from scipy.io import loadmat
        self._meta = loadmat(get_meta_path())

    def get_wordnet_ids(self):
        """Get a tuple of wordnet IDs in class index order."""
        return tuple(syn['WNID'][0][0] for syn in self._meta['synsets'][:1000])

    def get_wordnet_indices(self, zero_based=True):
        """Get a dict mapping wordnet IDs to the corresponding class index."""
        ids = self.get_wordnet_ids()
        if zero_based:
            return {k: i for i, k in enumerate(ids)}
        else:
            return {k: i+1 for i, k in enumerate(ids)}


meta = _ImagenetMeta()


def load_class_names():
    """
    Load class names.

    Sourced from
    https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    """
    import os
    path = os.path.join(os.path.dirname(__file__), 'class_names.txt')
    # path = os.path.join(os.path.dirname(__file__), 'class_names.txt')
    with open(path, 'r') as fp:
        names = tuple(line.rstrip() for line in fp.readlines())
    return names


def load_val_labels(zero_based=True):
    """Load labels for validation dataset."""
    import numpy as np
    with open(get_validation_labels_path(), 'r') as fp:
        n = tuple(int(l.rstrip()) for l in fp.readlines())
    n = np.array(n, dtype=np.int32)
    if zero_based:
        n -= 1
    return n
