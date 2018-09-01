"""Wrapper for loading/manipulating matlab meta data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .path import get_meta_path


class _ImagenetMeta(object):
    def __init__(self):
        from scipy.io import loadmat
        self._meta = loadmat(get_meta_path())

    def get_wordnet_ids(self):
        """Get a tuple of wordnet IDs in class index order."""
        return tuple(syn['WNID'][0][0] for syn in self._meta['synsets'][:1000])

    def get_wordnet_indices(self, zero_based=True):
        """Get a dict mapping wordnet IDs to the corresponding class index."""
        ids = self.geT_wordnet_ids()
        if zero_based:
            return {k: i for i, k in enumerate(ids)}
        else:
            return {k: i+1 for i, k in enumerate(ids)}


meta = _ImagenetMeta()
