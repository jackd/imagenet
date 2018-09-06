from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py

from .mode import get_mode


def to_buffer(data):
    import io
    import numpy as np
    return io.BytesIO(np.array(data, dtype=np.uint8))


def to_image(data):
    from PIL import Image
    return Image.open(to_buffer(data))


def from_buffer(fp):
    import numpy as np
    return np.fromstring(fp.read(), dtype=np.uint8)


def get_hdf5_dir():
    from .path import get_imagenet_dir
    dir = os.path.join(get_imagenet_dir(), 'hdf5')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_file_path(mode='train'):
    return os.path.join(get_hdf5_dir(), 'ILSVRC2012_img_%s.hdf5' % mode)


def get_file(mode='train', file_mode='r'):
    return h5py.File(get_file_path(get_mode(mode)), file_mode)


class IndexedLoader(object):
    def __init__(self, mode):
        self._mode = mode
        self._f = None
        self._filename_index = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        import numpy as np
        self._f = get_file(self._mode)
        fns = np.array(self._f['filenames'])
        self._filename_index = {k: i for i, k in enumerate(fns)}

    def close(self):
        self._f = None
        self._filename_index = None

    def load_data(self, filename):
        import numpy as np
        self._assert_open()
        i = self._filename_index[filename]
        image = to_image(self._f['encoded_images'][i])
        bbox = np.reshape(self._f['bounding_boxes'][i], (-1, 4))
        return image, bbox

    def filenames(self):
        self._assert_open()
        return self._filename_index.keys()

    def get_index(self, filename):
        self._assert_open()
        return self._filename_index[filename]

    @property
    def is_open(self):
        return self._f is not None

    def _assert_open(self):
        if not self.is_open:
            raise RuntimeError('`IndexedLoader` not open.')
