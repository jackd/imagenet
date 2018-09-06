#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import logging
from progress.bar import IncrementalBar
# import numpy as np
# import h5py

from imagenet.hdf5 import get_file
from imagenet.bbox import get_tarred_xml_file, parse_xml_bbox, get_xml_subpath
from imagenet.mode import get_mode

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', default='val', help='data mode')

flags.register_validator(
    'mode',
    lambda mode: mode in ('val', 'train', 'eval', 'validation'),
    message="--mode must be in ('val', 'train', 'eval', 'validation').")

flags.DEFINE_boolean(
    'overwrite', default=False,
    help='if the target hdf5 file already exists and this is True, '
         'overwrites it, otherwise throws.')

logger = logging.getLogger(__name__)


def compute_bbox(mode, overwrite):
    from multiprocessing import Pool
    with get_file(mode, 'r') as src:
        filenames = list(src['filenames'])
        n = len(filenames)

    with get_tarred_xml_file(mode) as tar:
        bar = IncrementalBar(max=n)

        result = {}

        def f(filename):
            subpath = get_xml_subpath(filename)
            member = tar.getmember(subpath)
            fp = tar.extractfile(member)
            bb = parse_xml_bbox(fp).flatten()
            return filename, bb

        pool = Pool(processes=4)
        for filename, bb in pool.imap_unordered(f, filenames):
            result[filename] = bb
            bar.next()
        bar.finish()
    return result


def main(_):
    compute_bbox(get_mode(FLAGS.mode), FLAGS.overwrite)


app.run(main)
