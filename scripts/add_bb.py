#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import logging
from progress.bar import IncrementalBar
import numpy as np
import h5py

from imagenet.hdf5 import get_file
from imagenet.bbox import get_tarred_xml_file, parse_xml_bbox
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


def add_bbox(mode, overwrite):
    with get_file(mode, 'a') as src:
        if 'bounding_boxes' in src:
            if overwrite:
                logger.info('Deleting existing bounding_boxes')
                del src['bounding_boxes']
                bbs = None
            else:
                # raise RuntimeError(
                #     '"bounding_boxes" group already exists. '
                #     'Use --overwrite if sure.')
                logger.info('Found existing bounding_boxes')
                bbs = src['bounding_boxes']
        else:
            bbs = None
        logger.info('Loading filenames...')
        filenames = np.array(src['filenames'])
        assert(all(fn.endswith('.JPEG') for fn in filenames))
        filenames = [fn[:-5] for fn in filenames]
        logger.info('Indexing filenames...')
        filename_indices = {k: i for i, k in enumerate(filenames)}
        if bbs is None:
            logger.info('Creating bounding_boxes dataset')
            vlen_dtype = h5py.special_dtype(vlen=np.dtype(np.int32))
            bbs = src.create_dataset(
                'bounding_boxes', shape=(len(filenames),), dtype=vlen_dtype)
            empty_filenames = None
        else:
            logger.info('Finding empty bounding box filenames...')
            empty_filenames = set(
                fn for fn, bb in zip(filenames, np.array(bbs))
                if bb.shape[0] == 0)

        with get_tarred_xml_file(mode) as tar:
            logger.info('Getting members...')
            members = tar.getmembers()

            if empty_filenames is not None:
                members = [m for m in members if
                           m.name.endswith('.xml') and
                           m.name.split('/')[-1][:-4] in empty_filenames]
            logger.info('Parsing remaining xml files...')
            bar = IncrementalBar(max=len(members))
            for member in members:
                filename = member.name.split('/')[-1][:-4]
                i = filename_indices[filename]
                fp = tar.extractfile(member)
                bb = parse_xml_bbox(fp).flatten()
                bbs[i] = bb
                bar.next()
            bar.finish()


def main(_):
    add_bbox(get_mode(FLAGS.mode), FLAGS.overwrite)


app.run(main)
