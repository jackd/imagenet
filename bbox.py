from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import logging

from .tarred import get_tar_dir
# from .extracted import get_extracted_dir
from .mode import get_mode

logger = logging.getLogger(__name__)


train_base = 'ILSVRC2012_bbox_train_v2'
train_ext = 'tar.gz'
val_base = 'ILSVRC2012_bbox_val_v3'
val_ext = 'tgz'


def _get_path_info(mode):
    mode = get_mode(mode)
    if mode == 'train':
        return train_base, train_ext
    elif mode == 'val':
        return val_base, val_ext
    else:
        raise ValueError('No bbox file for mode "%s"' % mode)


def get_tarred_xml_path(mode):
    base, ext = _get_path_info(mode)
    return os.path.join(get_tar_dir(), '%s.%s' % (base, ext))


def get_tarred_xml_file(mode):
    return tarfile.open(get_tarred_xml_path(mode), 'r:gz')


# def get_extracted_xml_dir(mode):
#     base, ext = _get_path_info(mode)
#     return os.path.join(get_extracted_dir(), base)


# def extract_bbox_xml(mode):
#     src = get_tar_xml_path(mode)
#     dst = get_extracted_xml_dir(mode)
#     if not os.path.isdir(dst):
#         os.makedirs(dst)
#     with tarfile.TarFile(src, 'r') as tar:
#         tar.extractall(dst)
#     logger.info('Extracted bbox info: %s -> %s' % (src, dst))


def get_xml_subpath(filename):
    base = filename.split('.')[0]
    splits = base.split('_')
    if len(splits) == 3:
        # val
        assert(splits[0] == 'ILSVRC2012' and splits[1] == 'val')
        return os.path.join('val', '%s.xml' % base)
        pass
    elif len(splits) == 2:
        wordnet_id = splits[0]
        return os.path.join(wordnet_id, '%s.xml' % base)
    else:
        raise NotImplementedError('Not recognized filename: "%s"' % filename)


def _parse_bbox(bbox):
    d = {bb.tag: int(bb.text) for bb in bbox}
    return [d[k] for k in ('ymin', 'xmin', 'ymax', 'xmax')]


def parse_xml_bbox(source):
    import numpy as np
    import xml.etree.ElementTree as ET
    tree = ET.parse(source)
    root = tree.getroot()
    return np.array(
        [_parse_bbox(bbox) for bbox in root.iter('bndbox')], dtype=np.int32)
