from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_mode_aliases = (
    ('train', ()),
    ('val', ('eval', 'validation')),
    ('test', ('infer', 'predict')),
)

_modes = {}
for k, v in _mode_aliases:
    for vi in v:
        assert(vi not in k)
        _modes[vi] = k
    assert(k not in _modes)
    _modes[k] = k


def get_mode(mode):
    return _modes[mode]


_n_examples = {
    'train': 1000 * 1300,
    'val': 50000,
    'test': 100000
}


def get_n_examples(mode):
    return _n_examples[get_mode(mode)]
