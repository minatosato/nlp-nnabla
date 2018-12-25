# 
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from typing import Optional
from pathlib import Path

def load_imdb(vocab_size):
    dataset_path = Path('./imdb.npz')

    if not dataset_path.exists():
        import os
        os.system('wget https://s3.amazonaws.com/text-datasets/imdb.npz')

    unk_index = vocab_size - 1
    raw = np.load(dataset_path)
    ret = dict()
    for k, v in raw.items():
        if 'x' in k:
            for i, sentence in enumerate(v):
                v[i] = [word if word < unk_index else unk_index for word in sentence]
        ret[k] = v
    return ret['x_train'], ret['x_test'], ret['y_train'], ret['y_test']

def with_padding(sequences, padding_type:str='post', max_sequence_length:Optional[int]=None) -> np.ndarray:
    if max_sequence_length is None:
        max_sequence_length = max(map(lambda x: len(x), sequences))
    else:
        assert type(max_sequence_length) == int, 'max_sequence_length must be an integer.'
        assert max_sequence_length > 0, 'max_sequence_length must be a positive integer.'

    def _with_padding(sequence):
        sequence = sequence[:max_sequence_length]
        sequence_length = len(sequence)
        pad_length = max_sequence_length - sequence_length
        if padding_type == 'post':
            return sequence + [0] * pad_length
        elif padding_type == 'pre':
            return [0] * pad_length + sequence
        else:
            raise Exception('padding type error. padding type must be "post" or "pre"')

    return np.array(list(map(_with_padding, sequences)), dtype=np.int32)

