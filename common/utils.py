# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os

from scipy import sparse
from collections import Counter
from pathlib import Path
from itertools import combinations
from typing import List
from typing import Tuple

w2i = {}
i2w = {}

c2i = {}
i2c = {}

w2i['pad'] = 0
i2w[0] = 'pad'
w2i['<eos>'] = 1
i2w[1] = '<eos>'

c2i[' '] = 0
i2c[0] = ' '

word_length = 20


def load_data(filename, with_bos=False) -> List[List[int]]:
    global w2i, i2w
    global c2i, i2c
    
    if with_bos:
        w2i['<bos>'] = 2
        i2w[2] = '<bos>'

    with open(filename) as f:
        lines = f.read().replace('\n', '<eos>')
        for char in set(lines):
            if char not in c2i:
                c2i[char] = len(c2i)
            if c2i[char] not in i2c:
                i2c[c2i[char]] = char

        words = lines.strip().split()
    dataset = np.ndarray((len(words), ), dtype=np.int32)

    for i, word in enumerate(words):
        if word not in w2i:
            w2i[word] = len(w2i)
        if w2i[word] not in i2w:
            i2w[w2i[word]] = word
        dataset[i] = w2i[word]

    sentences = []
    sentence = []
    if with_bos:
        sentence.append(w2i['<bos>'])
    for index in dataset:
        if i2w[index] != '<eos>':
            sentence.append(index)
        else:
            sentence.append(index)
            sentences.append(sentence)
            sentence = []
            if with_bos:
                sentence.append(w2i['<bos>'])
    return sentences


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


def with_padding(sequences, padding_type='post', max_sequence_length=None):
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

ptb_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{0}.txt'
types = ['train', 'valid', 'test']
ptb_urls = map(lambda x: ptb_url.format(x), types)
os.makedirs('./ptb/', exist_ok=True)
for _url, _type in zip(ptb_urls, types):
    if not os.path.exists('./ptb/' + _type + '.txt'):
        os.system('wget -O ' + './ptb/' + _type + '.txt ' + _url)

