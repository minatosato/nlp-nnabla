# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from pathlib import Path
from typing import List
from typing import Tuple
from typing import Dict
from dataclasses import dataclass
from dataclasses import field

from nnabla.utils.data_source_loader import download
from nnabla.utils.data_source_loader import load_npy
from nnabla.utils.data_source_loader import get_data_home


def with_padding(sequences: List[List[int]], padding_type: str = 'post',
                 max_sequence_length: int = 0) -> np.ndarray:
    if max_sequence_length == 0:
        max_sequence_length = max(map(lambda x: len(x), sequences))
    else:
        assert type(max_sequence_length) == int, 'max_sequence_length must be an integer.'
        assert max_sequence_length > 0, 'max_sequence_length must be a positive integer.'

    def _with_padding(sequence: List[int]) -> np.ndarray:
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



@dataclass
class PTBDataset(object):
    with_bos: bool = False
    return_char_info: bool = False
    word_length: int = 20
    w2i: Dict[str, int] = field(default_factory=dict)
    i2w: Dict[int, str] = field(default_factory=dict)
    c2i: Dict[str, int] = field(default_factory=dict)
    i2c: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:       
        self.w2i['pad'] = 0
        self.i2w[0] = 'pad'
        self.w2i['<eos>'] = 1
        self.i2w[1] = '<eos>'

        self.c2i[' '] = 0
        self.i2c[0] = ' '

        if self.with_bos:
            self.w2i['<bos>'] = 2
            self.i2w[2] = '<bos>'

        self.ptb_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{0}.txt'
        
        self.train_data: List[List[int]] = self._load_data('train')
        self.valid_data: List[List[int]] = self._load_data('valid')
        self.test_data: List[List[int]] = self._load_data('test')

    def _load_data(self, type_name: str) -> List[List[int]]:
        url = self.ptb_url.format(type_name)
        with download(url, open_file=True) as f:
            lines: str = f.read().decode('utf-8').replace('\n', '<eos>')

            if self.return_char_info:
                for char in set(lines):
                    if char not in self.c2i:
                        self.c2i[char] = len(self.c2i)
                    if self.c2i[char] not in self.i2c:
                        self.i2c[self.c2i[char]] = char

            words = lines.strip().split()
        dataset = np.ndarray((len(words), ), dtype=np.int32)

        for i, word in enumerate(words):
            if word not in self.w2i:
                self.w2i[word] = len(self.w2i)
            if self.w2i[word] not in self.i2w:
                self.i2w[self.w2i[word]] = word
            dataset[i] = self.w2i[word]

        sentences = []
        sentence = []
        if self.with_bos:
            sentence.append(self.w2i['<bos>'])
        for index in dataset:
            if self.i2w[index] != '<eos>':
                sentence.append(index)
            else:
                sentence.append(index)
                sentences.append(sentence)
                sentence = []
                if self.with_bos:
                    sentence.append(self.w2i['<bos>'])
        return sentences


def load_imdb(vocab_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_name = 'imdb.npz'
    url = f'https://s3.amazonaws.com/text-datasets/{file_name}'
    download(url, open_file=False)

    dataset_path = Path(get_data_home()) / file_name

    unk_index = vocab_size - 1
    raw = load_npy(dataset_path)
    ret = dict()
    for k, v in raw.items():
        if 'x' in k:
            for i, sentence in enumerate(v):
                v[i] = [word if word < unk_index else unk_index for word in sentence]
        ret[k] = v
    return ret['x_train'], ret['x_test'], ret['y_train'], ret['y_test']


def load_enja_parallel_data(lang: str):
    url = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/{0}.{1}'
    data_types = ['train', 'dev', 'test']
    url_list = [url.format(data_type, lang) for data_type in data_types]

    w2i = {}
    i2w = {}

    w2i['pad'] = 0
    i2w[0] = 'pad'
    w2i['<bos>'] = 1
    i2w[1] = '<bos>'
    w2i['<eos>'] = 2
    i2w[2] = '<eos>'

    def _load_data(url: str):
        with download(url, open_file=True) as f:
            lines = f.read().decode('utf-8').replace('\n', ' <eos> ')
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
        for index in dataset:
            if i2w[index] != '<eos>':
                sentence.append(index)
            else:
                sentences.append(sentence)
                sentence = []
        return sentences

    return list(map(_load_data, url_list)) + [w2i, i2w]


