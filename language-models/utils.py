# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os

from collections import Counter

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


def load_data(filename, with_bos=False):
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

def wordseq2charseq(data):
    global word_length
    data = np.repeat(np.expand_dims(data, axis=2), word_length, axis=2)
    data[:, :, 1:] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            word = data[i][j][0]
            if word != 0:
                for k, char in enumerate(i2w[word]):
                    data[i][j][k] = c2i[char]
            else:
                data[i, j, :] = 0
    return data

def calc_sampling_prob(sentences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    oneline: List[int] = []
    for sentence in sentences:
        oneline.extend(sentence)
    counter = Counter(oneline)
    words = np.array(list(counter.keys()), dtype=np.int32)
    prob = np.array(list(counter.values())) / sum(counter.values())
    prob = np.power(prob, 0.75)
    prob = prob / np.sum(prob)
    return words, prob

def negative_sampling(target: int, words: np.ndarray, prob: np.ndarray, k: int = 5):
    ret = np.random.choice(words, size=k, p=prob)
    while target in ret:
        ret = np.random.choice(words, size=k, p=prob)
    return ret

def to_cbow_dataset(sentences: List[List[int]], window_size: int = 1, ns: bool = False):
    contexts: List[List[int]] = []
    targets: List[int] = []

    if ns:
        negative_samples: List[np.ndarray] = []
        words, prob = calc_sampling_prob(sentences)

    for sentence in sentences:
        for _index in range(window_size, len(sentence)-window_size):
            targets.append(sentence[_index])
            if ns:
                negative_samples.append(negative_sampling(sentence[_index], words, prob))
            ctx: List[int] = []
            for t in range(-window_size, window_size+1):
                if t == 0:
                    continue
                ctx.append(sentence[_index + t])
            contexts.append(ctx)
    
    ret = [np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)[:, None]]
    if ns:
        ret.append(np.array(negative_samples, dtype=np.int32))
    return tuple(ret)

def to_cooccurrences(sentences: List[List[int]], window_size: int = 1) -> np.ndarray:
    matrix = np.zeros((len(w2i), len(w2i)), dtype=np.int32)
    for sentence in sentences:
        for word_index in range(window_size, len(sentence)-window_size):
            for t in range(-window_size, window_size+1):
                if t == 0:
                    continue
                context_word_index = sentence[word_index + t]
                matrix[sentence[word_index]][context_word_index] += 1
    return matrix

def to_glove_dataset(sentences: List[List[int]], window_size: int = 1) -> np.ndarray:
    matrix = to_cooccurrences(sentences, window_size=window_size)
    central: List[int] = []
    context: List[int] = []
    y: List[int] = []
    from itertools import combinations

    for i, j in combinations(range(len(w2i)), 2):
        central.append(i)
        context.append(j)
        y.append(matrix[i, j])

    return np.array(central), np.array(context), np.array(y)[:, None]



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

