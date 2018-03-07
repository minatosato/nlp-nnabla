# 
# Copyright (c) 2017-2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

def load_data(dirname, lang):
    files = list(map(lambda x: dirname+x+'.'+lang, ['/train', '/dev', '/test']))

    w2i = {}
    i2w = {}

    w2i['pad'] = 0
    i2w[0] = 'pad'
    # w2i['<bos>'] = 1
    # i2w[1] = '<bos>'
    w2i['<eos>'] = 1
    i2w[1] = '<eos>'

    def _load_data(filename):
        with open(filename) as f:
            lines = f.read().replace('\n', ' <eos> ')
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

    return list(map(_load_data, files)) + [w2i, i2w]


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

