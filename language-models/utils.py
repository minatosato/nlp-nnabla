# MIT License
# 
# Copyright (c) 2017-2018 Minato Sato
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

w2i = {}
i2w = {}

c2i = {}
i2c = {}

w2i['pad'] = 0
i2w[0] = 'pad'

c2i[' '] = 0
i2c[0] = ' '

word_length = 20


def load_data(filename):
    global w2i, i2w
    global c2i, i2c
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
    for index in dataset:
        if i2w[index] != '<eos>':
            sentence.append(index)
        else:
            sentence.append(index)
            sentences.append(sentence)
            sentence = []
    return sentences

def wordseq2charseq(data):
    global word_length
    data = np.repeat(np.expand_dims(data, axis=2), word_length, axis=2)
    data[:, :, 1:] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            word = data[i][j][0]
            for k, char in enumerate(i2w[word]):
                data[i][j][k] = c2i[char]
    return data
