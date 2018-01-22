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
