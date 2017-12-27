# MIT License
# 
# Copyright (c) 2016-2017 Minato Sato
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


from collections import OrderedDict
import pickle
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

from tqdm import tqdm

"""cuda setting"""
from nnabla.contrib.context import extension_context
ctx = extension_context("cuda.cudnn", device_id=1)
nn.set_default_context(ctx)
""""""


w2i = {}
i2w = {}

w2i['pad'] = 0
i2w[0] = 'pad'

def load_data(filename):
    global w2i
    with open(filename) as f:
        words = f.read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words), ), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in w2i:
            w2i[word] = len(w2i)
        if w2i[word] not in i2w:
            i2w[w2i[word]] = word
        dataset[i] = w2i[word]
    return dataset

def data2sentences(data):
    global w2i, i2w
    sentences = []
    sentence = []
    for index in data:
        if i2w[index] != '<eos>':
            sentence.append(index)
        else:
            sentence.append(index)
            sentences.append(sentence)
            sentence = []
    return sentences

from keras.preprocessing import sequence

train_data = load_data('./ptb/train.txt')
train_data = data2sentences(train_data)
train_data = sequence.pad_sequences(train_data, padding="post")

vocab_size = len(w2i)
sentence_length = 40
embedding_size = 512
hidden = 512
batch_size = 64
max_epoch = 100

x_train = train_data[:, :sentence_length].astype(np.int32)
y_train = train_data[:, 1:sentence_length+1].astype(np.int32)

num_batch = len(x_train)//batch_size

def load_func(index):
    return x_train[index], y_train[index]

data_iter = data_iterator_simple(load_func, len(x_train), batch_size, shuffle=False, with_file_cache=True)

def SimpleRNN(inputs, units, return_sequences=False):
    '''
    A vanilla recurrent neural network layer
    Args:
        inputs (nnabla.Variable): A shape of [B, SentenceLength, EmbeddingSize].
        units (int): Dimensionality of the output space.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
    Returns:
        nn.Variable: A shape [B, SentenceLength, units].
        or
        nn.Variable: A shape [B, units]
    '''

    hs = []
    batch_size = inputs.shape[0]
    sentence_length = inputs.shape[1]

    inputs = F.split(inputs, axis=1)

    h = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))
    for x in inputs:
        with nn.parameter_scope('x2h'):
            x2h = PF.affine(x, units, with_bias=False)
        with nn.parameter_scope('h2h'):
            h2h = PF.affine(h, units)
        h = F.tanh(x2h + h2h)
        hs.append(h)

    if return_sequences:
        hs = F.concatenate(*hs, axis=0)
        hs = F.reshape(hs, (batch_size, sentence_length, units))
    else:
        return hs[-1]
    return hs

def TimeDistributedAffine(x, units, name='output'):
    return PF.affine(x, units, base_axis=2, name=name)

def TimeDistributedSoftmaxCrossEntropy(y_pred, y_true):
    '''
    A time distributed softmax crossentropy
    Args:
        y_pred (nnabla.Variable): A shape of [B, SentenceLength, O]. # one-hot
        y_true (nnabla.Variable): A shape of [B, SentenceLength, 1]. # index
    Returns:
        nn.Variable: A shape [B, SentenceLength].
    '''

    ret = []
    for y_p, y_t in zip(F.split(y_pred, axis=1), F.split(y_true, axis=1)):
        ret.append(F.softmax_cross_entropy(y_p, y_t))
    return F.concatenate(*ret)

x = nn.Variable((batch_size, sentence_length))
t = nn.Variable((batch_size, sentence_length, 1))
h = PF.embed(x, vocab_size, embedding_size)
h = SimpleRNN(h, hidden, return_sequences=True)
y = TimeDistributedAffine(h, vocab_size)

entropy = TimeDistributedSoftmaxCrossEntropy(y, t) * F.sum(F.sign(t), axis=2) # do not predict 'pad'.
count = F.sum(F.sum(F.sign(t), axis=2), axis=1)
loss = F.mean(F.sum(entropy, axis=1) / count)

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-lstm')
monitor_loss = MonitorSeries("Training loss", monitor, interval=1)

for epoch in range(max_epoch):
    loss_set = []
    for i in tqdm(range(num_batch)):
        x_batch, y_batch = data_iter.next()
        y_batch = y_batch.reshape(list(y_batch.shape) + [1])

        x.d, t.d = x_batch, y_batch

        solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True)
        loss.forward()
        loss_set.append(np.exp(loss.d.copy()))
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()
    monitor_loss.add(epoch, np.mean(loss_set))






