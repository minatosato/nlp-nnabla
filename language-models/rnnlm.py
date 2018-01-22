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


from collections import OrderedDict
import pickle
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

from tqdm import tqdm

from layers import SimpleRNN
from layers import TimeDistributed
from layers import TimeDistributedSoftmaxCrossEntropy

"""cuda setting"""
from nnabla.contrib.context import extension_context
ctx = extension_context('cuda.cudnn', device_id=1)
nn.set_default_context(ctx)
""""""

from utils import load_data
from utils import wordseq2charseq
from utils import w2i, i2w, c2i, i2c, word_length

from keras.preprocessing import sequence

train_data = load_data('./ptb/train.txt')
train_data = sequence.pad_sequences(train_data, padding='post')

valid_data = load_data('./ptb/valid.txt')
valid_data = sequence.pad_sequences(valid_data, padding='post')

vocab_size = len(w2i)
sentence_length = 20
embedding_size = 128
hidden = 128
batch_size = 256
max_epoch = 100

x_train = train_data[:, :sentence_length].astype(np.int32)
y_train = train_data[:, 1:sentence_length+1].astype(np.int32)

x_valid = valid_data[:, :sentence_length].astype(np.int32)
y_valid = valid_data[:, 1:sentence_length+1].astype(np.int32)

num_train_batch = len(x_train)//batch_size
num_valid_batch = len(x_valid)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index]

def load_valid_func(index):
    return x_valid[index], y_valid[index]

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(x_valid), batch_size, shuffle=True, with_file_cache=False)

x = nn.Variable((batch_size, sentence_length))
t = nn.Variable((batch_size, sentence_length, 1))
h = PF.embed(x, vocab_size, embedding_size)
h = SimpleRNN(h, hidden, return_sequences=True)
h = TimeDistributed(PF.affine)(h, hidden, name='hidden')
y = TimeDistributed(PF.affine)(h, vocab_size, name='output')

mask = F.sum(F.greater_scalar(t, 0), axis=2) # do not predict 'pad'.
# mask = F.sum(F.sign(t), axis=2) # do not predict 'pad'.
entropy = TimeDistributedSoftmaxCrossEntropy(y, t) * mask
count = F.sum(mask, axis=1)
loss = F.mean(F.div2(F.sum(entropy, axis=1), count))

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-rnnlm')
monitor_perplexity = MonitorSeries('perplexity', monitor, interval=1)
monitor_perplexity_valid = MonitorSeries('perplexity_valid', monitor, interval=1)


for epoch in range(max_epoch):
    train_loss_set = []
    for i in tqdm(range(num_train_batch)):
        x_batch, y_batch = train_data_iter.next()
        y_batch = y_batch.reshape(list(y_batch.shape) + [1])

        x.d, t.d = x_batch, y_batch

        loss.forward(clear_no_need_grad=True)
        train_loss_set.append(loss.d.copy())
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()

    valid_loss_set = []
    for i in range(num_valid_batch):
        x_batch, y_batch = valid_data_iter.next()
        y_batch = y_batch.reshape(list(y_batch.shape) + [1])

        x.d, t.d = x_batch, y_batch

        loss.forward(clear_no_need_grad=True)
        valid_loss_set.append(loss.d.copy())

    monitor_perplexity.add(epoch+1, np.e**np.mean(train_loss_set))
    monitor_perplexity_valid.add(epoch+1, np.e**np.mean(valid_loss_set))






