# 
# Copyright (c) 2017-2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict
import pickle
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

from tqdm import tqdm

from layers import LSTM
from layers import Highway
from layers import TimeDistributed
from layers import TimeDistributedSoftmaxCrossEntropy

"""cuda setting"""
from nnabla.contrib.context import extension_context
ctx = extension_context('cuda.cudnn', device_id=0)
nn.set_default_context(ctx)
""""""

from utils import load_data
from utils import wordseq2charseq
from utils import w2i, i2w, c2i, i2c, word_length
from utils import with_padding

train_data = load_data('./ptb/train.txt')
train_data = with_padding(train_data, padding_type='post')

valid_data = load_data('./ptb/valid.txt')
valid_data = with_padding(valid_data, padding_type='post')

sentence_length = 20
batch_size = 256
max_epoch = 300

word_vocab_size = len(w2i)
char_vocab_size = len(c2i)

x_train = train_data[:, :sentence_length].astype(np.int32)
y_train = train_data[:, 1:sentence_length+1].astype(np.int32)

x_train = wordseq2charseq(x_train)

x_valid = valid_data[:, :sentence_length].astype(np.int32)
y_valid = valid_data[:, 1:sentence_length+1].astype(np.int32)

x_valid = wordseq2charseq(x_valid)

num_train_batch = len(x_train)//batch_size
num_valid_batch = len(x_valid)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index]

def load_valid_func(index):
    return x_valid[index], y_valid[index]

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(x_valid), batch_size, shuffle=True, with_file_cache=False)

char_embedding_dim = 16
lstm_size = 650
filters = [50, 100, 150, 200, 200, 200, 200]
filster_sizes = [1, 2, 3, 4, 5, 6, 7]

def build_model(get_embeddings=False):
    x = nn.Variable((batch_size, sentence_length, word_length))
    h = PF.embed(x, char_vocab_size, char_embedding_dim)
    h = F.transpose(h, (0, 3, 1, 2))
    output = []
    for f, f_size in zip(filters, filster_sizes):
        _h = PF.convolution(h, f, kernel=(1, f_size), pad=(0, f_size//2), name='conv_{}'.format(f_size))
        _h = F.max_pooling(_h, kernel=(1, word_length))
        output.append(_h)
    h = F.concatenate(*output, axis=1)
    h = F.transpose(h, (0, 2, 1, 3))
    embeddings = F.reshape(h, (batch_size, sentence_length, sum(filters)))

    if get_embeddings:
        return x, embeddings

    h = TimeDistributed(Highway)(embeddings, name='highway1')
    h = TimeDistributed(Highway)(h, name='highway2')
    h = LSTM(h, lstm_size, return_sequences=True, name='lstm1')
    h = LSTM(h, lstm_size, return_sequences=True, name='lstm2')
    h = TimeDistributed(PF.affine)(h, lstm_size, name='hidden')
    y = TimeDistributed(PF.affine)(h, word_vocab_size, name='output')
    t = nn.Variable((batch_size, sentence_length, 1))

    mask = F.sum(F.sign(t), axis=2) # do not predict 'pad'.
    entropy = TimeDistributedSoftmaxCrossEntropy(y, t) * mask
    count = F.sum(mask, axis=1)
    loss = F.mean(F.div2(F.sum(entropy, axis=1), count))
    return x, t, loss

x, t, loss = build_model()

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-char-cnn-lstm')
monitor_perplexity = MonitorSeries('perplexity', monitor, interval=1)
monitor_perplexity_valid = MonitorSeries('perplexity_valid', monitor, interval=1)

best_dev_loss = 9999

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
        solver.weight_decay(1e-5)
        solver.update()

    dev_loss_set = []
    for i in range(num_valid_batch):
        x_batch, y_batch = valid_data_iter.next()
        y_batch = y_batch.reshape(list(y_batch.shape) + [1])

        x.d, t.d = x_batch, y_batch

        loss.forward(clear_no_need_grad=True)
        dev_loss_set.append(loss.d.copy())

    monitor_perplexity.add(epoch+1, np.e**np.mean(train_loss_set))
    monitor_perplexity_valid.add(epoch+1, np.e**np.mean(dev_loss_set))

    dev_loss = np.e**np.mean(dev_loss_set)
    if best_dev_loss > dev_loss:
        best_dev_loss = dev_loss
        print('best dev loss updated! {}'.format(dev_loss))
        nn.save_parameters('char-cnn-lstm_best.h5')

nn.load_parameters('char-cnn-lstm_best.h5')

batch_size = 1
sentence_length = 1
x, embeddings = build_model(get_embeddings=True)

W = np.zeros((len(w2i), sum(filters)))
for i, word in enumerate(w2i):
    vec = wordseq2charseq([[w2i[word]]])
    x.d = vec
    embeddings.forward()
    W[w2i[word], :] = embeddings.d[0][0]

def get_word_from_id(id):
    return i2w[id]

from sklearn.metrics.pairwise import cosine_similarity

def get_top_k(word, k=5):
    global W
    if word not in w2i:
        w2i[word] = len(w2i)
        i2w[w2i[word]] = word
        W = np.zeros((len(w2i), sum(filters)))
        for i, w in enumerate(w2i):
            vec = wordseq2charseq([[w2i[w]]])
            x.d = vec
            embeddings.forward()
            W[w2i[w], :] = embeddings.d[0][0]
    cosine_similarity_set = cosine_similarity([W[w2i[word]]], W)[0]
    top_k = cosine_similarity_set.argsort()[-(k+1):-1][::-1]
    return list(map(get_word_from_id, top_k))

search_words = ['while', 'his', 'you', 'richard', 'trading', 'computer-aided', 'misinformed', 'looooook']

for query in search_words:
    print("----------------")
    print(query + ' -> ', end='')
    print(get_top_k(query))
print("----------------")





