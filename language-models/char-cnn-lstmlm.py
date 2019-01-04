# 
# Copyright (c) 2017-2019 Minato Sato
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

from parametric_functions import lstm
from parametric_functions import highway
from functions import time_distributed
from functions import time_distributed_softmax_cross_entropy
from functions import expand_dims
from functions import get_mask

from utils import load_data
from utils import wordseq2charseq
from utils import w2i, i2w, c2i, i2c, word_length
from utils import with_padding

import argparse
parser = argparse.ArgumentParser(description='Char-cnn-lstm language model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)


train_data = load_data('./ptb/train.txt')
train_data = with_padding(train_data, padding_type='post')

valid_data = load_data('./ptb/valid.txt')
valid_data = with_padding(valid_data, padding_type='post')

sentence_length = 60
batch_size = 100
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
dropout_ratio = 0.5

def build_model(train=True, get_embeddings=False):
    x = nn.Variable((batch_size, sentence_length, word_length))
    mask = expand_dims(F.sign(x), axis=-1)
    t = nn.Variable((batch_size, sentence_length))

    with nn.parameter_scope('char_embedding'):
        h = PF.embed(x, char_vocab_size, char_embedding_dim) * mask
    h = F.transpose(h, (0, 3, 1, 2))
    output = []
    for f, f_size in zip(filters, filster_sizes):
        _h = PF.convolution(h, f, kernel=(1, f_size), pad=(0, f_size//2), name='conv_{}'.format(f_size))
        _h = F.max_pooling(_h, kernel=(1, word_length))
        output.append(_h)
    h = F.concatenate(*output, axis=1)
    h = F.transpose(h, (0, 2, 1, 3))

    mask = get_mask(F.sum(x, axis=2))
    embeddings = F.reshape(h, (batch_size, sentence_length, sum(filters))) * mask

    if get_embeddings:
        return x, embeddings

    with nn.parameter_scope('highway1'):
        h = time_distributed(highway)(embeddings)
    with nn.parameter_scope('highway2'):
        h = time_distributed(highway)(h)
    with nn.parameter_scope('lstm1'):
        h = lstm(h, lstm_size, mask=mask, return_sequences=True)
    with nn.parameter_scope('lstm2'):
        h = lstm(h, lstm_size, mask=mask, return_sequences=True)
    with nn.parameter_scope('hidden'):
        h = F.relu(time_distributed(PF.affine)(h, lstm_size))
    if train:
        h = F.dropout(h, p=dropout_ratio)
    with nn.parameter_scope('output'):
        y = time_distributed(PF.affine)(h, word_vocab_size)

    mask = F.sign(t) # do not predict 'pad'.
    entropy = time_distributed_softmax_cross_entropy(y, expand_dims(t, axis=-1)) * mask
    count = F.sum(mask, axis=1)
    loss = F.mean(F.div2(F.sum(entropy, axis=1), count))
    return x, t, loss

x, t, loss = build_model()

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())

from trainer import Trainer

x, t, loss = build_model(train=True)
trainer = Trainer(inputs=[x, t], loss=loss, metrics={'PPL': np.e**loss}, solver=solver, save_path='char-cnn-lstmlm')
trainer.run(train_data_iter, valid_data_iter, epochs=max_epoch)

for epoch in range(max_epoch):
    x, t, loss = build_model(train=True)
    trainer.update_variables(inputs=[x, t], loss=loss, metrics={'PPL': np.e**loss})
    trainer.run(train_data_iter, None, epochs=1, verbose=1)
    
    x, t, loss = build_model(train=False)
    trainer.update_variables(inputs=[x, t], loss=loss, metrics={'PPL': np.e**loss})
    trainer.evaluate(valid_data_iter, verbose=1)

# nn.load_parameters('char-cnn-lstm_best.h5')

# batch_size = 1
# sentence_length = 1
# x, embeddings = build_model(get_embeddings=True)

# W = np.zeros((len(w2i), sum(filters)))
# for i, word in enumerate(w2i):
#     vec = wordseq2charseq([[w2i[word]]])
#     x.d = vec
#     embeddings.forward(clear_no_need_grad=True)
#     W[w2i[word], :] = embeddings.d[0][0]

# def get_word_from_id(id):
#     return i2w[id]

# from sklearn.metrics.pairwise import cosine_similarity

# def get_top_k(word, k=5):
#     global W
#     if word not in w2i:
#         w2i[word] = len(w2i)
#         i2w[w2i[word]] = word
#         W = np.zeros((len(w2i), sum(filters)))
#         for i, w in enumerate(w2i):
#             vec = wordseq2charseq([[w2i[w]]])
#             x.d = vec
#             embeddings.forward()
#             W[w2i[w], :] = embeddings.d[0][0]
#     cosine_similarity_set = cosine_similarity([W[w2i[word]]], W)[0]
#     top_k = cosine_similarity_set.argsort()[-(k+1):-1][::-1]
#     return list(map(get_word_from_id, top_k))

# search_words = ['while', 'his', 'you', 'richard', 'trading', 'computer-aided', 'misinformed', 'looooook']

# for query in search_words:
#     print("----------------")
#     print(query + ' -> ', end='')
#     print(get_top_k(query))
# print("----------------")





