# 
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import nnabla.solvers as S

from nnabla.utils.data_iterator import data_iterator_simple

import numpy as np

from typing import Optional

from utils import with_padding
from utils import load_imdb

from functions import time_distributed
from functions import residual_normalization_wrapper
from functions import position_encoding
from functions import multihead_self_attention
from functions import positionwise_feed_forward
from functions import token_embedding

from functions import get_mask
from functions import get_attention_logit_mask

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Encoder-decoder model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)



batch_size = 64
max_len = 400
embedding_size = 128
vocab_size = 20000
head_num = 8
hopping_num = 2
max_epoch = 20
l2_penalty_coef = 1e-4

x_train, x_test, y_train, y_test = load_imdb(vocab_size)
for i, sentence in enumerate(tqdm(x_train)):
    x_train[i] = [vocab_size] + sentence
for i, sentence in enumerate(tqdm(x_test)):
    x_test[i] = [vocab_size] + sentence

x_train = with_padding(x_train, padding_type='post', max_sequence_length=max_len)
x_test = with_padding(x_test, padding_type='post', max_sequence_length=max_len)
y_train = y_train[:, None]
y_test = y_test[:, None]

num_train_batch = len(x_train)//batch_size
num_dev_batch = len(x_test)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index]

def load_dev_func(index):
    return x_test[index], y_test[index]

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
dev_data_iter = data_iterator_simple(load_dev_func, len(x_test), batch_size, shuffle=True, with_file_cache=False)

vocab_size += 1

def transformer(train=True, droput_ratio=0.1):
    x = nn.Variable((batch_size, max_len))
    t = nn.Variable((batch_size, 1))
    mask = get_mask(x)
    with nn.parameter_scope('embedding_layer'):
        # h = time_distributed(PF.embed)(x, vocab_size, embedding_size) * mask
        h = token_embedding(x, vocab_size, embedding_size)
    h = position_encoding(h)

    if train:
        h = F.dropout(h, p=droput_ratio)

    for i in range(hopping_num):
        with nn.parameter_scope(f'encoder_hopping_{i}'):
            h = residual_normalization_wrapper(multihead_self_attention)(h, head_num, mask=mask, train=train, dropout_ratio=droput_ratio)
            h = residual_normalization_wrapper(positionwise_feed_forward)(h, train=train, dropout_ratio=droput_ratio)
        
    with nn.parameter_scope('output_layer'):
        y = F.sigmoid(PF.affine(h[:, 0, :], 1))


    accuracy = F.mean(F.equal(F.round(y), t))
    loss = F.mean(F.binary_cross_entropy(y, t))

    return x, y, t, accuracy, loss

x, y, t, accuracy, loss = transformer(train=True, droput_ratio=0.1)

# Create solver.
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-transformer')
ce_train = MonitorSeries('ce_train', monitor, interval=1)
ce_dev = MonitorSeries('ce_dev', monitor, interval=1)

for epoch in range(max_epoch):
    train_loss_set = []
    train_acc_set = []
    progress = tqdm(total=train_data_iter.size//batch_size)
    x, y, t, accuracy, loss = transformer(train=True, droput_ratio=0.1)
    for i in range(num_train_batch):
        x.d, t.d = train_data_iter.next()
        loss.forward()
        accuracy.forward()
        solver.zero_grad()
        loss.backward()
        solver.weight_decay(l2_penalty_coef)
        solver.update()
        train_loss_set.append(loss.d.copy())
        train_acc_set.append(accuracy.d.copy())

        progress.set_description(f"epoch: {epoch+1}, train cross entropy: {np.mean(train_loss_set):.5f}, train accuracy: {np.mean(train_acc_set):.5f}")
        progress.update(1)
    progress.close()

    dev_loss_set = []
    dev_acc_set = []
    x, y, t, accuracy, loss = transformer(train=False)
    for i in range(num_dev_batch):
        x.d, t.d = dev_data_iter.next()
        loss.forward()
        accuracy.forward()
        dev_loss_set.append(loss.d.copy())
        dev_acc_set.append(accuracy.d.copy())
    print(f"epoch: {epoch+1}, test accuracy: {np.mean(dev_acc_set):.5f}")
    ce_train.add(epoch+1, train_loss_set)
    ce_dev.add(epoch+1, dev_loss_set)





