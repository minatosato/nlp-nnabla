# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.append('../../') 

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

from tqdm import tqdm

from common.parametric_functions import lstm
from common.functions import time_distributed
from common.functions import time_distributed_softmax_cross_entropy
from common.functions import get_mask
from common.functions import expand_dims

from common.utils import PTBDataset
from common.utils import with_padding

from utils import to_cbow_dataset
from utils import calc_sampling_prob
from utils import negative_sampling

from common.trainer import Trainer

from typing import List

import argparse
parser = argparse.ArgumentParser(description='CBOW model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)

window_size = 2

ptb_dataset = PTBDataset()

train_data = ptb_dataset.train_data
valid_data = ptb_dataset.valid_data

words, prob = calc_sampling_prob(train_data)
x_train, y_train = to_cbow_dataset(train_data, window_size=window_size)
x_valid, y_valid = to_cbow_dataset(valid_data, window_size=window_size)

vocab_size = len(ptb_dataset.w2i)
embedding_size = 128
batch_size = 128
max_epoch = 10
k = 5

num_train_batch = len(x_train)//batch_size
num_valid_batch = len(x_valid)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index], negative_sampling(y_train[index], words, prob)

def load_valid_func(index):
    return x_valid[index], y_valid[index], negative_sampling(y_valid[index], words, prob)

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(x_valid), batch_size, shuffle=True, with_file_cache=False)

x = nn.Variable([batch_size, window_size*2])
with nn.parameter_scope('W_in'):
    h = PF.embed(x, vocab_size, embedding_size)
h = F.mean(h, axis=1)
h = expand_dims(h, axis=-1) # (batch_size, embedding_size, 1)
t = nn.Variable([batch_size, 1])
t_neg = nn.Variable([batch_size, k])
with nn.parameter_scope('W_out'):
    _t = PF.embed(t, vocab_size, embedding_size) # (batch_size, 1, embedding_size)
    _t_neg = PF.embed(t_neg, vocab_size, embedding_size) # (batch_size, k, embedding_size)

t_score = F.sigmoid(F.reshape(F.batch_matmul(_t, h), shape=(batch_size, 1)))
t_neg_score = F.sigmoid(F.reshape(F.batch_matmul(_t_neg, h), shape=(batch_size, k)))

t_loss = F.binary_cross_entropy(t_score, F.constant(1, shape=(batch_size, 1)))
t_neg_loss = F.binary_cross_entropy(t_neg_score, F.constant(0, shape=(batch_size, k)))

loss = F.mean(F.sum(t_loss, axis=1) + F.sum(t_neg_loss, axis=1))


# Create solver.
solver = S.Adam()
solver.set_parameters(nn.get_parameters())


trainer = Trainer(inputs=[x, t, t_neg], loss=loss, solver=solver)
trainer.run(train_data_iter, valid_data_iter, epochs=max_epoch)

with open('vectors.txt', 'w') as f:
    f.write('{} {}\n'.format(vocab_size-1, embedding_size))
    with nn.parameter_scope('W_in'):
        x = nn.Variable((1, 1))
        y = PF.embed(x, vocab_size, embedding_size)
    for word, i in ptb_dataset.w2i.items():
        x.d = np.array([[i]])
        y.forward()
        str_vec = ' '.join(map(str, list(y.d.copy()[0][0])))
        f.write('{} {}\n'.format(word, str_vec))

