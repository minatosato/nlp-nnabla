# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator_simple

from tqdm import tqdm

from parametric_functions import lstm
from functions import time_distributed
from functions import time_distributed_softmax_cross_entropy
from functions import get_mask
from functions import expand_dims

from utils import load_data
from utils import wordseq2charseq
from utils import w2i, i2w, c2i, i2c, word_length
from utils import with_padding
from utils import to_cbow_dataset

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

window_size = 5

train_data = load_data('./ptb/train.txt')
x_train, y_train = to_cbow_dataset(train_data, window_size=window_size)

valid_data = load_data('./ptb/valid.txt')
x_valid, y_valid = to_cbow_dataset(valid_data, window_size=window_size)

vocab_size = len(w2i)
embedding_size = 128
hidden_size = 128
batch_size = 32
max_epoch = 100

num_train_batch = len(x_train)//batch_size
num_valid_batch = len(x_valid)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index]

def load_valid_func(index):
    return x_valid[index], y_valid[index]

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(x_valid), batch_size, shuffle=True, with_file_cache=False)

x = nn.Variable([batch_size, window_size*2])
with nn.parameter_scope('embedding_in'):
    h = PF.embed(x, vocab_size, embedding_size)
h = F.mean(h, axis=1)
with nn.parameter_scope('output'):
    y = PF.affine(h, vocab_size, with_bias=False)
t = nn.Variable((batch_size, 1))
entropy = F.softmax_cross_entropy(y, t)
loss = F.mean(entropy)

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())


from trainer import Trainer

trainer = Trainer(inputs=[x, t], loss=loss, solver=solver, save_path='cbow')
trainer.run(train_data_iter, valid_data_iter, epochs=max_epoch)
