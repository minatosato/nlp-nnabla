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

from common.utils import load_data
from common.utils import w2i, i2w, c2i, i2c, word_length
from common.utils import with_padding

from common.trainer import Trainer

from utils import to_glove_dataset

import argparse
parser = argparse.ArgumentParser(description='GloVe model training.')
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
valid_data = load_data('./ptb/valid.txt')

vocab_size = len(w2i)
embedding_size = 128
batch_size = 128
max_epoch = 100
k = 5
window_size = 10

central_train, context_train, target_train = to_glove_dataset(train_data, vocab_size=vocab_size, window_size=window_size)
central_valid, context_valid, target_valid = to_glove_dataset(valid_data, vocab_size=vocab_size, window_size=window_size)

num_train_batch = len(central_train)//batch_size
num_valid_batch = len(central_valid)//batch_size

def load_train_func(index):
    return central_train[index], context_train[index], target_train[index]

def load_valid_func(index):
    return central_valid[index], context_valid[index], target_valid[index]

train_data_iter = data_iterator_simple(load_train_func, len(central_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(central_valid), batch_size, shuffle=True, with_file_cache=False)


x_central = nn.Variable((batch_size, ))
x_context = nn.Variable((batch_size, ))

with nn.parameter_scope('central_embedding'):
    central_embedding = PF.embed(x_central, vocab_size, embedding_size)
with nn.parameter_scope('context_embedding'):
    context_embedding = PF.embed(x_context, vocab_size, embedding_size)

with nn.parameter_scope('central_bias'):
    central_bias = PF.embed(x_central, vocab_size, 1)
with nn.parameter_scope('context_bias'):
    context_bias = PF.embed(x_context, vocab_size, 1)

dot_product = F.reshape(
    F.batch_matmul(
        F.reshape(central_embedding, shape=(batch_size, 1, embedding_size)),
        F.reshape(context_embedding, shape=(batch_size, embedding_size, 1))
    ),
    shape=(batch_size, 1)
)

prediction = dot_product + central_bias + context_bias

t = nn.Variable((batch_size, 1))
zero = F.constant(0, shape=(batch_size, 1))
one = F.constant(1, shape=(batch_size, 1))
weight = F.clip_by_value(t / 100, zero, one) ** 0.75
loss = F.sum(weight * ((prediction - F.log(t+1)) ** 2))

# # Create solver.
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

trainer = Trainer(inputs=[x_central, x_context, t], loss=loss, metrics=dict(loss=loss), solver=solver)
trainer.run(train_data_iter, valid_data_iter, epochs=max_epoch)

with open('vectors.txt', 'w') as f:
    f.write('{} {}\n'.format(vocab_size-1, embedding_size))
    with nn.parameter_scope('central_embedding'):
        x = nn.Variable((1, 1))
        y = PF.embed(x, vocab_size, embedding_size)
    for word, i in w2i.items():
        x.d = np.array([[i]])
        y.forward()
        str_vec = ' '.join(map(str, list(y.d.copy()[0][0])))
        f.write('{} {}\n'.format(word, str_vec))

