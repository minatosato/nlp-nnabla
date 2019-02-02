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
import nnabla.monitor as M
from nnabla.experimental.trainers import Trainer
from nnabla.experimental.trainers import Updater
from nnabla.experimental.trainers import Evaluator

from nnabla.utils.data_iterator import data_iterator_simple

from common.parametric_functions import lstm
from common.functions import time_distributed
from common.functions import time_distributed_softmax_cross_entropy
from common.functions import get_mask
from common.functions import expand_dims

from common.utils import PTBDataset
from common.utils import with_padding

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

ptb_dataset = PTBDataset()

vocab_size = len(ptb_dataset.w2i)
embedding_size = 128
batch_size = 128
max_epoch = 100
window_size = 10

central_train, context_train, target_train = to_glove_dataset(ptb_dataset.train_data, vocab_size=vocab_size, window_size=window_size)
central_valid, context_valid, target_valid = to_glove_dataset(ptb_dataset.valid_data, vocab_size=vocab_size, window_size=window_size)

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
loss = F.sum(weight * ((prediction - F.log(t)) ** 2))

# Create solver.
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

# Create monitor
monitor = M.Monitor('./log')
monitor_loss = M.MonitorSeries("Training loss", monitor, interval=1000)
monitor_valid_loss = M.MonitorSeries("Validation loss", monitor, interval=1)
monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=1000)


# Create updater
def train_data_feeder():
    x_central.d, x_context.d, t.d = train_data_iter.next()
def update_callback_on_finish(i):
    monitor_loss.add(i, loss.d)
    monitor_time.add(i)
updater = Updater(solver=solver,
                  loss=loss,
                  data_feeder=train_data_feeder,
                  update_callback_on_finish=update_callback_on_finish)

# Evaluator
def valid_data_feeder():
    x_central.d, x_context.d, t.d = valid_data_iter.next()
def eval_callback_on_finish(i, ve):
    monitor_valid_loss.add(i, ve)
evaluator = Evaluator(loss,
                      data_feeder=valid_data_feeder,
                      val_iter=valid_data_iter.size // batch_size,
                      callback_on_finish=eval_callback_on_finish)

trainer = Trainer(updater=updater, evaluator=evaluator, model_save_path='./log',
                  max_epoch=3, iter_per_epoch=train_data_iter.size//batch_size)
trainer.train()

with open('vectors.txt', 'w') as f:
    f.write('{} {}\n'.format(vocab_size-1, embedding_size))
    with nn.parameter_scope('central_embedding'):
        x = nn.Variable((1, 1))
        y = PF.embed(x, vocab_size, embedding_size)
    for word, i in ptb_dataset.w2i.items():
        x.d = np.array([[i]])
        y.forward()
        str_vec = ' '.join(map(str, list(y.d.copy()[0][0])))
        f.write('{} {}\n'.format(word, str_vec))

