# 
# Copyright (c) 2018 Minato Sato
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

from pathlib import Path
from tqdm import tqdm

from parametric_functions import lstm
from functions import time_distributed
from utils import with_padding

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
dataset_path = Path('./imdb.npz')

if not dataset_path.exists():
    import os
    os.system('wget https://s3.amazonaws.com/text-datasets/imdb.npz')

def load_imdb():
    raw = np.load(dataset_path.as_posix())
    ret = dict()
    for k, v in raw.items():
        ret[k] = v
    return ret['x_train'], ret['x_test'], ret['y_train'], ret['y_test']

x_train, x_test, y_train, y_test = load_imdb()

max_len: int = 80
batch_size: int = 128
embedding_size: int = 300
hidden_size: int = 300
da: int = 350
r: int = 30
output_mlp_size: int = 2000
max_epoch: int = 20
vocab_size: int = max(max(map(lambda x: max(x), x_train)),
                      max(map(lambda x: max(x), x_test))) + 1

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


x = nn.Variable((batch_size, max_len))
t = nn.Variable((batch_size, 1))
mask = F.reshape(F.sign(x), shape=(batch_size, max_len, 1))
attention_mask = (F.constant(1, shape=mask.shape) - mask) * F.constant(np.finfo(np.float32).min, shape=mask.shape)
with nn.parameter_scope('embedding'):
    h = time_distributed(PF.embed)(x, vocab_size, embedding_size) * mask
h_f = lstm(h,            hidden_size, return_sequences=True, return_state=False)
h_b = lstm(h[:, ::-1, ], hidden_size, return_sequences=True, return_state=False)
h = F.concatenate(h_f, h_b, axis=2)
with nn.parameter_scope('da'):
    a = F.tanh(time_distributed(PF.affine)(h, da))
with nn.parameter_scope('r'):
    a = F.softmax(time_distributed(PF.affine)(a, r) + attention_mask, axis=1)
m = F.batch_matmul(a, h, transpose_a=True)
with nn.parameter_scope('output_mlp'):
    output = F.relu(PF.affine(m, output_mlp_size))
with nn.parameter_scope('output'):
    y = F.sigmoid(PF.affine(output, 1))

accuracy = F.mean(F.equal(F.round(y), t))
penalty = F.batch_matmul(a, a, transpose_a=True) - F.broadcast(F.reshape(F.matrix_diag(F.constant(1, shape=(r,))), shape=(1, r, r)), shape=(batch_size, r, r))
loss = F.mean(F.binary_cross_entropy(y, t)) + F.mean(F.sum(F.sum(penalty ** 2, axis=2), axis=1))

# Create solver.
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-fasttext')
ce_train = MonitorSeries('ce_train', monitor, interval=1)
ce_dev = MonitorSeries('ce_dev', monitor, interval=1)

for epoch in range(max_epoch):
    train_loss_set = []
    train_acc_set = []
    progress = tqdm(total=train_data_iter.size//batch_size)
    for i in range(num_train_batch):
        x.d, t.d = train_data_iter.next()
        loss.forward()
        accuracy.forward()
        solver.zero_grad()
        loss.backward()
        solver.update()
        train_loss_set.append(loss.d.copy())
        train_acc_set.append(accuracy.d.copy())

        progress.set_description(f"epoch: {epoch+1}, train cross entropy: {np.mean(train_loss_set):.5f}, train accuracy: {np.mean(train_acc_set):.5f}")
        progress.update(1)
    progress.close()

    dev_loss_set = []
    dev_acc_set = []
    for i in range(num_dev_batch):
        x.d, t.d = train_data_iter.next()
        loss.forward()
        accuracy.forward()
        dev_loss_set.append(loss.d.copy())
        dev_acc_set.append(accuracy.d.copy())
    print(f"epoch: {epoch+1}, test accuracy: {np.mean(dev_acc_set):.5f}")
    ce_train.add(epoch+1, train_loss_set)
    ce_dev.add(epoch+1, dev_loss_set)



