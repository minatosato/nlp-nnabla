# 
# Copyright (c) 2018 Minato Sato
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

from pathlib import Path
from tqdm import tqdm

from common.parametric_functions import lstm
from common.functions import time_distributed
from common.functions import frobenius
from common.functions import batch_eye
from common.functions import get_mask
from common.utils import load_imdb
from common.utils import with_padding
from common.trainer import Trainer

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

max_len: int = 400
batch_size: int = 64
embedding_size: int = 300
hidden_size: int = 300
da: int = 350
r: int = 30
output_mlp_size: int = 3000
max_epoch: int = 20
vocab_size: int = 20000
dropout_ratio: float = 0.3
attention_penalty_coef: float = 0.03
l2_penalty_coef: float = 1e-4

x_train, x_test, y_train, y_test = load_imdb(vocab_size)

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


def build_self_attention_model(train=True):
    x = nn.Variable((batch_size, max_len))
    t = nn.Variable((batch_size, 1))
    mask = get_mask(x)
    attention_mask = (F.constant(1, shape=mask.shape) - mask) * F.constant(np.finfo(np.float32).min, shape=mask.shape)
    with nn.parameter_scope('embedding'):
        h = time_distributed(PF.embed)(x, vocab_size, embedding_size) * mask
    with nn.parameter_scope('forward'):
        h_f = lstm(h,            hidden_size, mask=mask, return_sequences=True, return_state=False)
    with nn.parameter_scope('backward'):
        h_b = lstm(h[:, ::-1, ], hidden_size, mask=mask, return_sequences=True, return_state=False)[:, ::-1, ]
    h = F.concatenate(h_f, h_b, axis=2)
    if train:
        h = F.dropout(h, p=dropout_ratio)
    with nn.parameter_scope('da'):
        a = F.tanh(time_distributed(PF.affine)(h, da))
        if train:
            a = F.dropout(a, p=dropout_ratio)
    with nn.parameter_scope('r'):
        a = time_distributed(PF.affine)(a, r)
        if train:
            a = F.dropout(a, p=dropout_ratio)
        a = F.softmax(a + attention_mask, axis=1)
    m = F.batch_matmul(a, h, transpose_a=True)
    with nn.parameter_scope('output_mlp'):
        output = F.relu(PF.affine(m, output_mlp_size))
        if train:
            output = F.dropout(output, p=dropout_ratio)
    with nn.parameter_scope('output'):
        y = F.sigmoid(PF.affine(output, 1))

    accuracy = F.mean(F.equal(F.round(y), t))
    loss = F.mean(F.binary_cross_entropy(y, t)) + attention_penalty_coef * frobenius(F.batch_matmul(a, a, transpose_a=True) - batch_eye(batch_size, r))
    return x, t, accuracy, loss

# Create solver.
x, t, accuracy, loss = build_self_attention_model(train=True)
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

x, t, accuracy, loss = build_self_attention_model(train=True)
trainer = Trainer(inputs=[x, t], loss=loss, metrics={'cross entropy': loss, 'accuracy': accuracy}, solver=solver)
for epoch in range(max_epoch):
    x, t, accuracy, loss = build_self_attention_model(train=True)
    trainer.update_variables(inputs=[x, t], loss=loss, metrics={'cross entropy': loss, 'accuracy': accuracy})
    trainer.run(train_data_iter, None, epochs=1, verbose=1)
    
    x, t, accuracy, loss = build_self_attention_model(train=False)
    trainer.update_variables(inputs=[x, t], loss=loss, metrics={'cross entropy': loss, 'accuracy': accuracy})
    trainer.evaluate(dev_data_iter, verbose=1)

