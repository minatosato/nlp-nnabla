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

import argparse
parser = argparse.ArgumentParser(description='LSTM language model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)


train_data = load_data('./ptb/train.txt', with_bos=True)
train_data = with_padding(train_data, padding_type='post')

valid_data = load_data('./ptb/valid.txt', with_bos=True)
valid_data = with_padding(valid_data, padding_type='post')

vocab_size = len(w2i)
sentence_length = 60
embedding_size = 128
hidden_size = 128
batch_size = 32
max_epoch = 100

x_train = train_data[:, :sentence_length].astype(np.int32)
y_train = train_data[:, 1:sentence_length-1].astype(np.int32)
# y_train[y_train == w2i['<eos>']] = 0

x_valid = valid_data[:, :sentence_length].astype(np.int32)
y_valid = valid_data[:, 1:sentence_length-1].astype(np.int32)
# y_valid[y_valid == w2i['<eos>']] = 0

num_train_batch = len(x_train)//batch_size
num_valid_batch = len(x_valid)//batch_size

def load_train_func(index):
    return x_train[index], y_train[index]

def load_valid_func(index):
    return x_valid[index], y_valid[index]

train_data_iter = data_iterator_simple(load_train_func, len(x_train), batch_size, shuffle=True, with_file_cache=False)
valid_data_iter = data_iterator_simple(load_valid_func, len(x_valid), batch_size, shuffle=True, with_file_cache=False)

x = nn.Variable((batch_size, sentence_length))
mask = get_mask(x)
t = nn.Variable((batch_size, sentence_length-2))
# t = x[:, 1:-1]

with nn.parameter_scope('embedding'):
    h = PF.embed(x, vocab_size, embedding_size) * mask
with nn.parameter_scope('lstm_forward'):
    h_f = lstm(h, hidden_size, mask=mask, return_sequences=True)
with nn.parameter_scope('lstm_backward'):
    h_b = lstm(h[:, ::-1,], hidden_size, mask=mask, return_sequences=True)[:, ::-1,]
h_f = h_f[:, :-2, ]
h_b = h_b[:, 2:, ]
h = F.concatenate(h_f, h_b, axis=2)
with nn.parameter_scope('output'):
    y = time_distributed(PF.affine)(h, vocab_size)

mask = F.sum(get_mask(t), axis=2) # do not predict 'pad'.
entropy = time_distributed_softmax_cross_entropy(y, expand_dims(t, axis=-1)) * mask
count = F.sum(mask, axis=1)
loss = F.mean(F.div2(F.sum(entropy, axis=1), count))

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())


from trainer import Trainer

trainer = Trainer(inputs=[x, t], loss=loss, metrics={'PPL': np.e**loss}, solver=solver, save_path='bilstmlm')
trainer.run(train_data_iter, valid_data_iter, epochs=max_epoch)
