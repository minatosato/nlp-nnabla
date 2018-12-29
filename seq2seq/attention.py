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

from parametric_functions import lstm
from parametric_functions import lstm_cell
from parametric_functions import global_attention

from functions import get_mask
from functions import get_attention_logit_mask
from functions import time_distributed
from functions import time_distributed_softmax_cross_entropy

import argparse
parser = argparse.ArgumentParser(description='Encoder-decoder with attention model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)

# nn.load_parameters('new_attention.h5')

from utils import load_data
from utils import with_padding

train_source, dev_source, test_source, w2i_source, i2w_source = load_data('./data', 'en')
train_source = with_padding(train_source, padding_type='post')[:,::-1].astype(np.int32)
dev_source = with_padding(dev_source, padding_type='post')[:,::-1].astype(np.int32)
test_source = with_padding(test_source, padding_type='post')[:,::-1].astype(np.int32)

train_target, dev_target, test_target, w2i_target, i2w_target = load_data('./data', 'ja')
train_target = with_padding(train_target, padding_type='post').astype(np.int32)
dev_target = with_padding(dev_target, padding_type='post').astype(np.int32)
test_target = with_padding(test_target, padding_type='post').astype(np.int32)

vocab_size_source = len(w2i_source)
vocab_size_target = len(w2i_target)
sentence_length_source = train_source.shape[1]
sentence_length_target = train_target.shape[1]
embedding_size = 1024
hidden = 1024
batch_size = 64
max_epoch = 500

num_train_batch = len(train_source)//batch_size
num_dev_batch = len(dev_source)//batch_size

def load_train_func(index):
    return train_source[index], train_target[index]

def load_dev_func(index):
    return dev_source[index], dev_target[index]

train_data_iter = data_iterator_simple(load_train_func, len(train_source), batch_size, shuffle=True, with_file_cache=False)
dev_data_iter = data_iterator_simple(load_dev_func, len(dev_source), batch_size, shuffle=True, with_file_cache=False)


def build_model():
    x = nn.Variable((batch_size, sentence_length_source))
    mask = get_mask(x)
    y = nn.Variable((batch_size, sentence_length_target))
    
    enc_input = time_distributed(PF.embed)(x, vocab_size_source, embedding_size, name='enc_embeddings') * mask
    # -> (batch_size, sentence_length_source, embedding_size)

    dec_input = F.concatenate(F.constant(w2i_target['<bos>'], shape=(batch_size, 1)),
                              y[:, :sentence_length_target-1],
                              axis=1)

    dec_input = time_distributed(PF.embed)(dec_input, vocab_size_target, embedding_size, name='dec_embeddings')
    # -> (batch_size, sentence_length_target, embedding_size)

    # encoder
    with nn.parameter_scope('encoder'):
        enc_output, c, h = lstm(enc_input, hidden, mask=mask, return_sequences=True, return_state=True)
        # -> (batch_size, sentence_length_source, hidden), (batch_size, hidden), (batch_size, hidden)

    # decoder
    with nn.parameter_scope('decoder'):
        dec_output = lstm(dec_input, hidden, initial_state=(c, h), return_sequences=True)
        # -> (batch_size, sentence_length_target, hidden)

        attention_output = global_attention(dec_output, enc_output, mask=mask, score='dot')
        # -> (batch_size, sentence_length_target, hidden)

    output = F.concatenate(dec_output, attention_output, axis=2)

    output = time_distributed(PF.affine)(output, vocab_size_target, name='output')
    # -> (batch_size, sentence_length_target, vocab_size_target)

    t = F.reshape(y, (batch_size, sentence_length_target, 1))

    entropy = time_distributed_softmax_cross_entropy(output, t)

    mask = F.sum(F.sign(t), axis=2) # do not predict 'pad'.
    count = F.sum(mask, axis=1)

    entropy *= mask
    loss = F.mean(F.sum(entropy, axis=1)/count)
    return x, y, loss


def predict(x):
    with nn.auto_forward():
        x = x.reshape((1, sentence_length_source))
        enc_input = nn.Variable.from_numpy_array(x)
        mask = get_mask(enc_input)
        enc_input = time_distributed(PF.embed)(enc_input, vocab_size_source, embedding_size, name='enc_embeddings') * mask

        # encoder
        with nn.parameter_scope('encoder'):
            enc_output, c, h = lstm(enc_input, hidden, mask=mask, return_sequences=True, return_state=True)
        
        # decode
        pad = nn.Variable.from_numpy_array(np.array([w2i_target['<bos>']]))
        x = PF.embed(pad, vocab_size_target, embedding_size, name='dec_embeddings')

        _cell, _hidden = c, h

        word_index = 0
        ret = []
        i = 0
        while i2w_target[word_index] != '。' and i < 20:
            with nn.parameter_scope('decoder'):
                with nn.parameter_scope('lstm'):
                    _cell, _hidden = lstm_cell(x, _cell, _hidden)
                    q = F.reshape(_hidden, (1, 1, hidden))
                    attention_output = global_attention(q, enc_output, mask=mask, score='dot')
            attention_output = F.reshape(attention_output, (1, hidden))
            output = F.concatenate(_hidden, attention_output, axis=1)
            output = PF.affine(output, vocab_size_target, name='output')

            word_index = np.argmax(output.d[0])
            ret.append(word_index)
            x = nn.Variable.from_numpy_array(np.array([word_index], dtype=np.int32))
            x = PF.embed(x, vocab_size_target, embedding_size, name='dec_embeddings')

            i+=1

        return ret

def translate_test(index):
    print('source:')
    print(' '.join([i2w_source[i] for i in test_source[index]][::-1]).strip(' pad'))
    print('target:')
    print(''.join([i2w_target[i] for i in test_target[index]]).strip('pad'))
    print('encoder-decoder output:')
    print(''.join([i2w_target[i] for i in predict(test_source[index])]).strip('pad'))

def translate(sentence):
    sentence = list(map(lambda x: w2i_source[x], sentence.split()))
    sentence += [0]*(sentence_length_source - len(sentence))
    sentence.reverse()
    return ''.join([i2w_target[i] for i in predict(np.array([sentence]))])

x, y, loss = build_model()

# Create solver.
solver = S.Momentum(1e-2, momentum=0.9)
solver.set_parameters(nn.get_parameters())

# Create monitor.
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
monitor = Monitor('./tmp-attention')
monitor_perplexity_train = MonitorSeries('perplexity_train', monitor, interval=1)
monitor_perplexity_dev = MonitorSeries('perplexity_dev', monitor, interval=1)

best_dev_loss = 9999

for epoch in range(max_epoch):
    train_loss_set = []
    progress = tqdm(total=train_data_iter.size//batch_size)
    for i in range(num_train_batch):
        x.d, y.d = train_data_iter.next()
        loss.forward()
        solver.zero_grad()
        loss.backward()
        solver.update()
        train_loss_set.append(loss.d.copy())

        progress.set_description(f"epoch: {epoch+1}, train perplexity: {np.e**np.mean(train_loss_set):.5f}")
        progress.update(1)
    progress.close()

    dev_loss_set = []
    for i in range(num_dev_batch):
        x.d, y.d = dev_data_iter.next()
        loss.forward()
        dev_loss_set.append(loss.d.copy())

    monitor_perplexity_train.add(epoch+1, np.e**np.mean(train_loss_set))
    monitor_perplexity_dev.add(epoch+1, np.e**np.mean(dev_loss_set))

    # dev_loss = np.e**np.mean(dev_loss_set)
    # if best_dev_loss > dev_loss:
    #     best_dev_loss = dev_loss
    #     print('best dev loss updated! {}'.format(dev_loss))
    #     nn.save_parameters('encdec_best.h5')



