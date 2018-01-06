# MIT License
# 
# Copyright (c) 2017-2018 Minato Sato
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

def SimpleRNN(inputs, units, return_sequences=False):
    '''
    A vanilla recurrent neural network layer
    Args:
        inputs (nnabla.Variable): A shape of [B, SentenceLength, EmbeddingSize].
        units (int): Dimensionality of the output space.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
    Returns:
        nn.Variable: A shape [B, SentenceLength, units].
        or
        nn.Variable: A shape [B, units]
    '''

    hs = []
    batch_size = inputs.shape[0]
    sentence_length = inputs.shape[1]
    h0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))

    inputs = F.split(inputs, axis=1) # split in the direction of sequence

    h = h0
    for x in inputs:
        with nn.parameter_scope('x2h'):
            x2h = PF.affine(x, units, with_bias=False)
        with nn.parameter_scope('h2h'):
            h2h = PF.affine(h, units)
        h = F.tanh(x2h + h2h)
        if return_sequences:
            h = F.reshape(h, (batch_size, 1, units))
        hs.append(h)

    if return_sequences:
        hs = F.concatenate(*hs, axis=1)
        return hs
    else:
        return hs[-1]


def LSTM(inputs, units, return_sequences=False, name='lstm'):
    '''
    A long short-term memory layer
    Args:
        inputs (nnabla.Variable): A shape of [B, SentenceLength, EmbeddingSize].
        units (int): Dimensionality of the output space.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
    Returns:
        nn.Variable: A shape [B, SentenceLength, units].
        or
        nn.Variable: A shape [B, units]
    '''

    hs = []
    batch_size = inputs.shape[0]
    sentence_length = inputs.shape[1]
    c0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))
    h0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))

    inputs = F.split(inputs, axis=1)

    cell = c0
    hidden = h0

    with nn.parameter_scope(name):
        for x in inputs:
            a = F.tanh(PF.affine(x, units, with_bias=False, name='Wa') + PF.affine(hidden, units, name='Ra'))
            input_gate = F.sigmoid(PF.affine(x, units, with_bias=False, name='Wi') + PF.affine(hidden, units, name='Ri'))
            forgate_gate = F.sigmoid(PF.affine(x, units, with_bias=False, name='Wf') + PF.affine(hidden, units, name='Rf'))
            cell = input_gate * a + forgate_gate * cell
            output_gate = F.sigmoid(PF.affine(x, units, with_bias=False, name='Wo') + PF.affine(hidden, units, name='Ro'))
            hidden = output_gate * F.tanh(cell)
            if return_sequences:
                hidden = F.reshape(hidden, (batch_size, 1, units))
            hs.append(hidden)

    if return_sequences:
        hs = F.concatenate(*hs, axis=1)
        hs = F.reshape(hs, (batch_size, sentence_length, units))
        return hs
    else:
        return hs[-1]


def Highway(x, name='highway'):
    '''
    A densely connected highway network layer
    Args:
        x (nnabla.Variable): A shape of [B, units]
    Returns:
        nn.Variable: A shape [B, units].
    '''
    batch_size, in_out_size = x.shape

    with nn.parameter_scope(name):
        with nn.parameter_scope('plain'):
            out_plain = F.relu(PF.affine(x, in_out_size))
        with nn.parameter_scope('transform'):
            out_transform = F.sigmoid(PF.affine(x, in_out_size))
    y = out_plain * out_transform + x * (1 - out_transform)
    return y

def TimeDistributed(func):
    def TimeDistributedFunc(x, *args, **kwargs):
        ret = []
        batch_size = x.shape[0]
        units = x.shape[2]
        for x_ in F.split(x, axis=1):
            value = func(x_, *args, **kwargs)
            _, output_dim = value.shape
            ret.append(F.reshape(value, (batch_size, 1, output_dim)))
        return F.concatenate(*ret, axis=1)
    return TimeDistributedFunc

def TimeDistributedSoftmaxCrossEntropy(y_pred, y_true):
    '''
    A time distributed softmax crossentropy
    Args:
        y_pred (nnabla.Variable): A shape of [B, SentenceLength, O]. # one-hot
        y_true (nnabla.Variable): A shape of [B, SentenceLength, 1]. # index
    Returns:
        nn.Variable: A shape [B, SentenceLength].
    '''
    ret = []
    for y_p, y_t in zip(F.split(y_pred, axis=1), F.split(y_true, axis=1)):
        ret.append(F.softmax_cross_entropy(y_p, y_t))
    return F.concatenate(*ret)
