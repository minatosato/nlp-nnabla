# 
# Copyright (c) 2017-2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


@PF.parametric_function_api('simple_rnn')
def simple_rnn(inputs, units, return_sequences=False, fix_parameters=False):
    '''
    A vanilla recurrent neural network layer
    Args:
        inputs (nnabla.Variable): A shape of [B, SentenceLength, EmbeddingSize].
        units (int): Dimensionality of the output space.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
        fix_parameters (bool): Fix parameters (Set need_grad=False).
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
        h = F.tanh(PF.affine(F.concatenate(x, h, axis=1), units, fix_parameters=fix_parameters))
        hs.append(h)

    if return_sequences:
        hs = F.stack(*hs, axis=1)
        return hs
    else:
        return hs[-1]

def lstm_cell(x, c, h):
    batch_size, units = c.shape
    _hidden = PF.affine(F.concatenate(x, h, axis=1), 4*units)

    a            = F.tanh   (F.slice(_hidden, start=(0, units*0), stop=(batch_size, units*1)))
    input_gate   = F.sigmoid(F.slice(_hidden, start=(0, units*1), stop=(batch_size, units*2)))
    forgate_gate = F.sigmoid(F.slice(_hidden, start=(0, units*2), stop=(batch_size, units*3)))
    output_gate  = F.sigmoid(F.slice(_hidden, start=(0, units*3), stop=(batch_size, units*4)))

    cell = input_gate * a + forgate_gate * c
    hidden = output_gate * F.tanh(cell)
    return cell, hidden

@PF.parametric_function_api('lstm')
def lstm(inputs, units, initial_state=None, return_sequences=False, return_state=False, fix_parameters=False):
    '''
    A long short-term memory
    Args:
        inputs (nnabla.Variable): A shape of [B, SentenceLength, EmbeddingSize].
        units (int): Dimensionality of the output space.
        initial_state ([nnabla.Variable, nnabla.Variable]): A tuple of an initial cell and an initial hidden state.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
        return_state (bool): Whether to return the last state which is consist of the cell and the hidden state.
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [B, SentenceLength, units].
        or
        nn.Variable: A shape [B, units]
    '''
    
    batch_size = inputs.shape[0]

    if initial_state is None:
        c0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))
        h0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))
    else:
        assert type(initial_state) is tuple or type(initial_state) is list, \
               'initial_state must be a typle or a list.'
        assert len(initial_state) == 2, \
               'initial_state must have only two states.'

        c0, h0 = initial_state

        assert c0.shape == h0.shape, 'shapes of initial_state must be same.'
        assert c0.shape[0] == batch_size, \
               'batch size of initial_state ({0}) is different from that of inputs ({1}).'.format(c0.shape[0], batch_size)
        assert c0.shape[1] == units, \
               'units size of initial_state ({0}) is different from that of units of args ({1}).'.format(c0.shape[1], units)

    cell = c0
    hidden = h0

    hs = []

    for x in F.split(inputs, axis=1):
        cell, hidden = lstm_cell(x, cell, hidden)
        hs.append(hidden)

    if return_sequences:
        ret = F.stack(*hs, axis=1)
    else:
        ret = hs[-1]

    if return_state:
        return ret, cell, hidden
    else:
        return ret


@PF.parametric_function_api('highway')
def highway(x, fix_parameters=False):
    '''
    A densely connected highway network layer
    Args:
        x (nnabla.Variable): A shape of [B, units]
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [B, units].
    '''
    batch_size, in_out_size = x.shape


    with nn.parameter_scope('plain'):
        out_plain = F.relu(PF.affine(x, in_out_size, fix_parameters=fix_parameters))
    with nn.parameter_scope('transform'):
        out_transform = F.sigmoid(PF.affine(x, in_out_size, fix_parameters=fix_parameters))
    y = out_plain * out_transform + x * (1 - out_transform)
    return y


