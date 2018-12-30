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

from typing import Optional
from typing import Tuple

from functions import time_distributed
from functions import where
from functions import get_attention_logit_mask

@PF.parametric_function_api('simple_rnn')
def simple_rnn(inputs: nn.Variable, units: int, mask: Optional[nn.Variable] = None,
               return_sequences: bool = False, fix_parameters=False) -> nn.Variable:
    '''
    A vanilla recurrent neural network layer
    Args:
        inputs (nnabla.Variable): A shape of [batch_size, length, embedding_size].
        units (int): Dimensionality of the output space.
        mask (nnabla.Variable): A shape of [batch_size, length, 1].
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [batch_size, length, units].
        or
        nn.Variable: A shape [batch_size units]
    '''

    hs = []
    batch_size, length, embedding_size = inputs.shape
    h0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)))

    h = h0

    if mask is None:
        mask = F.constant(1, shape=(batch_size, length, 1))

    for x, cond in zip(F.split(inputs, axis=1), F.split(mask, axis=1)):
        h_t = F.tanh(PF.affine(F.concatenate(x, h, axis=1), units, fix_parameters=fix_parameters))
        h = where(cond, h_t, h)
        hs.append(h)

    if return_sequences:
        hs = F.stack(*hs, axis=1)
        return hs
    else:
        return hs[-1]

def lstm_cell(x: nn.Variable, c: nn.Variable, h: nn.Variable) -> nn.Variable:
    batch_size, units = c.shape
    _hidden = PF.affine(F.concatenate(x, h, axis=1), 4*units)

    a            = F.tanh   (_hidden[:, units*0: units*1])
    input_gate   = F.sigmoid(_hidden[:, units*1: units*2])
    forgate_gate = F.sigmoid(_hidden[:, units*2: units*3])
    output_gate  = F.sigmoid(_hidden[:, units*3: units*4])

    cell = input_gate * a + forgate_gate * c
    hidden = output_gate * F.tanh(cell)
    return cell, hidden

@PF.parametric_function_api('lstm')
def lstm(inputs: nn.Variable, units: int, mask: Optional[nn.Variable] = None, initial_state: Tuple[nn.Variable, nn.Variable] = None,
         return_sequences: bool = False, return_state: bool = False, fix_parameters: bool = False) -> nn.Variable:
    '''
    A long short-term memory
    Args:
        inputs (nnabla.Variable): A shape of [batch_size, length, embedding_size].
        units (int): Dimensionality of the output space.
        mask (nnabla.Variable): A shape of [batch_size, length].
        initial_state ([nnabla.Variable, nnabla.Variable]): A tuple of an initial cell and an initial hidden state.
        return_sequences (bool): Whether to return the last output. in the output sequence, or the full sequence.
        return_state (bool): Whether to return the last state which is consist of the cell and the hidden state.
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [batch_size, length, units].
        or
        nn.Variable: A shape [batch_size units]
    '''
    
    batch_size, length, embedding_size = inputs.shape

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

    if mask is None:
        mask = F.constant(1, shape=(batch_size, length, 1))
    for x, cond in zip(F.split(inputs, axis=1), F.split(mask, axis=1)):
        cell_t, hidden_t = lstm_cell(x, cell, hidden)
        cell = where(cond, cell_t, cell)
        hidden = where(cond, hidden_t, hidden)
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
def highway(x: nn.Variable, fix_parameters: bool = False) -> nn.Variable:
    '''
    A densely connected highway network layer
    Args:
        x (nnabla.Variable): A shape of [batch_size, units]
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [batch_size, units].
    '''
    batch_size, in_out_size = x.shape

    with nn.parameter_scope('plain'):
        out_plain = F.relu(PF.affine(x, in_out_size, fix_parameters=fix_parameters))
    with nn.parameter_scope('transform'):
        out_transform = F.sigmoid(PF.affine(x, in_out_size, fix_parameters=fix_parameters))
    y = out_plain * out_transform + x * (1 - out_transform)
    return y

@PF.parametric_function_api('global_attention')
def global_attention(query: nn.Variable, memory: nn.Variable, mask: Optional[nn.Variable] = None,
                     score: str = 'general', fix_parameters: bool = False) -> nn.Variable:
    '''
    A global attention layer
    Args:
        query (nnabla.Variable): A shape of [batch_size, length_query, embedding_size]
        memory (nnabla.Variable): A shape of [batch_size, length_memory, embedding_size]
        mask (nnabla.Variable): A shape of [batch_size, length_query, length_memory]
        score (str): A kind of score functions for calculating attention weights.
                     'general', 'dot' or 'concat'.
                     see [Effective Approaches to Attention-based Neural Machine Translation]
                         (http://aclweb.org/anthology/D15-1166)
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [batch_size, length_query, embedding_size].
    '''
    batch_size, length_query, embedding_size =  query.shape
    _, length_memory, _ = memory.shape
    q = query
    # -> (batch_size, length_query, embedding_size)
    k = memory
    # -> (batch_size, length_memory, embedding_size)
    v = memory
    # -> (batch_size, length_memory, embedding_size)
    if score == 'dot':
        logit = F.batch_matmul(q, k, transpose_b=True)
        # -> (batch_size, length_query, length_memory)
    elif score == 'general':
        with nn.parameter_scope('Wa'):
            wa = time_distributed(PF.affine)(q, embedding_size, with_bias=False)
            # -> (batch_size, length_query, embeding_size)
        logit = F.batch_matmul(wa, k, transpose_b=True)
        # -> (batch_size, length_query, length_memory)
    elif score == 'concat':
        a_list = []
        for _q in F.split(q, axis=1):
            _q = F.reshape(_q, shape=(batch_size, 1, embedding_size))
            _q = F.broadcast(_q, shape=(batch_size, length_memory, embedding_size))
            concat = F.concatenate(_q, k, axis=2)
            # -> (batch_size, length_memory, embedding_size * 2)
            with nn.parameter_scope('Wa'):
                a = time_distributed(PF.affine)(concat, 1, with_bias=False)
                # -> (batch_size, length_memory, 1)
                a_list.append(a)
        
        logit = F.concatenate(*a_list, axis=2)
        # -> (batch_size, length_memory, length_query)
        logit = F.transpose(logit, axes=(0, 2, 1))
        # -> (batch_size, length_query, length_memory)
    
    # get_attention_logit_mask -> (batch_size, length_query, length_memory)である
    if mask is not None:
        logit += get_attention_logit_mask(mask)


    attention_weights = F.softmax(logit, axis=2)
    # -> (batch_size, length_query, length_memory)

    attention_output = F.batch_matmul(attention_weights, v)
    # -> (batch_size, length_query, embedding_size)

    return attention_output




