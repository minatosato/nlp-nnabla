# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os

from scipy import sparse
from collections import Counter
from itertools import combinations
from typing import List
from typing import Tuple


def to_cooccurrences(sentences: List[List[int]], vocab_size: int, window_size: int = 5) -> sparse.lil_matrix:
    matrix = sparse.lil_matrix((vocab_size, vocab_size))
    for sentence in sentences:
        for i, word_id in enumerate(sentence):
            contexts = sentence[max(0, i - window_size): i]
            for j, context_id in enumerate(contexts):
                #distance = len(contexts) - j
                matrix[word_id, context_id] += 1# / distance
                # matrix[context_id, word_id] += 1 / distance
    return matrix

def to_glove_dataset(sentences: List[List[int]], vocab_size: int, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = to_cooccurrences(sentences, vocab_size=vocab_size, window_size=window_size)
    central: List[int] = []
    context: List[int] = []
    y: List[int] = []

    for i, j in zip(*matrix.nonzero()):
        central.append(i)
        context.append(j)
        y.append(matrix[i, j])

    return np.array(central), np.array(context), np.array(y)[:, None]

