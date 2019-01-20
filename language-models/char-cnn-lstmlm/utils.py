# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from typing import List
from typing import Dict

def wordseq2charseq(data: np.ndarray, i2w: Dict[int, str], c2i: Dict[str, int], i2c: Dict[int, str], word_length: int = 20) -> np.ndarray:
    data = np.repeat(np.expand_dims(data, axis=2), word_length, axis=2)
    data[:, :, 1:] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            word = data[i][j][0]
            if word != 0:
                for k, char in enumerate(i2w[word]):
                    data[i][j][k] = c2i[char]
            else:
                data[i, j, :] = 0
    return data

