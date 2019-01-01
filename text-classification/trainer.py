# 
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

import numpy as np

from tqdm import tqdm

from typing import List
from typing import Dict

class Trainer(object):
    def __init__(self, inputs: List[nn.Variable], loss: nn.Variable, metrics: Dict[str, nn.Variable], solver: S.Solver) -> None:
        self.inputs: List[nn.Variable] = inputs
        self.loss: nn.Variable = loss
        self.metrics: Dict[str, nn.Variable] = metrics
        self.solver: S.Solver = solver
    
    def run(self, train_iter, valid_iter, epochs, verbose=0):
        assert len(train_iter.variables) == len(self.inputs)
        assert len(valid_iter.variables) == len(self.inputs)

        batch_size = self.inputs[0].shape[0]
        assert train_iter.batch_size == valid_iter.batch_size == batch_size

        num_train_batch = train_iter.size // batch_size
        for epoch in range(epochs):
            metrics_logger = self._init_metrics_logger()

            progress = tqdm(total=num_train_batch)

            for i in range(num_train_batch):
                # set data to variable
                for variable, data in zip(self.inputs, train_iter.next()):
                    variable.d = data
                
                self.loss.forward()
                for metric in list(self.metrics.values()):
                    metric.forward()
                self.solver.zero_grad()
                self.loss.backward()
                self.solver.update()
                
                for key in self.metrics:
                    metrics_logger[key].append(self.metrics[key].d.copy())
                
                description_list = []
                description_list.append(f"epoch: {epoch+1}")
                for key in self.metrics:
                    description_list.append(f"train {key}: {np.mean(metrics_logger[key]):.5f}")
                progress.set_description(', '.join(description_list))
                progress.update(1)

            progress.close()


            
    
    def _init_metrics_logger(self) -> Dict[str, List[float]]:
        logger: Dict[str, List[float]] = dict()
        for key in self.metrics:
            logger[key] = []
        return logger