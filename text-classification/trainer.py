# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from nnabla.utils.data_iterator import DataIterator

import numpy as np

from tqdm import tqdm

from typing import List
from typing import Dict
from typing import Optional

class Trainer(object):
    def __init__(self, inputs: List[nn.Variable], loss: nn.Variable, metrics: Dict[str, nn.Variable], solver: S.Solver) -> None:
        self.inputs: List[nn.Variable] = inputs
        self.loss: nn.Variable = loss
        self.metrics: Dict[str, nn.Variable] = metrics
        self.solver: S.Solver = solver
        self.current_epoch: int = 0
    
    def update_variables(self, inputs: List[nn.Variable], metrics: Dict[str, nn.Variable]):
        self.inputs: List[nn.Variable] = inputs
        self.metrics: Dict[str, nn.Variable] = metrics
    
    def run(self, train_iter: DataIterator, valid_iter: Optional[DataIterator], epochs: int, verbose=0) -> None:
        assert len(train_iter.variables) == len(self.inputs)
        if valid_iter is not None:
            assert len(valid_iter.variables) == len(self.inputs)

        batch_size = self.inputs[0].shape[0]
        assert train_iter.batch_size == batch_size
        if valid_iter is not None:
            assert valid_iter.batch_size == batch_size

        num_train_batch = train_iter.size // batch_size
        for epoch in range(self.current_epoch, epochs+self.current_epoch):
            self._run_one_epoch(num_train_batch, epoch, train_iter, train=True)
            if valid_iter is not None:
                num_valid_batch = valid_iter.size // batch_size
                self._run_one_epoch(num_valid_batch, epoch, valid_iter, train=False)
            self.current_epoch += 1
    
    def evaluate(self, valid_iter: DataIterator, verbose=0) -> None:
        assert len(valid_iter.variables) == len(self.inputs)

        batch_size = self.inputs[0].shape[0]
        assert valid_iter.batch_size == batch_size

        num_valid_batch = valid_iter.size // batch_size
        self._run_one_epoch(num_valid_batch, self.current_epoch-1, valid_iter, train=False)

    def _init_metrics_logger(self) -> Dict[str, List[float]]:
        logger: Dict[str, List[float]] = dict()
        for key in self.metrics:
            logger[key] = []
        return logger

    def _run_one_epoch(self, num_batch:int, epoch: int, iterator: DataIterator, train: bool):
        metrics_logger = self._init_metrics_logger()
        progress = tqdm(total=num_batch)

        for i in range(num_batch):
            # set data to variable
            for variable, data in zip(self.inputs, iterator.next()):
                variable.d = data
            
            for metric in list(self.metrics.values()):
                metric.forward()
            
            if train:
                loss_forward = True
                for metric in list(self.metrics.values()):
                    if metric is self.loss:
                        loss_forward = False
                if loss_forward:
                    self.loss.forward()
                self.solver.zero_grad()
                self.loss.backward()
                self.solver.update()
            
            for key in self.metrics:
                metrics_logger[key].append(self.metrics[key].d.copy())
            
            description_list = []
            description_list.append(f"epoch: {epoch+1}")
            for key in self.metrics:
                description_list.append(f"{'train' if train else 'valid'} {key}: {np.mean(metrics_logger[key]):.5f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

        progress.close()