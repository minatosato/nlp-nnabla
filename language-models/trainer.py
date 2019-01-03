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
import nnabla.monitor as M

from nnabla.utils.data_iterator import DataIterator

import numpy as np

from tqdm import tqdm

from typing import List
from typing import Dict
from typing import Optional
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

@dataclass
class Trainer:
    inputs: List[nn.Variable]
    loss: nn.Variable
    solver: S.Solver
    metrics: Dict[str, nn.Variable] = field(default_factory=dict)
    save_path: str = 'log'
    current_epoch: int = 0

    def __post_init__(self):
        if len(self.metrics) == 0:
            self.metrics['loss'] = self.loss
        
        self.monitor: M.Monitor = M.Monitor(self.save_path)
        self.monitor_series: Dict[str, M.MonitorSeries] = dict()
    
    def update_variables(self, inputs: List[nn.Variable], loss: nn.Variable, metrics: Dict[str, nn.Variable] = {}):
        self.inputs: List[nn.Variable] = inputs
        self.loss: nn.Variable = loss
        self.metrics: Dict[str, nn.Variable] = metrics

        self.__post_init__()
    
    def run(self, train_iter: DataIterator, valid_iter: Optional[DataIterator] = None, epochs: int = 5, verbose=0) -> None:
        assert len(train_iter.variables) == len(self.inputs), \
              'the number of varibales received from iterator must be equal to the number of input variables'
        if valid_iter is not None:
            assert len(valid_iter.variables) == len(self.inputs), \
                  'the number of varibales received from iterator must be equal to the number of input variables'

        batch_size = self.inputs[0].shape[0]
        assert train_iter.batch_size == batch_size
        if valid_iter is not None:
            assert valid_iter.batch_size == batch_size

        num_train_batch = train_iter.size // batch_size
        for epoch in range(self.current_epoch, epochs+self.current_epoch):
            epoch_result = self._run_one_epoch(num_train_batch, epoch, train_iter, train=True)
            self.save_result(epoch_result)
            if valid_iter is not None:
                num_valid_batch = valid_iter.size // batch_size
                epoch_result = self._run_one_epoch(num_valid_batch, epoch, valid_iter, train=False, show_epoch=False)
                self.save_result(epoch_result)
            self.current_epoch += 1
        
    def save_result(self, result: Dict[str, float]):
        for key in result:
            if key not in self.monitor_series:
                self.monitor_series[key] = M.MonitorSeries(key, self.monitor, interval=1)
            self.monitor_series[key].add(self.current_epoch, result[key])
    
    def evaluate(self, valid_iter: DataIterator, verbose=0) -> None:
        assert len(valid_iter.variables) == len(self.inputs)

        batch_size = self.inputs[0].shape[0]
        assert valid_iter.batch_size == batch_size

        num_valid_batch = valid_iter.size // batch_size
        self._run_one_epoch(num_valid_batch, self.current_epoch-1, valid_iter, train=False, show_epoch=False)

    def _init_metrics_logger(self) -> Dict[str, List[float]]:
        logger: Dict[str, List[float]] = dict()
        for key in self.metrics:
            logger[key] = []
        return logger

    def _run_one_epoch(self, num_batch:int, epoch: int, iterator: DataIterator,
                       train: bool, show_epoch=True) -> Dict[str, float]:
        metrics_logger: Dict[str, List[float]] = self._init_metrics_logger()
        
        with tqdm(total=num_batch) as progress:
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
                if show_epoch:
                    description_list.append(f"epoch: {epoch+1}")
                for key in self.metrics:
                    description_list.append(f"{'train' if train else 'valid'} {key}: {np.mean(metrics_logger[key]):.5f}")
                progress.set_description(', '.join(description_list))
                progress.update(1)
        
        epoch_result: Dict[str, float] = dict()
        for metric in metrics_logger:
            epoch_result[metric + '-' + ('train' if train else 'valid')] = np.mean(metrics_logger[metric])

        return epoch_result
