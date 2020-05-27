from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union
from collections import deque

import asyncio
import numpy as np
import torch
import torch.nn as nn


class Learner(ABC):
    def __init__(self, brain: Union[nn.Module, tuple], cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["learner_device"])
        self.brain = deepcopy(brain)
        self.batch_queue = deque()

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: tuple):
        pass

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return model params for synchronization"""
        pass

    @abstractmethod
    def run(self):
        """Run main loop"""
        pass

    async def recv_batch(self, batch):
        self.batch_queue.append(batch)

    def params_to_numpy(self, model):
        params = []
        state_dict = model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
        return params


class ApeXLearner(Learner):
    def __init__(self, brain: Union[nn.Module, tuple], cfg: dict):
        super().__init__(brain, cfg)
        self.max_num_updates = cfg["max_num_updates"]
        self.synchronize_interval = cfg["synchronize_interval"]
        self.batch_queue = deque(maxlen=10)

    async def run(self, global_buffer_handle, workers):
        print("starting learner")
        update_step = 0
        while update_step < self.max_num_updates:
            if self.batch_queue:
                batch = self.batch_queue.pop()
                step_info, idxes, new_priorities = self.learning_step(batch)
                print(step_info)
                global_buffer_handle.update_priorities.remote(idxes, new_priorities)
                update_step = update_step + 1

            if update_step % self.synchronize_interval == 0:
                new_params = self.get_params()
                for worker_handle in workers:
                    worker_handle.recv_params.remote(new_params)