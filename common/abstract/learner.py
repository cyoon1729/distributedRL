from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class Learner(ABC):
    def __init__(self, brain: Union[nn.Module, tuple], cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["learner_device"])
        self.brain = deepcopy(brain)
        self.gamma = self.cfg["gamma"]

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: tuple):
        pass

    @abstractmethod
    def get_params(self):
        """Return model params for synchronization"""
        pass

    @abstractmethod
    def get_worker_brain_sample(self):
        """for initializing workers"""
        pass

    def params_to_numpy(self, model):
        params = []
        state_dict = model.state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
        return params
