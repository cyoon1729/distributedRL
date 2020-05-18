from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import torch.nn as nn
import torch.optim as optim

class Learner(ABC):
    def __init__(self, brain: Union[nn.Module, tuple], cfg: dict):
        self.cfg = cfg
        self.device = torch.device(config['learner_device'])
        self.brain = deepcopy(brain)
        
    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: np.ndarray, weights: np.ndarray):
        pass

    @abstractmethod
    def get_params(self):
        """Return model params for synchronization"""
        pass 

    @abstractmethod 
    def get_worker_brain_sample(self)
        """for initializing workers"""
        pass

    @property
    def params_to_numpy(self, model):
        params = []
        state_dict = model.state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
        return numpy_params
