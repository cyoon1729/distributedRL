from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import torch.nn as nn


class Learner(ABC):
    def __init__(self, brain: Union[nn.Module, tuple], cfg: dict):
        self.cfg = cfg
        self.brain = deepcopy(brain)
        if type(self.brain) is tuple:
            self.brain_value = self.brain[0] 
            self.brain_policy =self.brain[1]
        
    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: np.ndarray):
        pass

    def get_params(self):
        """Return model params for synchronization"""
        if type(self.brain) is tuple:
            model = self.brain_policy
        else:
            model = self.brain

        params = []
        state_dict = model.state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
        return numpy_params

    def get_worker_brain_sample(self)
        """for initializing workers"""
        if type(self.brain) is tuple:
            return self.brain_policy
        else:
            return self.brain