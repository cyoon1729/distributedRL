from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from copy import deepcopy

class Learner(ABC):

    def __init__(self, brain: nn.Module, cfg: dict):
        self.cfg = cfg
        self.brain = deepcopy(brain)
        pass
    
    @abstractmethod
    def learning_step(self, data: np.ndarray):
        pass


    @abstractmethod
    def prepare_brain_to_send(self):
        """Retrieve brain for synchronization"""
        pass

    def return_params(self):
        """Return params for synchronization"""
        params = []
        brain_to_send = self.prepare_brain_to_send()
        state_dict = brain_to_send.state_dict()
        for param in list(state_dict):
            params.append(state_dict[param]) 
        return params
    
    @abstractmethod
    def write_log(self):
        pass