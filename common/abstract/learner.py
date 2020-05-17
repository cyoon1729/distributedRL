from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from copy import deepcopy


class Learner(ABC):
    def __init__(self, brain: nn.Module, cfg: dict):
        self.cfg = cfg
        self.brain = deepcopy(brain)

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: np.ndarray):
        pass

    def get_params(self, model):
        """Return model params for synchronization"""
        params = []
        state_dict = model.state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
        return params