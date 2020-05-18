from abc import ABC, abstractmethod

import numpy as np

from common.abstract.learner import Learner
from common.abstract.worker import Worker
from common.utils.buffer_helper import BufferHelper
from common.utils.param_server import ParameterServer


class Architecture(ABC):
    """Abstract Architecture used for all distributed architectures
    
    Attributes:
        self.num_workers = 
    
    """

    def __init__(self, cfg: dict):
        """Initialize"""
        self.cfg = cfg
        self.env = self.cfg["env"]
        self.num_workers = self.cfg["num_workers"]
        self.num_learners = self.cfg["num_learners"]
        self.num_step = self.cfg["num_step"]

    @abstractmethod
    def spawn(self, worker: type, learner: type):
        pass

    @abstractmethod
    def train(self):
        """Run main training loop"""
        pass
