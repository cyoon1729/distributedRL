from abc import ABC, abstractmethod

import numpy as np

from common.abstract.learner import Learner
from common.abstract.worker import Worker
from common.utils.buffer_helper import BufferHelper
from common.utils.param_server import ParameterServer


class Architecture(ABC):
    """Abstract Architecture used for all distributed architectures"""

    def __init__(self, cfg: dict):
        """Initialize"""
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.num_learners = self.cfg["num_learners"]

    @abstractmethod
    def spawn(self, worker: type, learner: type, global_buffer: BufferHelper):
        """Spawn distributed components"""
        pass

    @abstractmethod
    def train(self):
        """Run main training loop"""
        pass
