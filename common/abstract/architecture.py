from abc import ABC, abstractmethod

import numpy as np

from distributedRL.common.abstract.learner import Learner
from distributedRL.common.abstract.worker import Worker
from distributedRL.common.utils.buffer_helper import BufferHelper
from distributedRL.common.utils.param_server import ParameterServer


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

        self.initial_brain = None
        self.initial_worker_brain = None

    @abstractmethod
    def initialize_central_brain(self):
        """Initialize brain"""
        pass

    def spawn(
        self,
        Worker: type,
        Learner: type,
        param_server: ParameterServer,
        centralized_buffer: BufferHelper,
    ):
        """Spawn Components of Distributed Architecture"""
        self.initialize_central_brain()

        self.learner = Learner(self.initial_brain, self.cfg)
        self.worker_seeds = np.random.choice(np.arange(1, 999, 1), self.num_workers)
        self.cfg["worker_seeds"] = self.worker_seeds.tolist()
        self.workers = [
            Worker.remote(
                worker_id, self.initial_worker_brain, self.env, seed, self.cfg
            )
            for seed, worker_id in zip(self.worker_seeds, len(self.worker_seeds))
        ]
        self.param_server = param_server
        self.centralized_buffer = centralized_buffer

    @abstractmethod
    def train(self):
        """Run main training loop"""
        pass