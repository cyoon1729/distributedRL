from typing import Deque, Union

import numpy as np
import ray
import torch
import torch.nn as nn
import random
from datetime import datetime

from common.abstract.architecture import Architecture
from common.utils.buffer import PrioritizedReplayBuffer
from common.utils.buffer_helper import PrioritizedReplayBufferHelper
from common.utils.param_server import ParameterServer
from common.utils.utils import params_to_numpy

"""
TODO: kill all sockets at exit
"""

class ApeX(Architecture):
    def __init__(self, worker_cls: type, learner_cls: type, brain: Union[tuple, nn.Module], cfg: dict, comm_config: dict):
        self.cfg = cfg
        self.comm_config = comm_config
        super().__init__(self.cfg)
        
        self.brain = brain
        if type(brain) is tuple:
            worker_brain = self.brain[0]
        else:
            worker_brain = self.brain
        
        # Spawn all components
        self.workers = [
            worker_cls.remote(n, worker_brain, self.cfg, self.comm_config) for n in range(1, self.num_workers)
        ]
        self.learner = learner_cls.remote(self.brain, self.cfg, self.comm_config)
        self.global_buffer = PrioritizedReplayBufferHelper.remote(PrioritizedReplayBuffer, self.cfg, self.comm_config)
        self.all_actors = self.workers + [self.learner] + [self.global_buffer]

    def train(self):
        print("Running main training loop...")
        ray.wait([actor.run.remote() for actor in self.all_actors])