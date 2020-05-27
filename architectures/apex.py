from typing import Deque, Union

import numpy as np
import ray
import torch
import torch.nn as nn
import random
from datetime import datetime
from copy import deepcopy

from common.abstract.architecture import Architecture
from common.utils.buffer_helper import PrioritizedReplayBufferHelper


class ApeX(Architecture):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ApeX, self).__init__(self.cfg)

    def spawn(self, worker: type, learner: type, brain: Union[tuple, nn.Module]):
        """Spawn Components of Distributed Architecture"""
        self.learner = learner.remote(brain, self.cfg)
        if type(brain) is tuple:
            worker_brain = deepcopy(brain[0])
        else:
        	worker_brain = deepcopy(brain)
        self.worker_seeds = list(np.random.choice(np.arange(1, 999, 1), self.num_workers))
        self.workers = [
            worker.remote(worker_id, worker_brain, int(seed), self.cfg)
            for seed, worker_id in zip(self.worker_seeds, range(self.num_workers))
        ]
        self.global_buffer = PrioritizedReplayBufferHelper.remote(self.cfg)

        print("Spawned all components!")

    def train(self):
    	procs = [
    		worker.run.remote(self.global_buffer) for worker in self.workers
    	]
    	procs.append(self.global_buffer.send_replay_data.remote(self.learner))
    	procs.append(self.learner.run.remote(self.global_buffer, self.workers))

    	ray.wait(procs)