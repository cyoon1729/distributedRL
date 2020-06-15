from typing import Deque, Union

import numpy as np
import ray
import torch.nn as nn

from common.abstract.architecture import Architecture
from common.utils.buffer_helper import PrioritizedReplayBufferHelper


class ApeX(Architecture):
    def __init__(
        self,
        worker_cls: type,
        learner_cls: type,
        brain: Union[tuple, nn.Module],
        cfg: dict,
        comm_cfg: dict,
    ):
        self.cfg = cfg
        self.comm_cfg = comm_cfg
        super().__init__(worker_cls, learner_cls, self.cfg)

        self.brain = brain
        if type(brain) is tuple:
            self.worker_brain = self.brain[0]
        else:
            self.worker_brain = self.brain

    def spawn(self):
        # Spawn all components
        self.workers = [
            self.worker_cls.remote(n, self.worker_brain, self.cfg, self.comm_cfg)
            for n in range(1, self.num_workers + 1)
        ]
        # self.performance_worker = self.worker_cls.remote(
        #     "Test", self.worker_brain, self.cfg, self.comm_cfg
        # )
        self.learner = self.learner_cls.remote(self.brain, self.cfg, self.comm_cfg)
        self.global_buffer = PrioritizedReplayBufferHelper.remote(
            self.cfg, self.comm_cfg
        )
        self.all_actors = self.workers + [self.learner] + [self.global_buffer]

    def train(self):
        print("Running main training loop...")
        ray.wait(
            [actor.run.remote() for actor in self.all_actors]
        )

                    # + [self.performance_worker.test_run.remote()]
