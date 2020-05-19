from typing import Deque, Union

import numpy as np
import ray
import torch.nn as nn

from common.abstract.architecture import Architecture
from common.utils.buffer import PrioritizedReplayBuffer
from common.utils.buffer_helper import BufferHelper


class ApeX(Architecture):
    def __init__(self, cfg):
        self.cfg = cfg
        super(ApeX, self).__init__(self.cfg)

        self.max_num_updates = cfg["max_num_updates"]
        self.buffer_max_size = self.cfg["buffer_max_size"]
        self.priority_alpha = self.cfg["priority_alpha"]
        self.priority_beta = cfg["priority_beta_start"]
        self.priority_beta_end = cfg["priority_beta_end"]
        self.priority_beta_increment = (
            self.priority_beta_end - self.priority_beta
        ) / self.max_num_updates

        self.batch_size = self.cfg["batch_size"]

    def spawn(self, worker: type, learner: type, brain: Union[tuple, nn.Module]):
        """Spawn Components of Distributed Architecture"""
        self.learner = learner.remote(brain, self.cfg)
        initial_worker_brain = ray.get(self.learner.get_worker_brain_sample.remote())
        self.worker_seeds = np.random.choice(np.arange(1, 999, 1), self.num_workers)
        self.cfg["worker_seeds"] = self.worker_seeds.tolist()
        self.workers = [
            worker.remote(worker_id, initial_worker_brain, seed, self.cfg)
            for seed, worker_id in zip(self.worker_seeds, range(len(self.worker_seeds)))
        ]

        self.global_buffer = BufferHelper.remote(
            PrioritizedReplayBuffer(self.buffer_max_size, self.priority_alpha)
        )

    def train(self):
        print("Pre-train check passed...")

        # pre-fill global buffer
        _, _ = ray.wait(
            [worker.collect_data.remote() for worker in self.workers],
            num_returns=len(self.workers),
        )

        # incorporate pre-fill data to global buffer
        for worker in self.workers:
            self.global_buffer.incorporate_new_data.remote(
                ray.get(worker.get_buffer.remote())
            )

        print("Initial buffer filled...")
        print("Running main training loop...")

        worker_buffers = {}
        for worker in self.workers:
            worker_buffers[worker.collect_data.remote()] = worker

        update_step = 0

        while update_step < self.num_train_steps:
            # 1. Incoporate worker data into global buffer
            ready_worker_list, _ = ray.wait(list(self.worker_buffers))
            ready_worker_id = ready_worker_list[0]
            worker = worker_buffers.pop(ready_worker_id)
            self.global_buffer.incorporate_new_data.remote(
                ray.get(worker.get_buffer.remote())
            )

            # 2. Sample from PER buffer
            batch, indices, weights = ray.get(
                self.global_buffer.sample.remote(self.batch_size, self.priority_beta)
            )

            # 3. Run learner learning step
            step_info, idxes, new_priorities = ray.get(
                self.learner.learning_step.remote(batch, indices, weights)
            )

            # 4. Update PER buffer priorities
            self.global_buffer.update_priorities.remote(idxes, new_priorities)
            self.priority_beta += self.priority_beta_increment

            # 5. Sync worker brain with new brain
            new_params = ray.get(self.learner.get_params())

            worker.set_params.remote(new_params)
