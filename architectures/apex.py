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







# # TODO: remove all remote calls
# @ray.remote
# def run_interim_test(self):
#     print("Start process: run_interim_test")
#     while True:
#         update_step = ray.get(self.param_server.get_update_step.remote())
#         if update_step % 10 == 0:
#             new_params = ray.get(self.param_server.get_params.remote())
#             self.performance_worker.synchronize.remote(new_params)
#             episode_reward = ray.get(self.performance_worker.test_run.remote())



# @ray.remote
# def run_learner(learner, global_buffer_handle, workers, batch_size, synchronize_interval, max_num_updates):
#     print("starting learner")
#     update_step = 0
#     while update_step < max_num_updates:
#         if learner.batch_queue:
#         	batch = learner.batch_queue.pop()
#         	step_info, idxes, new_priorities = learner.learning_step(batch)
#         	global_buffer_handle.update_priorities.remote(idxes, new_priorities)
#         	priority_beta = priority_beta + priority_beta_increment
#         	update_step = update_step + 1

# 		if update_step % synchronize_interval == 0:
#         	new_params = learner.get_params()
#         	for worker_handle in workers:
#         		worker_handle.recv_params(new_params)

