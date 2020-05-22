from typing import Deque, Union

import numpy as np
import ray
import torch
import torch.nn as nn
import random
from datetime import datetime

from common.abstract.architecture import Architecture
from common.utils.buffer import PrioritizedReplayBuffer
from common.utils.buffer_helper import BufferHelper
from common.utils.param_server import ParameterServer
from common.utils.utils import params_to_numpy


class ApeX(Architecture):
    def __init__(self, worker_cls: type, learner_cls: type, brain: Union[tuple, nn.Module], cfg):
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

        self.brain = brain
        self.worker_cls = worker_cls
        self.learner_cls = learner_cls

        self.update_step = 0

    @ray.remote
    def run_workers(self): 
        print(f"worker process cuda available: {torch.cuda.is_available()}")
        self.worker_seeds = list(np.random.choice(np.arange(1, 999, 1), self.num_workers))
        self.cfg["worker_seeds"] = self.worker_seeds
        self.workers = [
            self.worker_cls.remote(worker_id, self.brain[0].cpu(), int(seed), self.cfg)
            for seed, worker_id in zip(self.worker_seeds, range(self.num_workers))
        ]

        worker_buffers = {}
        for worker in self.workers:
            worker_buffers[worker.collect_data.remote()] = worker

        print("start workers")
        while True:
            ready_worker_list, _ = ray.wait(list(worker_buffers)) 
            ready_worker_id = ready_worker_list[0]
            worker = worker_buffers.pop(ready_worker_id)
            self.global_buffer.incorporate_new_data.remote(
                ray.get(worker.get_buffer.remote())
            )
            
            new_params = self.param_server.get_params.remote()
            worker.synchronize.remote(new_params)

            worker_buffers[worker.collect_data.remote()] = workerearner_
        print(f"Learner process cuda available: {torch.cuda.is_available()}")
        print(f"Learner process gpu ids: {ray.get_gpu_ids()}") 

        self.learner = self.learner_cls.remote(self.brain, self.cfg)
        new_params = ray.get(self.learner.get_params.remote())
        self.param_server.initialize.remote(new_params)           

        
        while ray.get(self.global_buffer.get_size.remote()) < self.batch_size:
            pass
        
        print("Start learner:")
        while True:
            batch = ray.get(
                self.global_buffer.sample_data.remote(self.batch_size, self.priority_beta)
            )

            step_info, idxes, new_priorities = ray.get(
                self.learner.learning_step.remote(batch)
            )

            self.global_buffer.update_priorities.remote(idxes, new_priorities)
            self.priority_beta += self.priority_beta_increment

            new_params = ray.get(self.learner.get_params.remote())
            self.param_server.update_params.remote(new_params)   

    @ray.remote
    def run_interim_test(self):
        print("Start performance worker:")
        self.performance_worker = self.worker_cls.remote(999, self.brain[0], random.randint(1, 999), self.cfg)
        
        while not ray.get(self.param_server.get_params.remote()):
            pass

        while True:
            update_step = ray.get(self.param_server.get_update_step.remote())
            if update_step > self.update_step and update_step % 100 == 0:
                new_params = ray.get(self.param_server.get_params.remote())
                self.performance_worker.synchronize.remote(new_params)
                episode_reward = ray.get(self.performance_worker.test_run.remote())
                print(f"update step {update_step}: {episode_reward}")
                self.update_step = update_step

    def train(self):
    
        print("Spawning global buffer")
        self.global_buffer = BufferHelper.remote(
            PrioritizedReplayBuffer(self.buffer_max_size, self.priority_alpha)
        )
        print("Spawning parameter suffer")
        self.param_server = ParameterServer.remote()
        # self.run_learner.remote(self)
        print("Running main training loop...")      
        ray.wait([self.run_workers.remote(self), self.run_learner.remote(self), self.run_interim_test.remote(self)]) #, ,   , ,  self.run_interim_test.remote(self)