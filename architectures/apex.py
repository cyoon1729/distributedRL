from typing import Deque, Union

import numpy as np
import ray
import torch
import torch.nn as nn
import random
from datetime import datetime
import asyncio 

from common.abstract.architecture import Architecture
from common.utils.buffer import PrioritizedReplayBuffer
from common.utils.buffer_helper import BufferHelper
from common.utils.param_server import ParameterServer


@ray.remote
def run_learner(learner, global_buffer_handle, param_server_handle, batch_size, synchronize_interval, max_num_updates):
    print("starting learner")
    update_step = 0
    while update_step < max_num_updates:
        if learner.batch_queue:
        	batch = learner.batch_queue.pop()
        	step_info, idxes, new_priorities = learner.learning_step(batch)
        	global_buffer_handle.update_priorities.remote(idxes, new_priorities)
        	priority_beta = priority_beta + priority_beta_increment
        	update_step = update_step + 1

		if update_step % synchronize_interval == 0:
        	new_params = learner.get_params()
        	param_server_handle.recv_new_params(new_params)


@ray.remote
def run_worker(worker, global_buffer_handle, param_server_handle, worker_update_interval):
	# if worker is not ray.remote actor, then how to receive
	# param_server_handle without calling ray.get()?

	worker_buffer = worker.collect_data()
	global_buffer_handle.recv_new_data.remote(worker_buffer)
        




@ray.remote
async def run_workers(workers, global_buffer_handle, learner_handle, worker_update_interval):
	print("starting worker")
    worker_buffers = {}
    worker_update_steps = {}
    for worker in workers:
        worker_buffers[worker.collect_data.remote()] = worker
        worker_update_steps[worker] = 0

    while True:
        ready_worker_list, _ = ray.wait(list(worker_buffers)) # -> NOTE: not allowed
        ready_worker_id = ready_worker_list[0]
        worker = worker_buffers.pop(ready_worker_id)
        
        global_buffer_handle.incorporate_new_data.remote(
        	ray.get(worker.get_buffer.remote()) # -> NOTE: not allowed
        )

       	if worker_update_steps[worker] % worker_update_interval == 0:
        	new_params = ray.get(self.param_server.get_params.remote())
        	worker.synchronize.remote(new_params)
        	worker_update_steps[worker] = worker_update_steps[worker] + 1
        
        worker_buffers[worker.collect_data.remote()] = worker


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
        
        self.worker_seeds = list(np.random.choice(np.arange(1, 999, 1), self.num_workers))
        self.cfg["worker_seeds"] = self.worker_seeds
        self.workers = [
            worker.remote(worker_id, initial_worker_brain, int(seed), self.cfg)
            for seed, worker_id in zip(self.worker_seeds, range(self.num_workers))
        ]

        self.global_buffer = BufferHelper.remote(
            PrioritizedReplayBuffer(self.buffer_max_size, self.priority_alpha)
        )

        learner_params = ray.get(self.learner.get_params.remote())
        self.param_server = ParameterServer.remote()
        self.param_server.initialize.remote(learner_params)

        self.performance_worker = worker.remote(999, initial_worker_brain, random.randint(1, 999), self.cfg)

        print("Spawned all components!")

    def train(self):
        print("Starting...")
        print(torch.cuda.is_available())
        print(ray.get_gpu_ids())
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

        while update_step < self.max_num_updates:
            # 1. Incoporate worker data into global buffer
            start = datetime.now()

            ready_worker_list, _ = ray.wait(list(worker_buffers))   # <-- lock (fix)
            ready_worker_id = ready_worker_list[0]
            worker = worker_buffers.pop(ready_worker_id)
            self.global_buffer.incorporate_new_data.remote(
                ray.get(worker.get_buffer.remote())
            )

            

            # 2. Sample from PER buffer
            batch = ray.get(
                self.global_buffer.sample_data.remote(self.batch_size, self.priority_beta)
            )

            # 3. Run learner learning step
            update_step  = update_step + 1
            step_info, idxes, new_priorities = ray.get(
                self.learner.learning_step.remote(batch)
            )

            # print(step_info)

            # 4. Update PER buffer priorities
            self.global_buffer.update_priorities.remote(idxes, new_priorities)
            self.priority_beta += self.priority_beta_increment

            print(f"one learning step: {datetime.now() - start}") 
            
            # 5. Sync worker brain with new brain
            new_params = ray.get(self.learner.get_params.remote())
            worker.synchronize.remote(new_params)

            worker_buffers[worker.collect_data.remote()] = worker

    @ray.remote
    def run_workers(self):
        print("starting worker")
        self.worker_buffers = {}
        for worker in self.workers:
            self.worker_buffers[worker.collect_data.remote()] = worker

        # 1. Incoporate worker data into global buffer
        while True:
            ready_worker_list, _ = ray.wait(list(self.worker_buffers))   # <-- lock (fix)
            ready_worker_id = ready_worker_list[0]
            worker = self.worker_buffers.pop(ready_worker_id)
            self.global_buffer.incorporate_new_data.remote(
                ray.get(worker.get_buffer.remote())
            )
            new_params = self.param_server.get_params.remote()
            worker.synchronize.remote(new_params)

            self.worker_buffers[worker.collect_data.remote()] = worker

    @ray.remote # (num_gpus=1)
    def run_learner(self):
        print("starting learner")
        while True:
            # 2. Sample from PER buffer
            
            batch = ray.get(
                self.global_buffer.sample_data.remote(self.batch_size, self.priority_beta)
            )

            # 3. Run learner learning step
            # self.update_step = self.update_step + 1
            step_info, idxes, new_priorities = ray.get(
                self.learner.learning_step.remote(batch)
            )

            # 4. Update PER buffer priorities
            self.global_buffer.update_priorities.remote(idxes, new_priorities)
            self.priority_beta += self.priority_beta_increment

            # 5. Sync worker brain with new brain
            new_params = ray.get(self.learner.get_params.remote())
            self.param_server.update_params.remote(new_params)    


    @ray.remote
    def run_interim_test(self):
        print("Start process: run_interim_test")
        while True:
            update_step = ray.get(self.param_server.get_update_step.remote())
            if update_step % 10 == 0:
                new_params = ray.get(self.param_server.get_params.remote())
                self.performance_worker.synchronize.remote(new_params)
                episode_reward = ray.get(self.performance_worker.test_run.remote())


    def train2(self):
        print("Starting...")

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
        
        # ray.wait([self.run_learner.remote(self)])
        print("Running main training loop...")


        ray.wait([self.run_learner.remote(self), self.run_workers.remote(self), self.run_interim_test.remote(self)])
        # ray.get([ 
        #     run_workers.remote(self.worker_buffers, self.param_server, self.global_buffer),
        #     run_learner.remote(self.learner, self.global_buffer, self.param_server, self.priority_beta, self.priority_beta_increment)
        #     ]
        # )



            # run_workers.remote(self.worker_buffers, self.param_server, self.global_buffer),
            # run_interim_test.remote(self.performance_worker, self.param_server)