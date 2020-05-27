from typing import Deque

import ray
import asyncio
from common.utils.buffer import PrioritizedReplayBuffer 


@ray.remote
class PrioritizedReplayBufferHelper(object):
    def __init__(self, cfg):
        self.buffer = PrioritizedReplayBuffer(cfg["max_global_buffer_size"], cfg["priority_alpha"])
        self.batch_size = cfg["batch_size"]
        self.max_num_updates = cfg["max_num_updates"]
        self.priority_beta = cfg["priority_beta_start"]
        self.priority_beta_end = cfg["priority_beta_end"]
        self.priority_beta_increment = (
            self.priority_beta_end - self.priority_beta
        ) / self.max_num_updates
        
    async def recv_new_data(self, new_data: Deque):
        "Method is called from worker main loop"
        for data in new_data:
            replay_data, priority_value = data
            self.buffer.add(*replay_data)
            self.buffer.update_priorities([len(self.buffer) - 1], priority_value)
    
    async def update_priorities(self, idxes, new_priorities):
        "Method is called from learner main loop"
        self.buffer.update_priorities(idxes, new_priorities)
    
    async def send_replay_data(self, learner_handle):
    	while True:
	    	if len(self.buffer) > self.batch_size:
	    		batch = self.buffer.sample(self.batch_size, self.priority_beta)
	    		self.priority_beta = self.priority_beta + self.priority_beta_increment
	    		learner_handle.recv_batch(batch)