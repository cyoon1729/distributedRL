from abc import ABC, abstractmethod

import numpy as np 
from collections import deque
import torch.nn as nn
import gym


class Worker(ABC):

    def __init__(self, worker_id: int, worker_brain: nn.Module, env: gym.Env, seed: int, cfg: dict)
        self.cfg
        self.brain = copy.deepcopy(worker_brain)
        self.env = env
        self.env = env.seed(seed)
        self.buffer = deque()

    @abstractmethod
    def select_action(self, state: np.ndarray)
        pass 

    def step(self, state:np.ndarray, action: np.ndarray):
        """Run one gym step"""
        action = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, next_state, done 

    @abstractmethod
    def work(self):
        """Fill worker buffer until some stopping criterion is satisfied"""
        pass
    
    def return_buffer(self):
        """Return buffer"""
        return self.buffer

    def synchronize(self, new_params: np.ndarray):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
    

"""save for later:"""
# def fill_buffer(self):
#     transitions_added = 0
#     state = self.env.reset()

#     while transitions_added < self.max_worker_buffer_size:

#         next_state, reward, done, = self.step()
        
#         if self.num_step == 1:
#             transition = (state, action, reward, next_state, done)
#             self.buffer.append(transition)

#         if self.num_steps > 1:
#             self.nstep_queue.append(state, action, reward, next_state, done)
#             if len(self.nstep_queue) == self.num_steps:
#                 transitions = organize_nstep(self.nstep_queue)
#                 self.buffer.append(transitions)
    

