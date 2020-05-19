from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple, Union
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import gym
from common.utils.baseline_wrappers import make_atari, wrap_deepmind, wrap_pytorch


class Worker(ABC):
    def __init__(
        self,
        worker_id: int,
        worker_brain: nn.Module,
        env_name: str,
        seed: int,
        cfg: dict,
    ):
        self.cfg = cfg
        self.device = torch.device(self.cfg['worker_device'])
        self.brain = deepcopy(worker_brain)
        if self.cfg['atari'] is True:
            self.env = make_atari(env_name)
            self.env = wrap_deepmind(self.env)
            self.env = wrap_pytorch(self.env)
        #self.env = self.env.seed(seed)
        self.buffer = deque()
        self.num_step = self.cfg['num_step']
        self.nstep_queue = deque(maxlen=self.num_step)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def environment_step(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple:
        """Run one gym step"""
        pass

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def stopping_criterion(self) -> bool:
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass

    def collect_data(self):
        """Fill worker buffer until some stopping criterion is satisfied"""
        self.buffer.clear()
        state = self.env.reset()
 
        while self.stopping_criterion():
            done = False
            while not done:
                action = self.select_action(state)
                transition = self.environment_step(state, action)
                # self.env.render()
                done = transition[-1]
                next_state = transition[-2]
        
                if self.num_step == 1:
                    self.buffer.append(transition)
        
                if self.num_step > 1:
                    self.nstep_queue.append(transition)
                    if (len(self.nstep_queue) == self.num_step) or done:
                        nstep_data = self.preprocess_data(self.nstep_queue)
                        self.buffer.append(nstep_data)
                state = next_state
        
            state = self.env.reset()
            self.nstep_queue.clear()
            #self.write_log()
   
    def get_buffer(self):
        """Return buffer"""
        return self.buffer

    def synchronize(self, new_params: np.ndarray):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
