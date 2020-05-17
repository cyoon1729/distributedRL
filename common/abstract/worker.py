from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple

import numpy as np
import torch.nn as nn

import gym


class Worker(ABC):
    def __init__(
        self,
        worker_id: int,
        worker_brain: nn.Module,
        env: gym.Env,
        seed: int,
        cfg: dict,
    ):
        self.cfg
        self.brain = copy.deepcopy(worker_brain)
        self.env = env
        self.env = env.seed(seed)
        self.buffer = deque()

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        return

    @abstractmethod
    def environment_step(self, state: np.ndarray, action: np.ndarray) -> Tuple, bool:
        """Run one gym step"""
        return

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def stopping_criterion(self) -> bool:
        pass

    def collect_data(self):
        """Fill worker buffer until some stopping criterion is satisfied"""
        self.buffer.clear()
        transitions_added = 0
        state = self.env.reset()

        while self.stopping_criterion():

            transition, done = self.environment_step(state, action)

            if self.num_step == 1:
                self.buffer.append(transition)
            
            if self.num_steps > 1:
                self.nstep_queue.append(transition)
                if (len(self.nstep_queue) == self.num_steps) or done:
                    self.buffer.append(transitions)

            if done:
                state = self.env.reset()
                self.nstep_queue.clear()
                self.write_log()

    def get_buffer(self):
        """Return buffer"""
        return self.buffer

    def synchronize(self, new_params: np.ndarray):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)