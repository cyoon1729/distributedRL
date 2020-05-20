from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Deque
import datetime as datetime

import numpy as np
import torch
import torch.nn as nn

from common.utils.utils import create_env


class Worker(ABC):
    def __init__(
        self, worker_id: int, worker_brain: nn.Module, seed: int, cfg: dict,
    ):
        self.cfg = cfg
        self.worker_id = worker_id
        self.device = torch.device(self.cfg["worker_device"])
        self.brain = deepcopy(worker_brain)
        self.buffer = deque()
        self.env = create_env(self.cfg["env_name"], self.cfg["atari"])
        self.env.seed(seed)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action with worker's brain"""
        pass

    @abstractmethod
    def environment_step(self, state: np.ndarray, action: np.ndarray) -> tuple:
        """Run one gym env step"""
        pass

    @abstractmethod
    def write_log(self):
        """Log performance (e.g. using Tensorboard)"""
        pass

    @abstractmethod
    def stopping_criterion(self) -> bool:
        """Stopping criterion for collect_data()"""
        pass

    @abstractmethod
    def preprocess_data(self, data):
        """Preprocess collected data if necessary (e.g. n-step)"""
        pass

    @abstractmethod
    def collect_data(self):
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    def get_buffer(self):
        """Return buffer"""
        return self.buffer

    def synchronize(self, new_params: list):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)



class ApeXWorker(Worker):
    """Abstract class for ApeX distrbuted workers """

    def __init__(self, worker_id: int, worker_brain: nn.Module, seed: int, cfg: dict):
        super().__init__(worker_id, worker_brain, seed, cfg)
        self.nstep_queue = deque(maxlen=self.cfg["num_step"])
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["num_step"]

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)
        return nstep_data

    def collect_data(self):
        """Fill worker buffer until some stopping criterion is satisfied"""
        self.buffer.clear()
        state = self.env.reset()

        while self.stopping_criterion():
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                transition = self.environment_step(state, action)
                next_state = transition[-2]
                done = transition[-1]
                reward = transition[-3]
                episode_reward += reward

                self.nstep_queue.append(transition)
                if (len(self.nstep_queue) == self.num_step) or done:
                    nstep_data = self.preprocess_data(self.nstep_queue)
                    self.buffer.append(nstep_data)

                state = next_state
            # print(f"Worker {self.worker_id}: {episode_reward}")
            state = self.env.reset()
            self.nstep_queue.clear()
