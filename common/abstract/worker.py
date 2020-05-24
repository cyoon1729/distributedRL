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
        self.param_queue = deque(maxlen=1)

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
    def collect_data(self) -> list:
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    @abstractmethod
    def run(self):
        """Run main loop """
        pass

    async def recv_params(self, new_params):
        """This function is called from the learner class"""
        self.param_queue.append(new_params)

    def update_params(self, new_params: list):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)


class ApeXWorker(Worker):
    """Abstract class for ApeX distrbuted workers """

    def __init__(self, worker_id: int, worker_brain: nn.Module, seed: int, cfg: dict):
        super().__init__(worker_id, worker_brain, seed, cfg)
        self.nstep_queue = deque(maxlen=self.cfg["num_step"])
        self.local_buffer = []
        self.max_local_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["num_step"]

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)

        q_value = self.brain.forward(
            torch.FloatTensor(state).unsqueeze(0).to(self.device),
            torch.FloatTensor(action).unsqueeze(0).to(self.device)
        )

        bootstrap_q = torch.max(
            self.brain.forward(
                torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            ),
            1
        )[0]

        target_q_value = discounted_reward + self.gamma**self.num_step * bootstrap_q
        priority_value = torch.abs(target_q_value - q_value).detach().view(-1)
        priority_value = priority_value.cpu().numpy().tolist()
        
        return nstep_data, priority_value

    def collect_data(self):
        """Fill worker buffer until some stopping criterion is satisfied"""
        local_buffer = []
        nstep_queue = deque(maxlen=self.num_step)

        state = self.env.reset()
        while len(local_buffer) < self.max_local_buffer_size:
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                transition = self.environment_step(state, action)
                next_state = transition[-2]
                done = transition[-1]
                reward = transition[-3]
                episode_reward += reward

                nstep_queue.append(transition)
                if (len(nstep_queue) == num_step) or done:
                    replay_data = self.preprocess_data(step_queue)
                    local_buffer.append(replay_data)

                state = next_state
            # print(f"Worker {self.worker_id}: {episode_reward}")
            state = self.env.reset()

        return local_buffer 

    async def run(self, global_buffer_handle):
        while True:
            local_buffer = self.collect_data()
            global_buffer_handle.recv_new_data(local_buffer)
            if self.param_queue:
                new_params = self.param_queue.pop()
                self.update_params(new_params)