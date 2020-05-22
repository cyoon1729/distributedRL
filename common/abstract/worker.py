from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Deque
import datetime as datetime

import zmq
import pyarrow as pa
import numpy as np
import torch
import torch.nn as nn
import random
from common.utils.utils import create_env


class Worker(ABC):
    def __init__(
        self, worker_brain: nn.Module, worker_id: int, worker_cfg: dict, comm_cfg: dict
    ):
        self.worker_id = worker_id
        self.cfg = worker_cfg
        self.devuce = worker_cfg["worker_device"]
        self.brain = deepcopy(worker_brain).to(self.device)
        
        # create env
        random.seed(self.worker_id)
        self.env = create_env(self.cfg["env_name"], self.cfg["atari"])
        self.seed = random.randin(1, 999)
        self.env.seed(self.seed)

        # unpack communication configs
        self.pubsub_port = comm_cfg["pubsub_port"]
        self.pullpush_port = comm_cfg["pullpush_port"]

        # initialize zmq sockets
        print(f"[Worker: {self.worker_id}]: initializing sockets..")
        self.initialize_sockets()

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action with worker's brain"""
        pass    def send_replay_data(self):
            ray, action: np.ndarray) -> tuple:
        """Run one gym env step"""
        pass

    @abstractmethod
    def write_log(self):
        """Log performance (e.g. using Tensorboard)"""
        pass

    @abstractmethod
    def preprocess_data(self, data) -> list:
        """Preprocess collected data if necessary (e.g. n-step)"""
        pass

    @abstractmethod
    def collect_data(self) -> list:
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    def synchronize(self, new_params: list):
        """Synchronize worker brain with parameter server"""
        for param, new_param in zip(self.brain.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def initialize_sockets(self):
        # for receiving params from learner
        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.pubsub_port}")

        # for sending replay data to buffer: MOVE THIS TO WORKER SET?
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://127.0.0.1:{self.pullpush_port}")
    
    def send_replay_data(self, replay_data):
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data)
    
    def receive_new_params(self):
        new_params_id = self.sub_socket.recv()
        if new_params_id:
            new_params = pa.deserialize(new_params_id)
            self.synchronize(new_params)
        else:
            pass
    
    def run(self):
        while True:
            local_buffer = self.collect_data()
            self.send_replay_data(local_buffer)
            self.receive_new_params()
            

class ApeXWorker(Worker):
    """Abstract class for ApeX distrbuted workers """

    def __init__(self, worker_id: int, worker_brain: nn.Module, cfg: dict):
        super().__init__(worker_id, worker_brain, cfg)
        self.nstep_queue = deque(maxlen=self.cfg["num_step"])
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["num_step"]
        # print(f"worker params: {next(self.brain.parameters()).is_cuda}")

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)
        return nstep_data

    def compute_priorities(self):
        pass

    def collect_data(self, verbose=False):
        """Fill worker buffer until some stopping criterion is satisfied"""
        local_buffer = []
        nstep_queue = deque(maxlen=self.num_step)

        while len(local_buffer) < self.worker_buffer_size:
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
                if (len(nstep_queue) == self.num_step) or done:
                    nstep_data = self.preprocess_data(nstep_queue)
                    buffer.append(nstep_data)

                state = next_state

            if verbose:
                if episode_reward > 0:
                    print(f"Worker {self.worker_id}: {episode_reward}")
            
        return local_buffer