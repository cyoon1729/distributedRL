import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Union

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
import zmq
from zmq.sugar.stopwatch import Stopwatch


class Learner(ABC):
    def __init__(
        self, brain: Union[nn.Module, tuple], learner_cfg: dict, comm_cfg: dict
    ):
        self.cfg = learner_cfg
        self.device = self.cfg["learner_device"]
        self.brain = deepcopy(brain)
        self.replay_data_queue = deque(maxlen=1000)

        # unpack communication configs
        self.param_update_interval = self.cfg["param_update_interval"]
        self.repreq_port = comm_cfg["repreq_port"]
        self.pubsub_port = comm_cfg["pubsub_port"]

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        self.initialize_sockets()

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: tuple):
        pass

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return model params for synchronization"""
        pass

    def params_to_numpy(self, model):
        params = []
        new_model = deepcopy(model)
        state_dict = new_model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param].numpy())
        return params

    def initialize_sockets(self):
        # For sending new params to workers
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pubsub_port}")

        # For receiving batch from, sending new priorities to Buffer # write another with PUSH/PULL for non PER version
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.repreq_port}")

    def publish_params(self, new_params: np.ndarray):
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_replay_data_(self):
        replay_data_id = self.rep_socket.recv()
        replay_data = pa.deserialize(replay_data_id)
        self.replay_data_queue.append(replay_data)

    def send_new_priorities(self, idxes: np.ndarray, priorities: np.ndarray):
        new_priors = [idxes, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)

    def run(self):
        time.sleep(3)
        tracker = Stopwatch()
        self.update_step = 0
        while True:
            self.recv_replay_data_()
            replay_data = self.replay_data_queue.pop()

            for _ in range(self.cfg["multiple_updates"]):
                step_info, idxes, priorities = self.learning_step(replay_data)

            self.update_step = self.update_step + 1

            self.send_new_priorities(idxes, priorities)

            if self.update_step % self.param_update_interval == 0:
                params = self.get_params()
                self.publish_params(params)
