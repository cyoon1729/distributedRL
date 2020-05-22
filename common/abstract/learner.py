from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import pyarrow as pa
import zmq

@ray.remote(num_gpus=1)
class Learner(ABC):
    def __init__(self, brain: Union[nn.Module, tuple], learner_cfg: dict, comm_cfg: dict):
        self.cfg = learner_cfg
        self.device = torch.device(self.learner_cfg["learner_device"])
        self.brain = deepcopy(brain)
        
        self.gamma = self.cfg["gamma"]
        self.param_update_interval = self.cfg["param_update_interval"]

        # unpack communication configs
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

    @abstractmethod
    def get_worker_brain_sample(self):
        """for initializing workers"""
        pass

    def params_to_numpy(self, model):
        params = []
        state_dict = model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param])
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
        return replay_data

    def send_new_priorities(self, idxes: np.ndarray, priorities: np.ndarray):
        new_priors = [idxes, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)
    
    def run(self):
        self.update_step = 0
        while True:
            replay_data = self.recv_replay_data_()
            step_info, idxes, priorities = self.learning_step(replay_data)
            self.update_step = self.update_step + 1
            self.send_new_priorities(idxes, priorities)

            if self.update_step % self.param_update_interval == 0:
                params = self.get_params()
                self.publish_params(params)
