import numpy as np
import ray
import asyncio

@ray.remote
class ParameterServer(object):
    def __init__(self, workers):
        self.params = []
        self.update_step = 0
        self.workers = workers

    def initialize(self, initial_params):
        for param in initial_params:
            self.params.append(param)

    def update_params(self, new_params):
        """Receive and synchronize new parameters"""
        self.update_step = self.update_step + 1
        for new_param, idx in zip(new_params, range(len(new_params))):
            self.params[idx] = new_param

    def get_params(self):
        return self.params
    
    def get_update_step(self):
        return self.update_step

    async def recv_params(self, new_params):
        self.update_params(new_params)
        