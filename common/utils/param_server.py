import ray
import numpy as np 


@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.params = []

    def initialize(self, initial_params):
    	for param in initial_params:
    		self.params.append(param)

    def update_params(self, new_params):
    	"""Receive and synchronize new parameters"""
    	for new_param, idx in zip(new params, range(len(new_params))):
    		self.params[idx] = new_param

    def get_params(self):
        return self.params