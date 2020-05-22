import ray
import numpy as np
from collections import deque
import random

ray.init()



@ray.remote
def send(receive_server_handle, learner_handle):
    while True:
        msg = np.random.rand(1, 3)
        receive_server_handle.recv.remote(msg, learner_handle)


@ray.remote 
class Buffer:
    def __init__(self):
        self.buffer = deque(maxlen=2000)

    def recv(self, data, learner_handle):
        self.buffer.append(data)
        learner_handle.recv.remote(data)

    def send_to_learner(self, learner_handle):
        while True:
            if len(self.buffer) > 32:
                data = random.sample(self.buffer, 1)
                print(data)
                learner_handle.recv.remote(data)

@ray.remote
class Learner:
    def __init__(self):
        self.data = None 

    def recv(self, data):
        print(data)


buffer = Buffer.remote()
learner = Learner.remote()
print(f"learner pid: {learner}")
x = input()
ray.get([send.remote(buffer, learner)])

