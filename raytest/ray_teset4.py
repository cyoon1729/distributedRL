import ray
import numpy as np
from collections import deque
import random

ray.init()

@ray.remote
def send(receive_server_handle):
    while True:
        msg = np.random.rand(1, 3)
        receive_server_handle.recv.remote(msg)


@ray.remote 
class ReceiveServer:
    def __init__(self):
        self.storage = deque(maxlen=2000)

    def recv(self, data):
        self.storage.append(data)
        
    def process(self):
        while True:
            if len(self.storage) > 0:
                data = random.sample(self.buffer, 1)
                
                # do something to data
                # ...

                print(data)

receive_server = ReceiveServer.remote()
ray.wait([send.remote(receive_server), receive_server.process.remote()])
