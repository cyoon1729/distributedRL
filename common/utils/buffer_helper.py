from typing import Deque

import ray


@ray.remote
class BufferHelper(object):
    def __init__(self, buffer):
        self.buffer = buffer

    def incorporate_new_data(self, new_data: Deque):
        self.buffer += new_data

    def sample_data(self, batch_size: int):
        return self.buffer.sample(batch_size)
