from typing import Deque

import ray


@ray.remote
class BufferHelper(object):
    def __init__(self, buffer):
        self.buffer = buffer

    def incorporate_new_data(self, new_data: Deque):
        for data in new_data:
            self.buffer.add(*data)

    def sample_data(self, batch_size: int, priority_beta):
        return self.buffer.sample(batch_size, priority_beta)
