from typing import Deque

import ray
import asyncio


@ray.remote
class BufferHelper(object):
    def __init__(self, buffer):
        self.buffer = buffer

    async def recv_new_data(self, new_data: Deque):
        for data in new_data:
            self.buffer.add(*data)

    async def sample_data(self, batch_size: int, priority_beta):
        return self.buffer.sample(batch_size, priority_beta)
    
    async def update_priorities(self, idxes, new_priorities):
        self.buffer.update_priorities(idxes, new_priorities)
