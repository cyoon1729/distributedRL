import ray
import pyarrow as pa
import asyncio
import numpy as np
import zmq

class WorkerSet:
    def __init__(self):
        msg = np.random.rand(1, 3)
        self.object_id = ....
        print("start!")
            
    @ray.remote
    def send(self):
        port = 5556
        context = zmq.Context()
        send_socket = context.socket(zmq.PUSH)
        send_socket.bind(f"tcp://127.0.0.1:{port}")

        print("send_loop starting....")
        while True:
            msg = np.random.rand(300,300)
            #object_id = ray.put(msg)
            object_id = pa.serialize(msg).to_buffer()
            send_socket.send(object_id)

    @ray.remote
    def multiple_send(self):
        port = 5556
        context = zmq.Context()
        send_socket = context.socket(zmq.PUSH)
        send_socket.bind(f"tcp://127.0.0.1:{port}")

        senders = {}
        for s_id in range(1, 4):
            senders[self.send.remote(self)] = 
        
        while senders:
            ready_sender_list, _ = ray.wait(list(senders))
            ready_sender = ready_sender_list[0]
            senders.pop(ready_sender)
            msg = ray.get(ready_sender)

            self.object_id = ray.put(msg)

            senders[self.send.remote(self)] 

    @ray.remote
    def recv(self):        
        port = 5556
        context = zmq.Context()
        recv_socket = context.socket(zmq.PULL)
        recv_socket.connect(f"tcp://127.0.0.1:{port}")

        print("recv starting....")
        while True:
            # await self.object_id
            object_id = recv_socket.recv()
            msg = pa.deserialize(object_id)
            #msg = ray.get(object_id)
            print(msg)

ray.init()

if __name__ == "__main__":
    worker = WorkerSet()
    ray.wait([worker.send.remote(worker), worker.recv.remote(worker)])