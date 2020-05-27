import numpy as np
import pyarrow as pa
import ray
import zmq
import torch
ray.init()


@ray.remote
def send():
    port = 5556
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    send_socket.bind(f"tcp://127.0.0.1:{port}")

    while True:
        msg = np.random.rand(1, 3) # this could be much bigger 
        object_id = pa.serialize(msg).to_buffer()
        send_socket.send(object_id)

@ray.remote(num_gpus=1)
def recv():        
    port = 5556
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.connect(f"tcp://127.0.0.1:{port}")

    while True:
        object_id = recv_socket.recv()
        msg = pa.deserialize(object_id)
        print(msg)


if __name__ == "__main__":
    ray.wait([send.remote(), recv.remote()])