# Distributed Reinforcement Learning 

Single-machine implementations of distributed reinforcement learning algorithms with [*Ray*](https://github.com/ray-project/ray) and PyTorch. See below for a quick, extremely simple demo of Ray. Also see below for general notes on implementations.

### Implementations include:

- Ape-X DQN (also with Quantile Regression DQN)
- Ape-X DPG
- D4PG (with Quantile Regression DQN instead of C51)

#### Planned:

- Distributed PPO
- IMPALA



### Known Dependencies

```
python = 3.7
pytorch = 1.2
ray
ray[debug]
gym
pybulletgym
pyyaml
```



### General notes:

- Ray allows zero-cost reading of numpy arrays. The distributed components are implemented such that only numpy arrays are communicated across processes (actor -> replay buffer, replay buffer -> learner, learner -> parameter server, parameter server -> actor). 
- At the moment, the implementations only run on single machines—I do not have the experience/resources to implement the algorithms on multi-node settings.
- I also do not have the computational resources to fully learn PyBullet environments, so I cannot provide pre-trained models.



### References/Papers:

- [Distributed Prioritized Experience Replay (Horgan et al., 2018), ICLR 2018](https://arxiv.org/abs/1803.00933)
- [Distributed Distributional Deterministic Policy Gradients (Barth-Maron et al., 2018), ICLR 2018](https://arxiv.org/abs/1804.08617)
- [Distributional Reinforcement Learning with Quantile Regression (Dabney et al., 2017), AAAI 2018](https://arxiv.org/abs/1710.10044)



### Ray demo:

Below is an extremely simple demonstration of multiprocessing using Ray. The names of the classes are meaningless. For more information, check out Ray's [official tutorial](https://github.com/ray-project/tutorial) and [documentation](https://ray.readthedocs.io/en/latest/index.html).

```python
import ray 
import random
from collections import deque 
import time 

ray.init()


@ray.remote
class Buffer_remote(object):
    
    def __init__(self):
        self.buffer = deque(maxlen=2000)
    
    def add(self, x):
        self.buffer.append(x)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def get_buffer(self):
        return self.buffer
    
    def get_size(self):
        return len(self.buffer)


@ray.remote
class Actor(object):

    def __init__(self, actor_id, learner_id, buffer_remote_id):
        self.actor_id = actor_id
        self.learner_id = learner_id
        self.buffer_remote_id = buffer_remote_id 

    def run(self):
        for i in range(1000):
            self.buffer_remote_id.add.remote(i)
            print(self.actor_id)
            print("from learner ", ray.get(self.learner_id.get_msg.remote()))

    
@ray.remote
class Learner(object):

    def __init__(self, buffer_remote_id):
        self.buffer_remote_id = buffer_remote_id
    
    def read(self):
        for _ in range(1000):
            if ray.get(self.buffer_remote_id.get_size.remote()) > 5:
                batch = ray.get(self.buffer_remote_id.sample.remote(5))
                print(batch)
    
    def get_msg(self):
        return random.randint(0, 100)

       
if __name__ == "__main__":
    buffer_id = Buffer_remote.remote()
    learner_id = Learner.remote(buffer_id)
    actors_ids = [Actor.remote(i, learner_id, buffer_id) for i in range(2)]

    procs = [actors_ids[0].run.remote(), learner_id.read.remote(), actors_ids[1].run.remote()]
    ray.wait(procs)
    ray.timeline()
```

