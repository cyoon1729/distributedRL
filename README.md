# distributedRL


distributedRL is a framework for prototyping disrtibuted reinforcement learning using **Pytorch**, **Ray**, and **ZeroMQ** (and more). Currently, an implementation of *Ape-X DQN* is available, and *IMPALA* and *SEED* are planned to be implemented. 
You can make use of the structural components to easily distribute your reinforcement learning algorithm.

Feel free to reach out (cjy2129 at columbia dot edu) or raise an issue if you have any questions!

### A note about a few choices 
- `ApeXLearner` and `ApeXWorker` are implemented as abstract classes (that inherit `common.abstract.Learner` and `common.abstract.Worker` respectively). To extend Ape-X to any off-policy RL algorithm, you just have to implement algorithm-specific details, such as action selection, environment step, and learning step (check the abstract classes for more detail). 
- I use [ZeroMQ](https://zeromq.org/) for inter-process communication instead of the Ray's built-in features. (1) I wanted the data-passing mechanism and serialization to be a bit more explicit, and (2) ray's inter-process communication forbids calling remote operations, which I found to be a bit restricting for this purpose.  


### References/Papers:
- [Distributed Prioritized Experience Replay (Horgan et al., 2018), ICLR 2018](https://arxiv.org/abs/1803.00933)
