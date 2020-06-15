# distributedRL


distributedRL is a framework for prototyping disrtibuted reinforcement learning using **Pytorch**, **Ray**, and **ZeroMQ** (and more). You can make use of the structural components to easily distribute your reinforcement learning algorithm, on a single-machine.

Currently, an implementation of *Ape-X DQN* is available. *IMPALA* and *SEED* are planned to be implemented. 

Feel free to reach out (cjy2129 at columbia dot edu) or raise an issue if you have any questions!

#### Task List
- `single_agent_test.py` (not distributed) has memory leak; ram usage drastically increases due to buffer. `apex_dqn` does not suffer any memory leaks. 
- Something wrong with dqn learning step; pong test hovers around -10 ~ 0 points. 
- Learner should broadcast params `with the corresponding learning step` (add).

### A note about a few choices 
- `ApeXLearner` and `ApeXWorker` are implemented as abstract classes (that inherit `common.abstract.Learner` and `common.abstract.Worker` respectively). To extend Ape-X to any off-policy RL algorithm, you just have to implement algorithm-specific details, such as action selection, environment step, and learning step (check the abstract classes for more detail). 
- I use [ZeroMQ](https://zeromq.org/) for inter-process communication instead of the Ray's built-in features. (1) I wanted the data-passing mechanism and serialization to be a bit more explicit, and (2) ray's inter-process communication forbids calling remote operations, which I found to be a bit restricting for this purpose.  

### Installation
clone the repository, then
```
conda env create -f environment.yml
conda activate distrl
pip install -e .
```

### A Short Tutorial
*To be added*

### Benchmarking
*To be added*

### Acknowledgements
I thank @Avrech for his helpful discussion, suggestions and enhancements. In particular, on identifying bugs and bottlenecks and improving asynchrony of the Ape-X implementation.

### References/Papers:
- [Distributed Prioritized Experience Replay (Horgan et al., 2018), ICLR 2018](https://arxiv.org/abs/1803.00933)
