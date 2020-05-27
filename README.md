# distributedRL


distributedRL is a framework for prototyping disrtibuted reinforcement learning using **Pytorch** and **Ray**. Out-of-the-box, I provide implementations of Ape-X DQN, IMPALA, and SEED. But you can make use of the structural components to easily distribute your reinforcement learning algorithm.

See below for *a note about design choice* and *quick tutorial* and *a short tutorial on Ray*.

### A note about design choice (and Ape-X):
Although Ape-X, IMPALA, and SEED are all introduced distributed architectures, IMPALA and SEED are techincally "algorithms," since they exclusively use V-trace as the learning algorithm. 

On the other hand, Ape-X is indeed an "architecture," since it can be applied to any (off-policy) RL algorithm (DQN, DDPG, SAC, ...). 

With this in mind, I created another set of abstract classes ApeXWorker and ApeXLearner, which can be used to incorporate ApeX to any appropriate algorithm. In particular, you only have to implement algorithm-specific features, such as action selection, environment step, and learning step, to distribute your algorithm. 

### A quick tutorial   



### Ray demo:
Below is an extremely simple demonstration of Ray. For more information, check out Ray's [official tutorial](https://github.com/ray-project/tutorial) and [documentation](https://ray.readthedocs.io/en/latest/index.html).


### References/Papers:

- [Distributed Prioritized Experience Replay (Horgan et al., 2018), ICLR 2018](https://arxiv.org/abs/1803.00933)
- [Distributed Distributional Deterministic Policy Gradients (Barth-Maron et al., 2018), ICLR 2018](https://arxiv.org/abs/1804.08617)
- [Distributional Reinforcement Learning with Quantile Regression (Dabney et al., 2017), AAAI 2018](https://arxiv.org/abs/1710.10044)
- [@schatty's implementation of D4PG](https://github.com/schatty/d4pg-pytorch)

