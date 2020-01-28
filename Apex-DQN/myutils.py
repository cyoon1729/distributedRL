"""
# ReplayBuffer and PrioritizedReplayBuffer adapted from:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

import torch
import random
import gym
from collections import deque
import numpy as np 
import yaml
import ray 

from segtree import SegmentTree, MinSegmentTree, SumSegmentTree


# huber loss
def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def toTensor(nparray):
    return torch.FloatTensor(nparray).unsqueeze(0)


def to_n_step(transition_buffer, gamma=0.99):
    transition_buffer = list(transition_buffer)
    discounted_reward = 0
    for transition in reversed(transition_buffer[:-1]):
        _, _, reward, _ = transition
        discounted_reward += gamma * reward
    
    state, action, _, _ = transition_buffer[0]
    last_state, _, _, _ = transition_buffer[-1]
    _, _, _, done = transition_buffer[-1]
    
    return np.array(state), action,\
        np.array([discounted_reward]), np.array(last_state), np.array(done)



def read_config(config_path):
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    env = gym.make(cfg['env'])
    cfg['obs_dim'] = env.observation_space.shape
    cfg['action_dim'] = env.action_space.n
    del env

    return cfg


@ray.remote
class ParameterServer(object):

    def __init__(self):
        self.params = []
        self.update_step = 0
    
    def define_param_list(self, param_list):
        self.param_list = param_list
    
    def update_params(self, new_params):
        if len(self.params) < len(self.param_list): # is empty
            for new_param in new_params:
                self.params.append(new_param)
        else:
            for new_param, idx in zip(new_params, range(len(self.param_list))):
                self.params[idx] = new_param

        self.update_step += 1

    def return_params(self):
        return self.params

    def get_update_step(self):
        return self.update_step



class Buffer(object):
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, policy_output, reward, last_state, done):
        experience = (state, policy_output, np.array([reward]), last_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, policy_output, reward, last_state, done = zip(*batch)
        
        return state, policy_output, reward, last_state, done
    
    def __len__(self):
        return len(self.buffer)
    
    def get_size(self):
        return len(self.buffer)
        
    def return_buffer(self):
        return self.buffer

@ray.remote
class PrioritizedReplayBuffer(object):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.
        See Also ReplayBuffer.__init__
        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        #super(PrioritizedReplayBuffer, self).__init__(size)
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def _add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        self._add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        
        sample = [
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        ]

        return sample

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

    def get_size(self):
        return len(self._storage)
    
    def incorporate_buffer(self, buffer):
        # elements of the buffer must be appropriately sized tuples
        for element in buffer:
            self.add(*element)

# @ray.remote
# class PrioritizedReplayBuffer(object):
#     def __init__(self, size, alpha):
#         """Create Prioritized Replay buffer.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         #super(PrioritizedReplayBuffer, self).__init__(size)

#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0

#         assert alpha >= 0
#         self._alpha = alpha

#         self.it_capacity = 1
#         while self.it_capacity < size * 2:  # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
#             self.it_capacity *= 2

#         self._it_sum = SumSegmentTree(self.it_capacity)
#         self._it_min = MinSegmentTree(self.it_capacity)
#         self._max_priority = 1.0

#     def _add(self, obs_t, action, reward, obs_tp1, done): # self, state, policy_output, reward, last_state, done
#         data = (obs_t, action, reward, obs_tp1, done)

#         self._storage.append(data)

#         self._next_idx += 1
    
#     def _remove(self, num_samples):
#         del self._storage[:num_samples]
#         self._next_idx = len(self._storage)

#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(action)
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#             dones.append(done)
        
#         sample = [
#             np.array(obses_t),
#             np.array(actions),
#             np.array(rewards),
#             np.array(obses_tp1),
#             np.array(dones)
#         ]

#         return sample

#     def add(self, state, policy_output, reward, last_state, done): # self, state, policy_output, reward, last_state, done
#         idx = self._next_idx
#         assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

#         self._add(state, policy_output, reward, last_state, done)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha

#     def remove(self, num_samples):
#         self._remove(num_samples)
#         self._it_sum.remove_items(num_samples)
#         self._it_min.remove_items(num_samples)

#     def _sample_proportional(self, batch_size):
#         res = []
#         p_total = self._it_sum.sum(0, len(self._storage) - 1)
#         every_range_len = p_total / batch_size
#         for i in range(batch_size):
#             mass = random.random() * every_range_len + i * every_range_len
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res

#     def sample(self, batch_size, beta, epsilon=1e-8):
#         """Sample a batch of experiences.
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         gammas: np.array
#             product of gammas for N-step returns
#         weights: np.array
#             Array of shape (batch_size,) and dtype np.float32
#             denoting importance weight of each sampled transition
#         idxes: np.array
#             Array of shape (batch_size,) and dtype np.int32
#             idexes in buffer of sampled experiences
#         """
#         assert beta > 0

#         idxes = self._sample_proportional(batch_size)

#         weights = []
#         p_min = self._it_min.min() / (self._it_sum.sum() + epsilon)
#         max_weight = (p_min * len(self._storage)) ** (-beta)

#         for idx in idxes:
#             p_sample = self._it_sum[idx] / (self._it_sum.sum() + epsilon)
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / (max_weight + epsilon))
#         weights = np.array(weights)
#         encoded_sample = self._encode_sample(idxes)

#         return encoded_sample, idxes, weights

#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions.
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#         Parameters
#         ----------
#         idxes: [int]
#             List of idxes of sampled transitions
#         priorities: [float]
#             List of updated priorities corresponding to
#             transitions at the sampled idxes denoted by
#             variable `idxes`.
#         """
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha

#             self._max_priority = max(self._max_priority, priority)
    
#     def get_size(self):
#         return len(self._storage)
    
#     def incorporate_buffer(self, buffer):
#         # elements of the buffer must be appropriately sized tuples
#         for element in buffer:
#             self.add(*element)