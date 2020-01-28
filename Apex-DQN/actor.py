import os
import time 
import random
import torch
import ray
import gym
import numpy as np 

from collections import deque
from myutils import to_n_step, toTensor, Buffer
from models import create_learner


@ray.remote
class Actor(object):

    def __init__(self,
                 worker_id,
                 config,
                 save_dir):
        self.device = config['actor_device']
        self.worker_id = worker_id

        env_id = config['env']
        random_seed = config['random_seed']
        self.unroll_cnt = config['unroll_steps']
        self.num_train_step = config['num_train_steps']
        self.max_ep_length = config['max_ep_length']
        self.actor_buffer_size = config['actor_buffer_size']
        self.local_buffer = Buffer(self.actor_buffer_size)

        self.env = gym.make(env_id)
        random.seed(worker_id+random_seed)
        self.env.seed(random.randint(1,random_seed))

        self.use_distributional = config['use_distributional']
        self.q_network = create_learner(config['network_type'],
                                        config['obs_dim'],
                                        config['action_dim'],
                                        config['num_quantiles'])
        self.eps = 0.2 # epsilon-greedy exploration

        # logging
        self.actor_save_interval = config['actor_save_interval']
        self.save_dir = save_dir
        self.episode = 0
        self.update_step = 0
        self.log = list()

    def sample(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if self.use_distributional:
            dist = self.q_network.forward(state)
            qvals = dist.mean(dim=2)
        else: 
            qvals = self.q_network.forward(state)

        if random.random() > self.eps:
            return np.argmax(qvals.cpu().detach().numpy())
        else:
            return self.env.action_space.sample()

    def set_params(self, new_params):
        for param, new_param in zip(self.q_network.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def start(self):
        print("Starting actor-{0}".format(self.worker_id))
        os.makedirs("{0}/agent-{1}".format(self.save_dir, self.worker_id))
        self.fill_buffer(0, 100)

    def fill_buffer(self, update_step=0, buffer_additions=None):
        if buffer_additions is None:
            buffer_additions = self.actor_buffer_size
        self.update_step = update_step
        transition_buffer = deque(maxlen=self.unroll_cnt + 1)
        state = self.env.reset()
        episode_reward = 0
        transitions_added = 0

        while transitions_added < buffer_additions:
            action = self.sample(state)  
            next_state, reward, done, _ = self.env.step(action)  
            transition = (state, action, reward, done)
            transition_buffer.append(transition)
            episode_reward += reward        

            if len(transition_buffer) == self.unroll_cnt + 1:
                self.local_buffer.add(*to_n_step(transition_buffer))
                transitions_added += 1

            if done:
                # add to log
                logging = {'update_step': self.update_step,
                           'episode': self.episode,
                           'episode reward': episode_reward}
                self.log.append(logging)
                
                if self.episode % 100 == 0:
                    print("Actor-{0}, episode {1} : {2}".format(
                            self.worker_id, self.episode, episode_reward))

                # prepare new rollout
                self.episode += 1
                episode_reward = 0
                transition_buffer.clear()
                next_state = self.env.reset()
                
            if self.episode % self.actor_save_interval == 0:
                save_dir = "{0}/agent-{1}/agent-{1}-rewards.npy".format(
                               self.save_dir, self.worker_id)
                np.save(save_dir, 
                        np.asarray(self.log))
                           
            state = next_state

    def return_buffer(self):
        return self.local_buffer.return_buffer()

    def final_save(self):
        # save results in result file
        print("Saving actor-{self.worker_id} data...".format(self.worker_id))
        
        file_dir = "{0}/agent-{1}/log.pkl".format(
            self.save_dir, 
            self.worker_id)

        with open(file_dir, 'wb') as f:
            pickle.dump(self.log, f)

        print("plotting actor data")
        plot_agent(
            "{0}/agent-{1}/log.pkl".format(self.save_dir, self.worker_id),
            "{0}/agent-{1}".format(self.save_dir, self.worker_id))

        print("Actor-{0} exit".format(self.worker_id))