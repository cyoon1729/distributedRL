import os
import time 
import random
import torch
import ray
import gym
import numpy as np 

from collections import deque
from myutils import to_n_step, toTensor, OUNoise
from models import create_policy


@ray.remote
class Actor(object):

    def __init__(self, worker_id, config, save_dir, buffer_remote, param_server):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.worker_id = worker_id

        env_id = config['env']
        random_seed = config['random_seed']
        self.unroll_cnt = config['unroll_steps']
        self.num_train_step = config['num_train_steps']
        self.max_ep_length = config['max_ep_length']

        self.env = gym.make(env_id)
        random.seed(worker_id+random_seed)
        self.env.seed(random.randint(1,random_seed))

        self.policy = create_policy(config['obs_dim'], config['action_dim'])
        self.noise = OUNoise(config['action_space'])

        # parallelization
        self.buffer_remote = buffer_remote
        self.param_server = param_server

        # logging
        self.save_dir = save_dir
        self.log = list()


    def sample(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action

    def sync_with_param_server(self):
        new_params = ray.get(self.param_server.return_params.remote())
        for param, new_param in zip(self.policy.parameters(), new_params):
            new_param = torch.Tensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def run(self):
        time.sleep(3)
        transition_buffer = deque(maxlen=self.unroll_cnt + 1)
        state = self.env.reset()
        episode_reward = 0 
        episode = 0
        ep_length = 0
        update_step = ray.get(self.param_server.get_update_step.remote())
        
        while update_step < self.num_train_step:
            ep_length += 1
            action = self.sample(state)  
            next_state, reward, done, _ = self.env.step(self.noise.get_action(action))  
            transition_buffer.append((state, action, reward, done))
            episode_reward += reward            

            if len(transition_buffer) == self.unroll_cnt + 1:
                self.buffer_remote.add.remote(*to_n_step(transition_buffer))

            if done or ep_length == self.max_ep_length:
                # add to log
                logging = {'update_step': update_step,
                    'episode': episode, 'episode reward': episode_reward}
                self.log.append(logging)

                # print results locally
                print(f"Episode {episode}: {episode_reward}")

                # prepare new rollout
                episode += 1
                episode_reward = 0
                ep_length = 0
                transition_buffer.clear()
                next_state = self.env.reset()
            
            if update_step % 3 == 0:
                self.sync_with_param_server()
            
            state = next_state
            update_step = ray.get(self.param_server.get_update_step.remote())
        
        # save results in result file
        print(f"Saving actor-{self.worker_id} data...")
        os.makedirs(f'{self.save_dir}/agent-{self.worker_id}')
        with open(f'{self.save_dir}/agent-{self.worker_id}/log.pkl', 'wb') as f:
            pickle.dump(self.log, f)

        print(f"Actor-{self.worker_id} exit")