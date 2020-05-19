import sys
sys.path.append('../common')

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.abstract.worker import Worker  
from common.abstract.learner import Learner 



@ray.remote
class DQNWorker(Worker):
    
    def __init__(
        self,
        worker_id: int,
        worker_brain: nn.Module,
        env_name: str,
        seed: int,
        cfg: dict,
    ): 
        super().__init__(worker_id, worker_brain, env_name, seed, cfg)
        self.num_updates = 0
        self.max_num_updates = self.cfg['max_num_updates']
        self.worker_buffer_size = self.cfg['worker_buffer_size']
        self.eps_greedy = self.cfg['eps_greedy']
        self.eps_decay = self.cfg['eps_decay']
        self.gamma = self.cfg['gamma']

    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.eps_greedy = self.eps_greedy * self.eps_decay
        if(np.random.randn() > self.eps_greedy):
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        qvals = self.brain.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        return action

    def environment_step(
        self, state: np.ndarray, action: np.ndarray
    ) -> tuple:
        next_state, reward, done, _ = self.env.step(action)
        return (state, action, reward, next_state, done)

    def preprocess_data(self, nstepqueue):
        discounted_reward = 0
        _, _, _,last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward  
        nstep_data = (state, action, discounted_reward, last_state, done)
        return nstep_data

    def write_log(self):
        print("TODO: include Tensorboard..")

    def stopping_criterion(self) -> bool:
       	return len(self.buffer) < self.worker_buffer_size


@ray.remote
class DQNLearner(Learner):

    def __init__(self, brain, cfg: dict):
        super().__init__(brain, cfg)
        self.num_step = self.cfg['num_step']

        self.network = self.brain[0]
        self.target_network = self.brain[1]
        self.network_optimizer = torch.optim.Adam(self.network.parameters())
        self.target_optimizer = torch.optim.Adam(self.target_network.parameters())

    def write_log(self):
        print("TODO: incorporate Tensorboard...")
    
    def learning_step(self, data: tuple):
        states, actions, rewards, next_states, dones, weights, idxes = data
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        curr_q1 = self.network.forward(states).gather(1, actions.unsqueeze(1))
        curr_q2 = self.target_network.forward(states).gather(1, actions.unsqueeze(1)) 
        
        bootstrap_q = torch.min(
            torch.max(self.network.forward(next_states), 1)[0],
            torch.max(self.target_network.forward(next_states), 1)[0]
        )

        bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        target_q = rewards + (1 - dones) * self.gamma**self.num_step * bootstrap_q
        weights = torch.FloatTensor(weights).to(self.device).mean()

        loss1 = weights * F.mse_loss(curr_q1, target_q.detach())
        loss2 = weights * F.mse_loss(curr_q2, target_q.detach())
        
        self.network_optimizer.zero_grad()
        loss1.backward()
        self.network_optimizer.step()

        self.target_optimizer.zero_grad()
        loss2.backward()
        self.target_optimizer.step()


    def get_params(self):
        return self.params_to_numpy(self.network)

    def get_worker_brain_sample(self):
        return self.network 