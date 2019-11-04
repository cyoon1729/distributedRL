import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import copy 
import time
import ray
import os 
import pickle 

from models import create_critic, create_policy
from myutils import huber


@ray.remote
class Learner:

    def __init__(self, config, buffer_remote, param_server, save_dir):
        self.device = torch.device(config['device'])
        self.num_train_steps = config['num_train_steps']   

        # environment_info
        network_type = config['network_type']
        obs_dim = config['obs_dim']
        action_dim = config['action_dim']
        
        # buffer info
        self.use_per = config['use_per']
        self.priority_alpha = config['priority_alpha']
        self.priority_beta_start = config['priority_beta_start']
        self.priority_beta_end = config['priority_beta_end']
        self.priority_beta = self.priority_beta_start
        self.priority_beta_increment = ((self.priority_beta_end - self.priority_beta_start)
                                        / self.num_train_steps)
        self.batch_size = config['batch_size']

        # hyperparameters
        self.gamma = config['gamma']
        self.unroll_steps = config['unroll_steps']
        q_lr = config['q_lr']
        policy_lr = config['policy_lr']
        self.tau = config['tau']

        # initialize networks
        self.q_network = create_critic(network_type, obs_dim, action_dim)
        self.target_q_network = create_critic(network_type, obs_dim, action_dim)
        self.policy = create_policy(obs_dim, action_dim)
        self.target_policy = create_policy(obs_dim, action_dim)

        self.param_list = list(self.policy.state_dict())
        self.param_server = param_server

        for target_param, param in zip(self.target_q_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_policy.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # parallelization
        self.buffer_remote = buffer_remote
        self.update_step = 0

        # logging
        self.save_dir = save_dir
        self.log = list()

    def optimize_parameters(self, batch, weights, eps=1e-4):
        states, actions, rewards, last_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        last_states = torch.FloatTensor(last_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        curr_q = self.q_network.forward(states, actions)
        last_actions = self.target_policy.forward(last_states)
        bootstrap_q = self.target_q_network.forward(last_states, last_actions)
        q_targets = (rewards + (1 - dones) *
                    self.gamma**self.unroll_steps * bootstrap_q)

        td_error = q_targets - curr_q + eps  # prevent priority from being 0
        q_loss = F.mse_loss(curr_q, q_targets.detach())

        if self.use_per:
            weights = torch.FloatTensor(weights).to(self.device).mean()
            q_loss = weights * q_loss
        
        # update q network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update policy
        policy_loss = -self.q_network.forward(states, self.policy.forward(states)).mean() 
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.target_q_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        
        for target_param, param in zip(self.target_policy.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return q_loss, policy_loss, td_error

    def return_update_step(self):
        return self.update_step
    
    def update_param_server(self):
        params = []
        state_dict = self.policy.state_dict()
        for param in self.param_list:
            params.append(state_dict[param].numpy())
        self.param_server.update_params.remote(params)
    
    def return_param_list(self):
        return list(self.policy.state_dict())
    
    def run(self):
        while ray.get(self.buffer_remote.get_size.remote()) < self.batch_size:
            continue

        while self.update_step < self.num_train_steps:
            # fetch batch
            if self.use_per:
                batch, indices, weights = ray.get(self.buffer_remote.sample.remote(self.batch_size, self.priority_beta))
                self.priority_beta += self.priority_beta_increment
            else:
                batch = ray.get(self.buffer_remote.sample.remote(self.batch_size))
                weights = None   # weights are set to non because no prioritized replay

            # update parameters
            q_loss, policy_loss, td_error = self.optimize_parameters(batch, weights)

            # update prioritized for prioritized experience replay
            if self.use_per:
                new_priorities = td_error.detach().squeeze().abs().cpu().numpy().tolist() 
                self.buffer_remote.update_priorities.remote(indices, new_priorities)

            # log
            logging = {'update_step': self.update_step, 'q_loss': q_loss, 'td_error': td_error, 'policy_loss': policy_loss}
            self.log.append(logging)

            # sync with global
            #self.global_policy_remote.update.remote(self.policy.state_dict())
            self.update_param_server()
            self.update_step += 1

            if self.update_step % 100 == 0:
                print(f"learner update step: {self.update_step}")

        # save results in result file
        print("Saving learner data...")
        os.makedirs(f'{self.save_dir}/learner')
        with open(f'{self.save_dir}/learner/log.pkl', 'wb') as f:
            pickle.dump(self.log, f)

        print("Learner exit")