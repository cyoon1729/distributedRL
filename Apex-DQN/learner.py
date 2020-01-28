import os 
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import ray

from models import create_learner
from myutils import huber


@ray.remote
class Learner:

    def __init__(self, config, save_dir):
        # read config 
        self.device = torch.device(config['learner_device'])
        self.num_train_steps = config['num_train_steps']   

        # environment_info
        network_type = config['network_type']
        obs_dim = config['obs_dim']
        action_dim = config['action_dim']
        
        # buffer info
        self.use_per = config['use_per']
        self.priority_alpha = config['priority_alpha']
        self.batch_size = config['batch_size']

        # hyperparameters
        self.gamma = config['gamma']
        self.unroll_steps = config['unroll_steps']
        q_lr = config['q_lr']
        self.tau = config['tau']
        self.use_distributional = config['use_distributional']
        num_quantiles = config['num_quantiles']

        # initialize networks
        self.q_network = create_learner(network_type,
                                        obs_dim,
                                        action_dim)
        self.target_q_network = create_learner(network_type,
                                               obs_dim,
                                               action_dim)
        self.param_list = list(self.q_network.state_dict())

        for target_param, param in zip(self.q_network.parameters(),
                                       self.target_q_network.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
    
        # logging
        self.update_step = 0 
        self.save_dir = save_dir
        self.log = list()
        os.makedirs("{0}/learner".format(self.save_dir))
        os.makedirs("{0}/learner/models".format(self.save_dir))

    def optimize_parameters(self, batch, weights, eps=1e-4):
        states, actions, rewards, last_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).view(-1, 1)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        last_states = torch.FloatTensor(last_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        if self.use_distributional:
            dist = self.q_network.forward(states)
            dist = dist[torch.arange(dist.size(0)), actions.view(-1)]
            bootstrap_q = self.target_q_network.forward(last_states)
            bootstrap_q_max = torch.max(
                bootstrap_q.mean(dim=2), 1, keepdim=True)[0]
            q_targets = (rewards + (1 - dones) *
                        self.gamma**self.unroll_steps * bootstrap_q_max)

            distance = q_targets.detach() - dist
            q_loss = (huber(distance) * 
                     (self.q_network.tau - (distance.detach() < 0).float()).abs())
            q_loss = q_loss.mean()

            q_target_values = q_targets.mean(dim=1).detach().view(-1)
            curr_q_values = dist.mean(dim=1).detach().view(-1)
            td_error = q_target_values - curr_q_values + eps # prevent priority from being 0

            if self.use_per:
                weights = torch.FloatTensor(weights).to(self.device).mean()
                q_loss = weights * q_loss
        else:
            curr_q = self.q_network.forward(states).gather(1, actions)
            bootstrap_q = self.target_q_network.forward(last_states)
            bootstrap_q_max = torch.max(bootstrap_q, 1, keepdim=True)[0]
            q_targets = (rewards + (1 - dones) *
                        self.gamma**self.unroll_steps * bootstrap_q_max)

            td_error = q_targets - curr_q + eps  # prevent priority from being 0
            q_loss = F.mse_loss(curr_q, q_targets.detach())
            if self.use_per:
                weights = torch.FloatTensor(weights).to(self.device).mean()
                q_loss = weights * q_loss
        
        # update q network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update target q target 
        for target_param, param in zip(self.target_q_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return q_loss, td_error
    
    def return_param_list(self):
        return self.param_list
    
    def return_numpy_policy(self):
        # return numpy state_dict
        params = []
        state_dict = self.q_network.state_dict()
        for param in self.param_list:
            params.append(state_dict[param].cpu().numpy())
        
        return params 

    def return_update_step(self):
        return self.update_step

    def learning_step(self, batch, indices, weights):
        step_info = list(self.optimize_parameters(batch, weights))
        if self.use_per:
            _, td_error = step_info
            new_priorities = td_error.detach().squeeze().abs()
            new_priorities = new_priorities.cpu().numpy().tolist() 
        else:
            None
        self.update_step += 1

        return step_info, indices, new_priorities
    
    def logger(self, step_info):
        q_loss, td_error = step_info
        logging = {
            'update_step': self.update_step,
            'q_loss': q_loss.detach().numpy(),
            'td_error': td_error.detach().numpy(),
        }
        self.log.append(logging)

        if self.update_step % 100 == 0:
            learner_state = {
                'epoch': self.update_step,
                'q_state_dict': self.q_network.state_dict(),
                'q_optimizer': self.q_optimizer.state_dict(),
            }
            torch.save(
                learner_state,
                "{0}/learner/models/learner_state-{1}.pth".format(
                    self.save_dir,
                    self.update_step)
            )
            
            print("learner update step: {0}".format(self.update_step))