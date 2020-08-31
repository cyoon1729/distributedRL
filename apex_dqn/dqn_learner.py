from copy import deepcopy

import ray
import torch
import torch.nn.functional as F

from common.abstract.learner import Learner
from zmq.sugar.stopwatch import Stopwatch
from torch.nn.utils import clip_grad_norm_


@ray.remote(num_gpus=1)
class DQNLearner(Learner):
    def __init__(self, brain, cfg: dict, comm_config: dict):
        super().__init__(brain, cfg, comm_config)
        self.num_step = self.cfg["num_step"]
        self.gamma = self.cfg["gamma"]
        self.tau = self.cfg["tau"]
        self.gradient_clip = self.cfg["gradient_clip"]
        self.q_regularization = self.cfg["q_regularization"]
        self.network = self.brain[0]
        self.network.to(self.device)
        self.target_network = self.brain[1]
        self.target_network.to(self.device)
        self.network_optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.cfg["learning_rate"]
        )

    def write_log(self):
        print("TODO: incorporate Tensorboard...")

    def learning_step(self, data: tuple):
        states, actions, rewards, next_states, dones, weights, idxes = data

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        # Toggle with comments if not using cuda
        states.cuda(non_blocking=True)
        actions.cuda(non_blocking=True)
        rewards.cuda(non_blocking=True)
        next_states.cuda(non_blocking=True)
        dones.cuda(non_blocking=True)

        curr_q = self.network.forward(states).gather(1, actions.unsqueeze(1))
        bootstrap_q = torch.max(self.target_network.forward(next_states), 1)[0]

        bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        target_q = rewards + (1 - dones) * self.gamma ** self.num_step * bootstrap_q
        weights = torch.FloatTensor(weights).to(self.device)
        weights.cuda(non_blocking=True)
        weights = weights.mean()

        q_loss = (
            weights * F.smooth_l1_loss(curr_q, target_q.detach(), reduction="none")
        ).mean()
        dqn_reg = torch.norm(q_loss, 2).mean() * self.q_regularization
        loss = q_loss + dqn_reg

        self.network_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.network_optimizer.step()

        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        new_priorities = torch.abs(target_q - curr_q).detach().view(-1)
        new_priorities = torch.clamp(new_priorities, min=1e-8)
        new_priorities = new_priorities.cpu().numpy().tolist()

        return loss, idxes, new_priorities

    def get_params(self):
        model = deepcopy(self.network)
        model = model.cpu()

        return self.params_to_numpy(self.network)
