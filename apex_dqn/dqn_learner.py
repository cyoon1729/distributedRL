from copy import deepcopy

import ray
import torch
import torch.nn.functional as F

from common.abstract.learner import Learner


@ray.remote(num_gpus=1)
class DQNLearner(Learner):
    def __init__(self, brain, cfg: dict, comm_config: dict):
        super().__init__(brain, cfg, comm_config)
        self.num_step = self.cfg["num_step"]
        self.gamma = self.cfg["gamma"]
        self.tau = self.cfg["tau"]
        self.network = self.brain[0]
        self.network.to(self.device)
        self.target_network = self.brain[1]
        self.target_network.to(self.device)
        self.network_optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.cfg["learning_rate"]
        )
        self.target_optimizer = torch.optim.Adam(
            self.target_network.parameters(), lr=self.cfg["learning_rate"]
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

        curr_q1 = self.network.forward(states).gather(1, actions.unsqueeze(1))
        curr_q2 = self.target_network.forward(states).gather(1, actions.unsqueeze(1))

        bootstrap_q = torch.min(
            torch.max(self.network.forward(next_states), 1)[0],
            torch.max(self.target_network.forward(next_states), 1)[0],
        )

        bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        target_q = rewards + (1 - dones) * self.gamma ** self.num_step * bootstrap_q
        weights = torch.FloatTensor(weights).to(self.device).mean()

        loss1 = weights * F.mse_loss(curr_q1, target_q.detach())
        loss2 = weights * F.mse_loss(curr_q2, target_q.detach())

        self.network_optimizer.zero_grad()
        loss1.backward()
        self.network_optimizer.step()

        self.target_optimizer.zero_grad()
        loss2.backward()
        self.target_optimizer.step()

        step_info = (loss1, loss2)
        new_priorities = torch.abs(target_q - curr_q1).detach().view(-1)
        new_priorities = new_priorities.cpu().numpy().tolist()

        return step_info, idxes, new_priorities

    def get_params(self):
        model = deepcopy(self.network)
        model = model.cpu()
        return self.params_to_numpy(self.network)
