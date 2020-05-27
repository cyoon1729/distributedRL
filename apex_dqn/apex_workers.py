import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from common.abstract.learner import ApeXLearner
from common.abstract.worker import ApeXWorker


@ray.remote
class DQNWorker(ApeXWorker):
    def __init__(
        self, worker_id: int, worker_brain: nn.Module, seed: int, cfg: dict,
    ):
        super().__init__(worker_id, worker_brain, seed, cfg)
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.eps_greedy = self.cfg["eps_greedy"]
        self.eps_decay = self.cfg["eps_decay"]
        self.gamma = self.cfg["gamma"]

        self.test_state = self.env.reset()
        self.test_state = torch.FloatTensor(self.test_state).unsqueeze(0).cpu()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.eps_greedy = self.eps_greedy * self.eps_decay
        if np.random.randn() < self.eps_greedy:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        qvals = self.brain.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        return action

    def environment_step(self, state: np.ndarray, action: np.ndarray) -> tuple:
        next_state, reward, done, _ = self.env.step(action)
        return (state, action, reward, next_state, done)

    def write_log(self):
        print("TODO: include Tensorboard..")

    def test_run(self):
        print(self.brain.forward(self.test_state))
        self.eps_greedy = 0
        episode_reward = 0 
        state = self.env.reset()
        done = False
        while not done:
                action = self.select_action(state)
                transition = self.environment_step(state, action)
                next_state = transition[-2]
                done = transition[-1]
                reward = transition[-3]
                episode_reward += reward

                state = next_state

        return episode_reward


@ray.remote(num_gpus=1)
class DQNLearner(ApeXLearner):
    """The DQN learner implements the enhanced DQN algorithm used in the Ape-X paper, and contains:
     - Dueling architecture
     - Double DQN
     - Prioritized Experience Replay
    """
    def __init__(self, brain, cfg: dict):
        super().__init__(brain, cfg)
        self.num_step = self.cfg["num_step"]
        self.tau = self.cfg["tau"]
        self.network = self.brain[0]
        self.network.to(self.device)
        self.target_network = self.brain[1]
        self.target_network.to(self.device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg["learning_rate"])
        self.target_optimizer = torch.optim.Adam(self.target_network.parameters(), lr=self.cfg["learning_rate"])

        print(f"From Learner: {torch.cuda.is_available()}")
        print(f"From Learner: {ray.get_gpu_ids()}")
        

    def write_log(self):
        print("TODO: incorporate Tensorboard...")

    def learning_step(self, data: tuple):
        states, actions, rewards, next_states, dones, weights, idxes = data
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        curr_q = self.network.forward(states).gather(1, actions.unsqueeze(1))
        bootstrap_q = torch.max(self.target_network.forward(next_states), 1)[0]
        bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        target_q = rewards + ((1 - dones) * self.gamma ** self.num_step) * bootstrap_q
        weights = torch.FloatTensor(weights).to(self.device).mean()

        loss = weights * F.mse_loss(curr_q, target_q.detach())

        self.network_optimizer.zero_grad()
        loss.backward()
        self.network_optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        step_info = loss
        new_priorities = torch.abs(target_q - curr_q).detach().view(-1)
        new_priorities = new_priorities.cpu().numpy().tolist()
        
        return step_info, idxes, new_priorities

    def get_params(self):
        model = deepcopy(self.network)
        return self.params_to_numpy(model)