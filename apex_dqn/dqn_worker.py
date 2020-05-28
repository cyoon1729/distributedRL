import numpy as np
import ray
import torch
import torch.nn as nn

from common.abstract.worker import ApeXWorker


@ray.remote
class DQNWorker(ApeXWorker):
    def __init__(
        self, worker_id: int, worker_brain: nn.Module, cfg: dict, common_config: dict
    ):
        super().__init__(worker_id, worker_brain, cfg, common_config)
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.eps_greedy = self.cfg["eps_greedy"]
        self.eps_decay = self.cfg["eps_decay"]
        self.gamma = self.cfg["gamma"]

        self.test_state = self.env.reset()

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
        self.eps_greedy = 0
        update_step = 0
        update_interval = self.cfg["param_update_interval"]

        while True:


            if self.receive_new_params():
                update_step = update_step + update_interval

                episode_reward = 0
                state = self.env.reset()
                done = False
                                
                while True:
                    #self.env.render()
                    action = self.select_action(state)
                    transition = self.environment_step(state, action)
                    next_state = transition[-2]
                    done = transition[-1]
                    reward = transition[-3]

                    episode_reward = episode_reward + reward
                    state = next_state

                    if done:
                        break

                print(f"Interim Test {update_step}: {episode_reward}")

            else:
                pass
