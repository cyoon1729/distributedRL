from model import ConvDuelingDQN, DuelingDQN
from common.utils.buffer import PrioritizedReplayBuffer
import gym

env = gym.make("PongNoFrameskip-v4")
dqn = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).cuda()
tdqn = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).cuda()


def select_action(self, state: np.ndarray) -> np.ndarray:
    eps_greedy = eps_greedy * self.eps_decay
    if np.random.randn() < eps_greedy:
        return env.action_space.sample()

    state = torch.FloatTensor(state).cuda()
    state = state.unsqueeze(0)
    qvals = dqn.forward(state)
    action = np.argmax(qvals.cpu().detach().numpy())
    return action

if __name__ == "__main__":

    for episode in range(50000):
        done = False
        episode_reward = 0
        state = env.reset()

        while not done:
                