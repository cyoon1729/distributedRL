import numpy as np
import torch
import torch.nn.functional as F
from models import ConvDuelingDQN, DuelingDQN
from common.utils.buffer import PrioritizedReplayBuffer
from common.utils.utils import create_env
import gym

env = create_env("PongNoFrameskip-v4", atari=True, max_episode_steps=None)
network = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).cuda()
target_network = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).cuda()
network_optimizer = torch.optim.Adam(
    network.parameters(), lr=3e-4
)
target_optimizer = torch.optim.Adam(
    target_network.parameters(), lr=3e-4
)
device = torch.device("cuda")

eps_greedy = 0.20
eps_decay = 0.95\
gamma = 0.9
priority_alpha = 0.6
priority_beta = 0.4
priority_beta_end = 1.0 
max_num_updates = 30000
priority_beta_increment = (priority_beta_end - priority_beta) / max_num_updates
replay_buffer = PrioritizedReplayBuffer(50000, priority_alpha)


def learning_step(data: tuple):
    states, actions, rewards, next_states, dones, weights, idxes = data
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device).view(-1, 1)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device).view(-1, 1)

    curr_q1 = network.forward(states).gather(1, actions.unsqueeze(1))
    curr_q2 = target_network.forward(states).gather(1, actions.unsqueeze(1))

    bootstrap_q = torch.min(
        torch.max(network.forward(next_states), 1)[0],
        torch.max(target_network.forward(next_states), 1)[0],
    )

    bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
    target_q = rewards + (1 - dones) * gamma * bootstrap_q
    weights = torch.FloatTensor(weights).to(device).mean()

    loss1 = weights * F.mse_loss(curr_q1, target_q.detach())
    loss2 = weights * F.mse_loss(curr_q2, target_q.detach())

    network_optimizer.zero_grad()
    loss1.backward()
    network_optimizer.step()

    target_optimizer.zero_grad()
    loss2.backward()
    target_optimizer.step()

    step_info = (loss1, loss2)
    new_priorities = torch.abs(target_q - curr_q1).detach().view(-1)
    new_priorities = torch.clamp(new_priorities, min=1e-8)
    new_priorities = new_priorities.cpu().numpy().tolist()

    return step_info, idxes, new_priorities



for episode in range(50000):
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        # action selection
        env.render()
        eps_greedy *= eps_decay
        if np.random.randn() < eps_greedy:
            action = env.action_space.sample()
        else:
            qvals = network.forward(
                torch.FloatTensor(state).unsqueeze(0).cuda()
            )
            action = np.argmax(qvals.cpu().detach().numpy())
        
        # environment step
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        replay_buffer.add(state, action, reward, next_state, done)
        
        # learning_step
        if len(replay_buffer) > 128:
            batch = replay_buffer.sample(128, priority_beta)
            priority_beta += priority_beta_increment
            step_info, idxes, new_priorities = learning_step(batch)
            replay_buffer.update_priorities(idxes, new_priorities)
        
        state = next_state
    
    print(f"Episode {episode}: {episode_reward}")