import numpy as np
import gym
import yaml
from common.utils.baseline_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def read_config(config_path: str):
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    if cfg["atari"] is True:
        env = make_atari(cfg["env"])
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
    else:
        env = gym.make(cfg["env"])

    cfg["obs_dim"] = env.observation_space.shape
    cfg["action_dim"] = env.action_space.n
    del env

    return cfg


def preprocess_nstep(transition_buffer, gamma=0.99):
    transition_buffer = list(transition_buffer)
    discounted_reward = 0
    for transition in reversed(transition_buffer[:-1]):
        _, _, reward, _ = transition
        discounted_reward += gamma * reward

    state, action, _, _, _ = transition_buffer[0]
    last_state, _, _, _, _ = transition_buffer[-1]
    _, _, _, _, done = transition_buffer[-1]

    return (
        np.array(state),
        action,
        np.array([discounted_reward]),
        np.array(last_state),
        np.array(done),
    )
