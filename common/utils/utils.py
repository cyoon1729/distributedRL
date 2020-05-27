from typing import Deque, Union

import gym
import numpy as np
import torch
import yaml
from gym.wrappers import TimeLimit

from common.utils.baseline_wrappers import (make_atari, wrap_deepmind,
                                            wrap_pytorch)


def read_config(config_path: str):
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    if cfg["atari"] is True:
        env = make_atari(cfg["env_name"])
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
    else:
        env = gym.make(cfg["env_name"])
    del env

    cfg["obs_dim"] = env.observation_space.shape
    cfg["action_dim"] = env.action_space.n

    comm_cfg = {}
    comm_cfg["pubsub_port"] = cfg["pubsub_port"]
    comm_cfg["repreq_port"] = cfg["repreq_port"]
    comm_cfg["pullpush_port"] = cfg["pullpush_port"]

    return cfg, comm_cfg


def create_env(env_name: str, atari: bool, max_episode_steps: Union[int, None]):
    if atari:
        env = make_atari(env_name)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
    else:
        env = gym.make(env_name)

    if max_episode_steps:
        env = TimeLimit(env, max_episode_steps=1000)

    return env


def preprocess_nstep(transition_buffer: Deque, gamma=0.99):
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


def params_to_numpy(model):
    params = []
    state_dict = model.cpu().state_dict()
    for param in list(state_dict):
        params.append(state_dict[param])
    return params
