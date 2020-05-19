import sys

sys.path.append("../")

import ray
import torch
from worker_learner import DQNWorker, DQNLearner
from common.utils.utils import read_config, preprocess_nstep
from model import ConvDuelingDQN, ConvDQN
from common.utils.baseline_wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.utils.buffer_helper import BufferHelper
from common.utils.buffer import PrioritizedReplayBuffer
from datetime import datetime

ray.init()

config_pth = "config.yml"
cfg = read_config("config.yml")
dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
tdqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
# dqn = ConvDQN(cfg['obs_dim'], cfg['action_dim'])
# print(cfg['obs_dim'], cfg['action_dim'])
# if cfg['atari'] is True:
#     env = make_atari(cfg['env'])
#     env = wrap_deepmind(env)
#     env = wrap_pytorch(env)

worker = DQNWorker.remote(1, dqn, cfg["env"], 1, cfg)
brains = (dqn, tdqn)
learner = DQNLearner.remote(brains, cfg)


buffer_max_size = cfg["buffer_max_size"]
priority_beta = cfg["priority_beta"]
priority_alpha = cfg["priority_alpha"]
priority_beta_start = cfg["priority_beta_start"]
priority_beta_end = cfg["priority_beta_end"]
num_train_steps = cfg["max_num_updates"]
priority_beta_increment = (priority_beta_end - priority_beta_start) / num_train_steps
priority_alpha = cfg["priority_alpha"]
batch_size = cfg["batch_size"]

global_buffer = BufferHelper.remote(
    PrioritizedReplayBuffer(buffer_max_size, priority_alpha)
)

if __name__ == "__main__":
    worker.collect_data.remote()
    wbuffer = ray.get(worker.get_buffer.remote())
    global_buffer.incorporate_new_data.remote(wbuffer)
    batch = ray.get(global_buffer.sample_data.remote(batch_size, priority_beta))
    learner.learning_step.remote(batch)
    # print(batch)
    print("done")
