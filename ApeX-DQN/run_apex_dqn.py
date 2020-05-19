import ray

from common.utils.utils import read_config
from architectures.apex import ApeX
from worker_learner import DQNWorker, DQNLearner
from model import ConvDuelingDQN, ConvDQN

ray.init()


if __name__ == "__main__":
    cfg = read_config("config.yml")

    dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
    target_dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
    brain = (dqn, target_dqn)

    ApeXDQN = ApeX(cfg)
    ApeXDQN.spawn(DQNWorker, DQNLearner, brain)
    ApeXDQN.train()