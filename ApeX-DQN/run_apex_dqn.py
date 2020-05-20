import ray
import torch 

from common.utils.utils import read_config
from architectures.apex import ApeX
from worker_learner import DQNWorker, DQNLearner
from model import ConvDuelingDQN, ConvDQN, DuelingDQN

ray.init(num_gpus=1)


if __name__ == "__main__":
    
    print(f"From main: {torch.cuda.is_available()}")
    print(f"From main: {ray.get_gpu_ids()}") 

    cfg = read_config("config.yml")
    device = torch.device(cfg["learner_device"])
    dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"]) #.to(device)
    target_dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"]) # .to(device)
    brain = (dqn, target_dqn)

    ApeXDQN = ApeX(cfg)
    ApeXDQN.spawn(DQNWorker, DQNLearner, brain)
    ApeXDQN.train2()