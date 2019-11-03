from time import sleep
from datetime import datetime
import os
import torch
from copy import deepcopy
import ray
from shutil import copyfile

from actor import Actor
from learner import Learner
from myutils import ParameterServer, Buffer, PrioritizedReplayBuffer, read_config, to_n_step


ray.init()
if __name__ == "__main__":
    
    # read config 
    config = "config.yml"
    config_path = None
    if isinstance(config, str):
        config_path = config
        config = read_config(config)
    elif not isinstance(config, dict):
        raise ValueError("Config should be either string or dict")
    
    # Environment info
    network_type = config['network_type']
    env_id = config['env']
    random_seed = config['random_seed']
    obs_dim = config['obs_dim']
    action_dim = config['action_dim']

    # Training parameters 
    n_agents = config['num_agents']
    unroll_cnt = config['unroll_steps']
    num_train_step = config['num_train_steps']

    # Buffer
    use_per = config['use_per']
    priority_alpha = config['priority_alpha']
    batch_size = config['batch_size']
    buffer_max_size = config['buffer_max_size']

    # Create directory for experiment
    experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/config.yml")

    #############
    #### RUN ####
    #############

    # parameter server
    param_server = ParameterServer.remote()

    # buffer
    if use_per:
        buffer = PrioritizedReplayBuffer.remote(size=buffer_max_size, alpha=priority_alpha)
    else:
        buffer = Buffer.remote(max_size=buffer_max_size)

    # learner:
    learner = Learner.remote(config, buffer, param_server, experiment_dir)
    param_list = ray.get(learner.return_param_list.remote())
    param_server.define_param_list.remote(param_list)
    learner.update_param_server.remote()

    # actors 
    actors = [Actor.remote(i, config, experiment_dir, buffer, param_server)
              for i in range(n_agents)]

    # processes
    procs = []
    procs.append(learner)    
    for actor in actors:
        procs.append(actor)

    print("run")

    ray.wait([proc.run.remote() for proc in procs])

    ray.timeline()

    print("End")