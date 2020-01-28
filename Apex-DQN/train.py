from time import sleep
from datetime import datetime
import os
import torch
from copy import deepcopy
import ray
from shutil import copyfile

from actor import Actor
from learner import Learner
from myutils import *


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
    num_train_steps = config['num_train_steps']

    # Buffer
    use_per = config['use_per']
    priority_alpha = config['priority_alpha']
    priority_beta_start = config['priority_beta_start']
    priority_beta_end = config['priority_beta_end']
    priority_beta = priority_beta_start
    priority_beta_increment = ((priority_beta_end - priority_beta_start)
                               / num_train_steps)
    batch_size = config['batch_size']
    buffer_max_size = config['buffer_max_size']

    # Create directory for experiment
    timenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    experiment_dir = "{0}/{1}-{2}-{3}".format(
        config['results_path'],
        config['env'],
        config['model'],
        timenow        
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, "{0}/config.yml".format(experiment_dir))

    #############
    #### RUN ####
    #############

    # parameter server
    # parameter server
    param_server = ParameterServer.remote() 
    
    # buffer
    if use_per:
        buffer = PrioritizedReplayBuffer.remote(
            size=buffer_max_size,
            alpha=priority_alpha)

    else:
        buffer = Buffer.remote(max_size=buffer_max_size)

    # learner:
    learner = Learner.remote(config, experiment_dir)
    param_list = ray.get(learner.return_param_list.remote())
    param_server.define_param_list.remote(param_list)

    # actors 
    actors = [Actor.remote(i, config, experiment_dir)
                for i in range(n_agents)]
   
    # fill up the actor buffers, then add them to global buffer
    _, _ = ray.wait([actor.start.remote() for actor in actors],
                     num_returns=n_agents)
    
    for actor in actors:
        buffer.incorporate_buffer.remote(
            ray.get(actor.return_buffer.remote()))
    
    sleep(3)
 
    print('Initial buffer filled')
    print("Run")
    
    actor_buffers = {}
    for actor in actors:
        actor_buffers[actor.fill_buffer.remote()] = actor

    update_step = ray.get(param_server.get_update_step.remote())

    while update_step < num_train_steps:
        # gather actor experience to centralized buffer
        ready_actor_list, _ = ray.wait(list(actor_buffers))
        ready_actor_id = ready_actor_list[0]
        actor = actor_buffers.pop(ready_actor_id)
        buffer.incorporate_buffer.remote(
            ray.get(actor.return_buffer.remote())
        )

        # learner step
        batch, indices, weights = ray.get(
            buffer.sample.remote(batch_size, priority_beta)
        )
        step_info, indices, new_priorities = ray.get(
            learner.learning_step.remote(batch, indices, weights)
        )
        learner.logger.remote(step_info)
        if use_per:
            buffer.update_priorities.remote(indices, new_priorities)
            priority_beta += priority_beta_increment
        
        # sync learner policy with global
        learner_params = ray.get(
            learner.return_numpy_policy.remote()
        )
        param_server.update_params.remote(learner_params)

        # sync actor policy with global policy
        actor.set_params.remote(
            ray.get(param_server.return_params.remote())
        )
        actor_buffers[actor.fill_buffer.remote()] = actor
        update_step = ray.get(
            param_server.get_update_step.remote()
        )
        
    # save data and end
    actors.append(learner)
    ray.wait([process.final_save.remote()
             for process in actors], num_returns=n_agents+1)

    ray.timeline()

    print("End")