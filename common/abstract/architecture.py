from abc import ABC, abstractmethod
import numpy as np

from learner import Learner
from worker import Worker
from distributedRL.

class Architecture(ABC):
    """Abstract Architecture used for all distributed architectures
    
    Attributes:
        self.num_workers =
    
    """

    def __init__(self, cfg: dict):

        self.cfg = cfg
        self.num_workers = cfg['num_workers'] 
        self.num_learners = cfg['num_learners']

    
    def spawn(self, worker: type, learner: type, param_server: ParameterServer, centralized_buffer: Buffer):
        
        self.workers

    

    @abstractmethod
    def train(self):
        pass  


