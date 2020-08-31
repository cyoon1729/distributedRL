from abc import ABC, abstractmethod


class Architecture(ABC):
    """Abstract Architecture used for all distributed architectures"""

    def __init__(self, worker_cls: type, learner_cls: type, cfg: dict):
        """Initialize"""
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.num_learners = self.cfg["num_learners"]
        self.worker_cls = worker_cls
        self.learner_cls = learner_cls

    @abstractmethod
    def spawn(self):
        """Spawn distributed components"""
        pass

    @abstractmethod
    def train(self):
        """Run main training loop"""
        pass
