from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class AbstractEstimator(metaclass=ABCMeta):
    def __init__(self, model, learning_rate=0.001):
        assert isinstance(model, nn.Module)
        self.model_ = model
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), learning_rate)

    @abstractmethod
    def update(self, rewards, log_probabilities):
        pass

    @abstractmethod
    def get_action(self, state):
        pass
