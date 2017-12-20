import numpy as np
import model
from abc import ABCMeta, abstractmethod

class Bot(metaclass=ABCMeta):
    def __init__(self, react_time):
        self.reaction_time = react_time

    @abstractmethod
    def move(self, state):
        pass

    @abstractmethod
    def init_counters(self):
        pass

    @abstractmethod
    def get_eval_count(self):
        pass

    @abstractmethod
    def get_avg_evals(self):
        pass
