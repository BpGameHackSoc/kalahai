import numpy as np
import model
from . import Bot

class RandomAgent(Bot.Bot):
    def __init__(self,**kwargs):
        react_time = kwargs["react_time"]
        super().__init__(react_time)
        self.name = 'RandomAgent'
        self.init_counters()

    def move(self, state):
        valid_moves = state.get_valid_moves()
        self.move_count += 1
        return np.random.choice(valid_moves)

    def init_counters(self):
        self.move_count = 0

    def get_eval_count(self):
        return self.move_count

    def get_avg_evals(self):
        return 1
