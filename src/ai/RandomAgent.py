import numpy as np
import model

class RandomAgent():
    def __init__(self):
        self.react_time = 1
        self.name = 'RandomAgent'

    def move(self, state):
        valid_moves = state.get_valid_moves()
        return np.random.choice(valid_moves)
    