import numpy as np
import model

def move(state):
    valid_moves = state.get_valid_moves()
    return np.random.choice(valid_moves)