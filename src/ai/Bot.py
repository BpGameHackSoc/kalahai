import numpy as np
import model

class Bot(object):
    def __init__(self, react_time):
        self.reaction_time = react_time
        
    def move(self, state):
        print('No move function was defined.')
        return -1