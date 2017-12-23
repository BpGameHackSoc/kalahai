import numpy as np
from . import AlphaBeta
from model import Side

class AlphaBetaOwnSeeds(AlphaBeta.AlphaBeta):

    def evaluate(self,state):
        south_holes = state.board.get_holes(Side.SOUTH)
        north_holes = state.board.get_holes(Side.NORTH)
        south_store = state.board.get_store(Side.SOUTH)
        north_store = state.board.get_store(Side.NORTH)
        val = (self.keepable_seeds(south_holes) -self.keepable_seeds(north_holes)) *0.25
        val += south_store - north_store
        return val

    def keepable_seeds(self,buckets):
        size = len(buckets)
        clipper = np.array(range(size,0,-10))-np.ones(size)
        return np.sum(np.clip(buckets,None,clipper))

    def move(self, state):
        return super().move(state)
