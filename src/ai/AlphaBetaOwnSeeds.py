import numpy as np
from . import AlphaBeta
from model import Side

class AlphaBetaOwnSeeds(AlphaBeta.AlphaBeta):

    def __evaluate(self,state):
        south_holes = state.board.get_holes(Side.SOUTH)
        north_holes = state.board.get_holes(Side.NORTH)
        south_store = state.board.get_store(Side.SOUTH)
        north_store = state.board.get_store(Side.NORTH)
        val = self.keepable_seeds(south_holes) -self.keepable_seeds(north_holes)
        val += south_store - north_store
        return val

    def keepable_seeds(self,buckets):
        return np.sum(np.clip(buckets,range(len(buckets),0,-1)))

    def move(self, state):
        return super().move(state)
