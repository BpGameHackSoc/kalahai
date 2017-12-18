import numpy as np
import model
from model import Side
from copy import deepcopy


class AlphaBeta():
    def __init__(self, depth, react_time=1):
        self.search = AlphaBetaSearch(depth)
        self.name = 'AlphaBeta'

    def move(self, state):
        score, move = self.search.run(state)
        return move


class AlphaBetaSearch():
    def __init__(self, depth):
        self.INITIAL_DEPTH = depth 

    def run(self, state):
        is_maximising = state.current_player == Side.SOUTH
        if is_maximising:
            return self.__maximize(state, self.INITIAL_DEPTH, -np.inf, np.inf)
        else:
            return self.__minimize(state, self.INITIAL_DEPTH, -np.inf, np.inf)

    def __maximize(self, state, depth, a, b):
        if depth == 0 or state.is_game_over():
            return self.__evalute(state), None
        valid_moves = state.get_valid_moves()
        score = -np.inf
        best_move = None
        for move in valid_moves:
            child_state = self.__get_child(state, move)
            score = max(score, self.__minimize(child_state, depth-1, a, b)[0])
            if score > a:
                a = score
                best_move = move
            if b <= a:
                break
        return score, best_move

    def __minimize(self, state, depth, a, b):

        if depth == 0 or state.is_game_over():
            return self.__evalute(state), None
        valid_moves = state.get_valid_moves()
        score = np.inf
        best_move = None
        for move in valid_moves:
            child_state = self.__get_child(state, move)
            score = min(score, self.__maximize(child_state, depth-1, a, b)[0])
            if score < b:
                b = score
                best_move = move
            if b <= a:
                break
        return score, best_move

    def __get_child(self, state, move_index):
        child_state = deepcopy(state)
        child_state.move(move_index)
        return child_state

    def __evalute(self, state):
        south = state.board.get_store(Side.SOUTH)
        north = state.board.get_store(Side.NORTH)
        return south - north