import numpy as np
import model
from model import Side
from copy import deepcopy
from . import Bot


class AlphaBeta(Bot.Bot):
    def __init__(self, depth, react_time=1):
        if react_time is not None:
            raise NotImplementedError("React time for AplhaBeta agent is not implemented")
        else:
            super().__init__(react_time)
        self.search = AlphaBetaSearch(depth)
        self.name = 'AlphaBeta'
        self.init_counters()

    def init_counters(self):
        self.move_num = 0
        self.nodes_evaled_agg = 0

    def move(self, state):
        score, move = self.search.run(state)
        self.move_num +=1
        self.nodes_evaled_agg += self.search.nodes_evaled
        return move

    def get_statistics_str(self):
        return "%d number of nodes were evaluated with depth %d".format(self.search.nodes_evaled,self.search.INITIAL_DEPTH)

    def get_eval_count(self):
        return self.search.nodes_evaled

    def get_avg_evals(self):
        return self.nodes_evaled_agg/self.move_num



class AlphaBetaSearch():
    def __init__(self, depth, show_evaled_nodes=True):
        self.INITIAL_DEPTH = depth 
        self.eval_nodes = show_evaled_nodes

    def run(self, state):
        is_maximising = state.current_player == Side.SOUTH
        self.nodes_evaled= 0
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
        if self.eval_nodes:
            self.nodes_evaled+=1
        south = state.board.get_store(Side.SOUTH)
        north = state.board.get_store(Side.NORTH)
        return south - north