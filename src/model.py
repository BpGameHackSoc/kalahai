import numpy as np
from enum import IntEnum
import sys
import copy

class MoveException(Exception):
    pass

class Winner(IntEnum):
    SOUTH = 0
    NORTH = 1
    DRAW = 2
    UNKNOWN = 3

class Side(IntEnum):
    SOUTH = 0
    NORTH = 1
    def opposite(self):
        return self.NORTH if self == self.SOUTH else self.SOUTH
    
class Board(object):
    """Represents a kalah board. Responsible for validating moves and executing them."""
    
    def __init__(self, no_of_holes, no_of_seeds):
        self.no_of_holes = no_of_holes
        self.no_of_initial_seeds = no_of_seeds
        self.buckets = np.array(([no_of_seeds] * no_of_holes + [0]) * 2)
        
    def move(self, index, side):
        # Normal move
        self.__validate_index(index)
        finish_index = self.__normal_move(index)
        
        # Grant another move if finished in store
        if self.__finished_in_store(finish_index, side):
            return side
        
        # Check if opposite side is zero
        self.__capture(finish_index, side)
        
        return side.opposite()
    
    def __capture(self, finish_index, side):
        n = self.buckets.size
        opposite_index = n-finish_index-2
        finished_on_own_side =  finish_index < n // 2 if side == Side.SOUTH else finish_index > n //2
        finished_on_zero = self.buckets[finish_index] == 0
        opposite_side_has_seeds = self.buckets[opposite_index] != 0
        if finished_on_own_side and finished_on_zero and opposite_side_has_seeds:
            store_index = int(n//2)-1 if side == Side.SOUTH else n-1
            seeds_won = self.buckets[opposite_index] + 1
            self.buckets[opposite_index] = 0
            self.buckets[finish_index] = 0
            self.buckets[store_index] = self.buckets[store_index] + seeds_won
    
    def __finished_in_store(self, finish_index, side):
        n = self.buckets.size
        right_index = int(n//2-1) if side == Side.SOUTH else n-1
        return right_index == finish_index
    
    def __normal_move(self, index):
        n = self.buckets.size
        no_of_seeds = self.buckets[index]
        self.buckets[index] = 0
        hanging_seeds = np.concatenate((
            np.zeros(index + 1),
            np.ones(no_of_seeds),
            np.zeros((n - index - no_of_seeds - 1) % n)))
        hanging_seeds = np.sum(hanging_seeds.reshape(hanging_seeds.size // n, n), axis=0)
        self.buckets = (self.buckets + hanging_seeds).astype(int)
        return (index + no_of_seeds) % n
            
    def get_holes(self, side):
        start = 0 if side == Side.SOUTH else self.no_of_holes + 1
        return self.buckets[start:start+self.no_of_holes]
    
    def get_store(self, side):
        size = self.buckets.size
        index = int(size//2-1) if side == Side.SOUTH else size-1
        return self.buckets[index]
    
    def sum_seeds(self, side):
        return self.get_holes(side).sum() + self.get_store(side)
    
    def to_str(self):
        """Calculate matrix resembling a board's visual arrangement."""
        n = self.buckets.size
        half = int(n // 2)
        south = self.buckets[:half]
        north = np.flip(self.buckets[half:], axis=0)
        south = np.append([' '], south).reshape(1, half + 1)
        north = np.append(north, [' ']).reshape(1, half + 1)
        table = np.concatenate((north, south), axis=0)
        strboard = ''
        for row in table:
            for column in row:
                strboard += '| %02s ' % str(column)
            strboard += '|\n'
        return strboard
    
    def __validate_index(self, index):
        """Check if move at index is a valid move."""
        n = self.buckets.size
        is_store = index == int(n // 2) - 1 or index == n - 1
        if is_store:
            raise ValueError('The following index is a store: ' + str(index))  
        if self.buckets[index] == 0:
            raise ValueError('The indexed bucket is empty: ' + str(index))        

            

class State(object):
    """Acts as interface between board actions and players. Stores board and player states."""
    
    def __init__(self, no_of_holes=6, no_of_seeds=4):       
        self.set_starting_position(no_of_holes, no_of_seeds)
        self.used_pie_rule = False
        self.players = np.array([Side.SOUTH, Side.NORTH])
        
    def set_starting_position(self, no_of_holes=6, no_of_seeds=4):
        self.current_player = Side.SOUTH
        self.board = Board(no_of_holes, no_of_seeds)
        self.min_seeds_to_win = no_of_holes * no_of_seeds
              
    def is_game_over(self):      
        return self.get_winner() != Winner.UNKNOWN
        
    def get_winner(self):
        board_size = int(self.board.buckets.size // 2)
        south_validation = np.all(self.board.get_holes(Side.SOUTH) == 0)
        north_validation = np.all(self.board.get_holes(Side.NORTH) == 0)
        store_validation_s = self.board.get_store(Side.SOUTH) > self.min_seeds_to_win
        store_validation_n = self.board.get_store(Side.NORTH) > self.min_seeds_to_win
        game_finished = south_validation or north_validation or store_validation_s or store_validation_n
        if not (game_finished):
            return Winner.UNKNOWN
        
        south_seeds, north_seeds = self.get_seed_sum()
        dif = south_seeds - north_seeds
        
        if dif > 0:
            return Winner.SOUTH 
        elif dif < 0:
            return Winner.NORTH
        else:
            return Winner.DRAW
        
    def get_seed_sum(self):
        return self.board.sum_seeds(Side.SOUTH), self.board.sum_seeds(Side.NORTH)
    
    def move(self, index):
        if self.is_game_over():
            raise MoveException("Can't move because game is over.")
        index = self.__validate_index(index, self.current_player)
        self.current_player = self.board.move(index, self.current_player) 

    def use_pie_rule(self):
        if not self.is_pie_rule_on:
            raise MoveException("Can't use pie rule, this game is played without it.")
        if self.moves_made == 1:
            self.__players_change_sides()
            self.used_pie_rule = True
        else:
            raise MoveException("The game is not right after the first move, so pie rule is not applicable.")
        
    def get_valid_moves(self):
        empty_holes = self.board.get_holes(self.current_player) != 0
        return np.argwhere(empty_holes).flatten()
        
    def show(self):
        print(self.board.to_str())
        
    def __change_current_player(self):
        self.current_player = self.current_player.opposite()
        
        
    def __validate_index(self, index, side):
        n = self.board.buckets.size
        half = int(n // 2)
        out_of_bounds = index < 0 or index >= half
        if out_of_bounds:
            raise ValueError('The following index is out of bounds: ' + str(index))
        return index if side == Side.SOUTH else index + half

    def __players_change_sides(self):
        self.players = np.flip(self.players, axis=0)
    
class Game(object):
    """Governs the game, handling start and end. Also stores history"""
    def __init__(self, no_of_holes=6, no_of_seeds=4, pie_rule=False, print_results=True):
        self.history = []
        self.current_state = State(no_of_holes, no_of_seeds)
        self.print_results = print_results
        self.winner = Winner.UNKNOWN
        self.moves_made = 0
        self.is_pie_rule_on = pie_rule

        
    def load(self):
        pass
    
    def save(self):
        pass

    def get_valid_moves(self):
        pseudo_valid_moves = self.current_state.get_valid_moves()
        can_swap = self.is_pie_rule_on and self.moves_made == 1
        valid_moves = np.append(pseudo_valid_moves, -1) if can_swap else pseudo_valid_moves
        return valid_moves
    
    def apply_move(self, index):
        self.history.append(copy.deepcopy(self.current_state))
        if index == -1 and not self.is_pie_rule_on:
            print ('You cannot use pie rule now.')
            return
        self.current_state.move(index)
        self.moves_made += 1
        if (self.game_over()):
            self.winner = self.current_state.get_winner()
            if self.print_results:
                self.__announce_winner()

    def undo_last_move(self):
        self.current_state = self.history.pop()
        self.winner = Winner.UNKNOWN
        self.moves_made -= 1
         
    def game_over(self):
        return self.current_state.is_game_over()
    
    def __announce_winner(self):
        self.current_state.show()
        south_seeds, north_seeds = self.current_state.get_seed_sum()
        print('Game over. Winner is: ' + str(self.winner), end=' ')
        print('(' + str(south_seeds) + ' - ' + str(north_seeds) + ')')



        