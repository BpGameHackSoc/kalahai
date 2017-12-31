import numpy as np
import model
from model import Side
from . import Bot

from copy import deepcopy,copy
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten,Dropout
from keras.utils import to_categorical,plot_model
from keras.models import load_model

MODEL_PATH = 'res/networks/qlearning_sample.h5'

class QLearningAgent(Bot.Bot):
    def __init__(self, neural_network_path, train_mode=False, T=1):
        self.model = load_model(neural_network_path)
        self.name = 'QLearningAgent'
        self.train_mode = train_mode
        self.T = T

    def move(self, state):
        inputs = self.__get_inputs(state)
        prediction = model.predict(inputs)
        mask = np.isin(np.arange(prediction.size), state.get_valid_moves())
        best_value = self.__choose_value(prediction[mask])
        move_index = np.arange(prediction.size)[prediction == best_value][0]
        return move_index

    # In train mode, we include some exploration, while in normal mode,
    # we simply choose the node with the highest win-probability
    def __choose_value(self, values):
        if self.train_mode:
            probabilities = self.__softmax(values)
            return np.random.choice(values, p=probabilities)
        else:
            return max(values)

    def __softmax(self, x):
        scoreMatExp = np.exp(np.asarray(x/self.T))
        return (scoreMatExp) / (scoreMatExp.sum(0))


class QMemory:
    def __init__(self, memory_size, batch_size):
        self.memory_x = []
        self.memory_y = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        
    def add(self, x, y):
        if len(self.memory) > self.memory_size:
            del self.memory_x[0]
            del self.memory_y[0]
        self.memory_x.append(x)
        self.memory_y.append(y)
        
    def random_choice(self):
        n = min(self.batch_size, len(self.memory_x))
        all_index = np.arange(len(self.memory_x))
        indexes = np.random_choice(all_index, n, replace=False)
        return np.array(self.memory_x)[indexes], np.array(self.memory_y)[indexes]

class QGroup:
    def __init__(self, state, move):
        self.state = state
        self.move = move
        self.next_state = deepcopy(state).move(move)



class QLearningTrainer:
    def __init__(self, epsilon=0.66, epsilon_decay=0.999, memory_size=200,
                 epochs=10, batch_size=50, input_size=14, output_size=7):
        self.memory = QMemory(memory_size, batch_size)
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.model = self.__init_model(input_size, output_size)

    def __init_model(self, n_in, n_out):
        model = Sequential()
        model.add(Dense(32, input_shape=(n_in, ), activation='relu'))  
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_out, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        plot_model(model, to_file='res/model.png', show_shapes=True)
        return model

    def train_batch(self, T):
        self.__play_games(T)
        train_x, train_y = np.array(self.memory.random_choice())
        train_x = train_x.reshape(1, -1).squeeze(0)
        train_x = train_x.reshape(1, -1).squeeze(0)
        # TEST DATA BEFORE RUNNING TRAINING!!!!
        print('SHAPES')
        print(train_x.shape)
        print(train_y.shape)
        print('TRAIN_X')
        print(train_x)
        print('TRAIN_Y')
        print(train_y)

        # model.train_on_batch(train_x, train_y)

    def train(self):
        T = 1000
        decay = math.pow(1/T, 1/(self.epochs-1))
        for i in range(self.epochs):
            print('Epoch ' + str(i) + '...')
            self.train_batch(T)
            T *= decay



    def __play_games(self, T):
        for i in range(self.batch_size):
            game = self.__play_one_game([7, 4, True, T])
            train_x = self.__game_to_inputs(game)
            train_y = self.__game_to_outputs(game)
            self.memory.add(train_x, train_y)

    def __game_to_inputs(self, game):
        """

        :param game:
        :return:
        """
        # Converting a game into inputs for the neural net (states)
        # 1. Each state is one input (game.history)
        # 2. A state means the board, current_player, and pie_rule_available
        # 3. Output shape should look like (no_of_states, no_of_inputs)
        states = []
        for state, move in game.history():
            current_player = np.array([(state.current_player == Side.SOUTH) * 1])
            board = state.board.buckets
            states.append(np.concatenate((current_player, board)))
        return np.array(states)

    def __change_q_values(self, original, indexes, target):
        for count, index in enumerate(indexes):
            original[count][index] = target
        return original

    def __determine_target(self, game):
        winner = game.get_winner_side()
        if winner == Side.SOUTH:
            return 1
        elif winner == Side.NORTH:
            return -1
        elif winner == Side.DRAW:
            retrun 0
        else:
            raise ValueError('There is no winner just yet: ' + str(winner))  

    def __game_to_output(self, game):
        # Converting a game into outputs for the neural net (actions)
        # 1. For each state-action pair the NN outputs the estimated outcome of the game (for invalid moves too)
        # 2. For each state in history, get estimated outcomes for each action (NN)
        # 3. In each state, those actions should be updated that have been chosen
        #    ---> Update those to the outcome of the game (-1, 0, 1)
        #    ---> Leave the rest of the outputs as they are
        # 4. Return the new outcome
        # 5. Shape should look like (no_of_states, no_of_all_moves)

        inputs = self.__game_to_inputs(game)
        predictions = model.predict(inputs)
        history = np.array(game.history)
        target = self.__determine_target(game)
        moves = np.hstack(history[:,1]).reshape(-1, len(history[0,1]))
        # moves : should be a numpy array of moves -> e.g. [0, 4, 2, 5]
        # NOTE
        # Probably there's a mess around local/global moves
        # (north move can be 3, but globally its 3+7=10)
        train_y = self.__change_q_values(predictions, moves, target)
        return train_y

    def __play_one_game(self, settings):
        """Make agent play a game against itself and return the history of the game.

        :param settings: parameters of the game and the agent
        :return: move sequence of the game, the history attribute of game_obj
        """
        buckets, seeds, pie_rule, temperature = settings
        game_obj = model.Game(buckets, seeds, pie_rule, print_results=False)
        bot1 = QLearningAgent(MODEL_PATH, train_mode=True, T=temperature)
        bot2 = QLearningAgent(MODEL_PATH, train_mode=True, T=temperature)
        players = [bot1, bot2]
        while not game_obj.game_over():
            current_player_index = game_obj.get_current_player_index()
            current_bot = players[current_player_index]
            move_index = current_bot.move(game_obj.current_state)
            game_obj.apply_move(move_index)
        return game_obj.history

