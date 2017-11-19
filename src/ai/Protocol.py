from . import RandomAgent
import model
import numpy as np

def run_competition(str_bot1, str_bot2, react_time=1, buckets=6, seeds=4,
                    pie_rule=True, print_states=False, print_games=False, no_of_games=1):
    bot1 = get_agent(str_bot1)
    bot2 = get_agent(str_bot2)
    wins = np.zeros(2)

    for i in range(no_of_games):
        game = model.Game(buckets, seeds, pie_rule, print_results=False) 
        while not game.game_over():
            current_player = game.current_state.current_player
            bot_index = np.argwhere(game.players == current_player)
            current_bot = bot1 if bot_index == 0 else bot2
            move_index = current_bot.move(game.current_state)
            game.apply_move(move_index)
            if print_states:
                game.current_state.show()
        if game.winner == model.Winner.DRAW:
            wins += 0.5
        else:
            side_won = model.Side.SOUTH if game.winner == model.Winner.SOUTH else model.Side.NORTH
            bot_index = np.argwhere(game.players == side_won)
            wins[bot_index] += 1

    print ("1st agent scored: " + str(wins[0]))
    print ("2nd agent scored: " + str(wins[1]))



def get_agent(str_bot):
    if str_bot == 'random':
        return RandomAgent