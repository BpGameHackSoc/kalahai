from . import RandomAgent
from . import AlphaBeta
import model
import numpy as np

from importlib import reload
reload(AlphaBeta)
reload(RandomAgent)
reload(model)

bot1,bot2 = None,None
def run_competition(str_bot1, str_bot2, react_time=1, buckets=6, seeds=4,
                    pie_rule=True, print_states=False, print_games=False, no_of_games=1,
                    depth1=None, depth2=None,):
    global bot1,bot2
    bot1 = get_agent(str_bot1, depth1,react_time=None)
    bot2 = get_agent(str_bot2, depth2,react_time=None)
    wins = np.zeros(2)

    bot1.avg_ne_across_g,bot2.avg_ne_across_g = (0,0)

    for i in range(no_of_games):
        #statistics
        move_count = 0
        bot1.init_counters()
        bot2.init_counters()

        game = model.Game(buckets, seeds, pie_rule, print_results=False)
        while not game.game_over():  # Calculate one move per loop
            current_player = game.current_state.current_player
            bot_index = np.argwhere(game.current_state.players == current_player)
            current_bot = bot1 if bot_index == 0 else bot2
            move_index = current_bot.move(game.current_state)
            bot_index = np.argwhere(game.current_state.players == game.current_state.current_player)
            game.apply_move(move_index)
            if print_states:
                print(game.current_state.players)
                game.current_state.show()
            #statictics
            move_count +=1
        if game.winner == model.Winner.DRAW:
            wins += 0.5
        else:
            side_won = model.Side.SOUTH if game.winner == model.Winner.SOUTH else model.Side.NORTH
            bot_index = np.argwhere(game.current_state.players == side_won)
            wins[bot_index] += 1

        bot1.avg_ne_across_g += bot1.get_avg_evals()
        bot2.avg_ne_across_g += bot2.get_avg_evals()

    print ("1st agent was {} and scored: {} with {} number of average evaluations across games".format(bot1.name,str(wins[0]),bot1.avg_ne_across_g/no_of_games))
    print("2st agent was {} and scored: {} with {} number of average evaluations across games".format(bot2.name,
                                                                                                      str(wins[1]),
                                                                                                      bot2.avg_ne_across_g/no_of_games))

    game.undo_last_move()
    game.current_state.show()


def get_agent(str_bot, depth=None,react_time=None):
    if str_bot == 'random':
        return RandomAgent.RandomAgent()
    if str_bot == 'alpha_beta':
        return AlphaBeta.AlphaBeta(depth,react_time)

