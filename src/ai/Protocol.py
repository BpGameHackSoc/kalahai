import importlib
import model
import numpy as np
import sys
import os

from importlib import reload

from . import Bot
reload(Bot)
# reload(RandomAgent)
reload(model)
# reload(AlphaBetaOwnSeeds)

bot1,bot2 = None,None
def run_competition(str_bot1, str_bot2, react_time=1, buckets=6, seeds=4,
                    pie_rule=True, print_states=False, print_games=False, no_of_games=1, change_sides=True,
                    depth1=None, depth2=None,):
    global bot1,bot2
    bot1 = get_agent(str_bot1, depth1,react_time=None)
    bot2 = get_agent(str_bot2, depth2,react_time=None)
    wins = np.zeros(2)

    bot1.avg_ne_across_g,bot2.avg_ne_across_g = (0,0)
    players = [bot1, bot2]
    player2_starts = False

    for i in range(no_of_games):
        #statistics
        move_count = 0
        bot1.init_counters()
        bot2.init_counters()

        game = model.Game(buckets, seeds, pie_rule, print_results=False,player2_starts=player2_starts)
        while not game.game_over():  # Calculate one move per loop
            current_player_index = game.get_current_player_index()
            current_bot = players[current_player_index]
            move_index = current_bot.move(game.current_state)
            game.apply_move(move_index)
            if print_states:
                print(game.current_state.players)
                game.current_state.show()
        winner_side = game.get_winner_side()
        if winner_side.is_draw():
            wins += 0.5
        else:
            winner_index=  game.side_index(winner_side)
            wins[winner_index] += 1

        bot1.avg_ne_across_g += bot1.get_avg_evals()
        bot2.avg_ne_across_g += bot2.get_avg_evals()

        if change_sides:
            player2_starts = not player2_starts


    print ("1st agent was {} and scored: {} with {} number of average evaluations across games".format(bot1.get_name(),str(wins[0]),bot1.avg_ne_across_g/no_of_games))
    print("2st agent was {} and scored: {} with {} number of average evaluations across games".format(bot2.get_name(),
                                                                                                      str(wins[1]),
                                                                                                      bot2.avg_ne_across_g/no_of_games))

    game.undo_last_move()
    game.current_state.show()


def get_agent(str_bot, depth=None,react_time=None):
    """Load strbot.strbot class as we expect all agent to be included this way"""
    top_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    module=  importlib.import_module("."+str_bot,top_dir)
    # module = getattr(sys.modules[__name__], str_bot)
    reload(module)
    cls = getattr(module,str_bot)
    return cls(depth=depth, react_time=react_time)

