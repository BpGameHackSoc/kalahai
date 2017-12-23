import importlib
import model
import numpy as np
import sys
import os
import copy

from importlib import reload

from . import Bot
reload(Bot)
# reload(RandomAgent)
reload(model)
# reload(AlphaBetaOwnSeeds)

bot1,bot2 = None,None
def run_competition(str_bot1, str_bot2, react_time=1, buckets=6, seeds=4,
                    pie_rule=True, print_states=False, print_games=False, no_of_games=2, change_sides=True,
                    depth1=None, depth2=None, test_openings=False):
    global bot1,bot2
    bot1 = get_agent(str_bot1, depth1,react_time=None)
    bot2 = get_agent(str_bot2, depth2,react_time=None)
    wins = np.zeros(2)

    bot1.avg_ne_across_g,bot2.avg_ne_across_g = (0,0)
    players = [bot1, bot2]

    game_list = [model.Game(buckets, seeds, pie_rule, print_results=False)]
    if test_openings:
        starting_game = game_list[0]
        game_list.extend(model.Game.fork_game(starting_game,use_all_moves=True))

    for starter_game in game_list:
        player2_starts = False
        for i in range(no_of_games):
            game = copy.deepcopy(starter_game)
            if player2_starts:
                game.set_player2_current()
            else:
                game.set_player1_current()
            #statistics
            move_count = 0
            bot1.init_counters()
            bot2.init_counters()
            run_game(game,players,print_states=print_states)
            inc_wins(wins,game)

            bot1.avg_ne_across_g += bot1.get_avg_evals()
            bot2.avg_ne_across_g += bot2.get_avg_evals()

            if change_sides:
                player2_starts = not player2_starts

    no_simulations = len(game_list)*no_of_games
    print ("1st agent was {} and scored: {} with {} number of average evaluations across games".format(bot1.get_name(),str(wins[0]),bot1.avg_ne_across_g/no_simulations))
    print("2st agent was {} and scored: {} with {} number of average evaluations across games".format(bot2.get_name(),
                                                                                                      str(wins[1]),
                                                                                                      bot2.avg_ne_across_g/no_simulations))

    # game.undo_last_move()
    # game.current_state.show()

def run_game(game_obj,players,print_states=False):
    while not game_obj.game_over():  # Calculate one move per loop
        current_player_index = game_obj.get_current_player_index()
        current_bot = players[current_player_index]
        move_index = current_bot.move(game_obj.current_state)
        game_obj.apply_move(move_index)
        if print_states:
            print(game_obj.current_state.players)
            game_obj.current_state.show()

def inc_wins(wins, game_obj):
    winner_side = game_obj.get_winner_side()
    if winner_side.is_draw():
        wins += 0.5
    else:
        winner_index = game_obj.side_index(winner_side)
        wins[winner_index] += 1


def get_agent(str_bot, depth=None,react_time=None):
    """Load strbot.strbot class as we expect all agent to be included this way"""
    top_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    module=  importlib.import_module("."+str_bot,top_dir)
    # module = getattr(sys.modules[__name__], str_bot)
    reload(module)
    cls = getattr(module,str_bot)
    return cls(depth=depth, react_time=react_time)

