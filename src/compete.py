import sys
import model                               # Game model
from ai import Protocol as referee         # Comparison of engines
try:
    engine1 = sys.argv[1]
    engine2 = sys.argv[2]
except:
    engine1="AlphaBeta"
    engine2 = "AlphaBetaOwnSeeds"

depth = 2

referee.run_competition(engine1, engine2, depth1=depth, depth2=depth, no_of_games=2,print_states=True)