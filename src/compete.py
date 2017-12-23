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

referee.run_competition("AlphaBeta", "AlphaBetaOwnSeeds", depth1=3, depth2=3, no_of_games=2,test_openings=True)