import sys
import model                               # Game model
from ai import Protocol as referee         # Comparison of engines
try:
    engine1 = sys.argv[1]
    engine2 = sys.argv[2]
except:
    engine1="AlphaBeta"
    engine2 = "AlphaBetaOwnSeeds"

referee.run_competition(engine1, engine2, depth1=6, depth2=6, no_of_games=10)