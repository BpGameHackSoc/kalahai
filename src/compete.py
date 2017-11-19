import sys
import model                               # Game model
from ai import Protocol as referee         # Comparison of engines

engine1 = sys.argv[1]
engine2 = sys.argv[2]

referee.run_competition(engine1, engine2, no_of_games=10)