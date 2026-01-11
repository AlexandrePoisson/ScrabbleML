import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import best_move, load_dictionary


def test_best_move_empty_board_reads_center():
    board = Board.empty()
    dictionary = load_dictionary(os.path.join(ROOT, "dictionaries", "en_small.txt"))
    move = best_move(board, "READING", dictionary)
    assert move is not None
    word, r, c, d, score = move
    assert word.upper() in dictionary
    # must cover center (7,7)
    assert 7 >= r and 7 >= c
