import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import best_move, load_dictionary


BOARD_WITH_CAT = "\n".join([
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    ".......CAT.....",
    ".........H.....",
    ".........E.....",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
])


def test_best_extension_left_with_S():
    # Expect 'SCAT' at (6,6) horizontally, which beats 'CATS'
    board = Board.from_string(BOARD_WITH_CAT)
    dictionary = load_dictionary(os.path.join(ROOT, "dictionaries", "en_small.txt"))
    move = best_move(board, "S", dictionary)
    assert move is not None
    word, r, c, d, score = move
    assert word == "SCAT"
    assert (r, c, d) == (6, 6, 'H')
    assert score == 7  # 3+1+1 + S on DL -> 2


def test_best_extension_with_blank():
    # With a blank, the new letter scores 0; both 'SCAT' and 'CATS' yield 5
    board = Board.from_string(BOARD_WITH_CAT)
    dictionary = load_dictionary(os.path.join(ROOT, "dictionaries", "en_small.txt"))
    move = best_move(board, "?", dictionary)
    assert move is not None
    word, r, c, d, score = move
    assert score == 5
    assert d == 'H' and r == 6
    # Should mark the newly placed letter as lowercase (blank)
    assert any(ch.islower() for ch in word)
    # Accept either left extension at (6,6) or right at (6,7)
    assert (r, c) in {(6, 6), (6, 7)}
