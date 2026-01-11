import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import best_move


def test_best_move_rejects_invalid_cross_words():
    # Existing board letters:
    # - D at (5,3)
    # - A at (6,3)
    # - R at (7,4)
    # Candidate main word: ANOR at (4,4) vertical overlaps R at (7,4)
    # But it creates cross-words like "DN" and "AO" which are NOT in dictionary.
    board = Board.from_string(
        "\n".join(
            [
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...D...........",
                "...A...........",
                "....R..........",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]
        )
    )

    dictionary = {"ANOR"}
    move = best_move(board, rack="ANO", dictionary=dictionary, lang="FR")
    assert move is None
