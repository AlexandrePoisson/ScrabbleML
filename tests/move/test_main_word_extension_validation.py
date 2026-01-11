import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import best_move, load_dictionary


def test_best_move_rejects_invalid_extended_main_word():
    # Board has an existing horizontal word segment: FANeZ (e is a blank tile for E)
    # Playing ZAMU starting at the existing Z would actually form the extended word FANEZAMU,
    # which must be dictionary-valid. This is NOT valid, so the move must be rejected.
    board = Board.from_string(
        "\n".join(
            [
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                ".......FANeZ...",
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

    fr_small = load_dictionary(os.path.join(ROOT, "dictionaries", "fr_small.txt"))
    assert "ZAMU" in fr_small

    # Keep this deterministic: only allow the tempting played word, not the extended one.
    # If the algorithm tries to play ZAMU horizontally starting on the existing Z,
    # the true main word would be FANEZAMU, which is not in the dictionary and must be rejected.
    dictionary = {"ZAMU"}

    move = best_move(board, rack="A?MELNR", dictionary=dictionary, lang="FR")
    assert move is not None
    word, r, c, d, _score = move
    assert word.upper() != "FANEZAMU"
    # The pre-fix buggy signature was returning the played segment at the Z horizontally.
    assert not (word.upper() == "ZAMU" and (r, c, d) == (7, 11, 'H'))
