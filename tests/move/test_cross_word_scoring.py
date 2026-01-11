import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import best_move
from scrabble.scoring import compute_main_word_score


def test_best_move_score_includes_cross_word_points_os():
    board = Board.from_string(
        "\n".join(
            [
                ".......MODELENT",
                ".......A.....O.",
                ".......L.....D.",
                "..TAVELLE....A.",
                "....I..E.....L.",
                "....N........E.",
                "....M..........",
                ".AUREOLA.......",
                "....S..........",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
                "...............",
            ]
        )
    )

    # Deterministic: only allow the intended main word + the cross-word it creates (OS).
    dictionary = {"RIBLERAS", "OS"}

    move = best_move(board, rack="EIRRLSB", dictionary=dictionary, lang="FR")
    assert move is not None
    word, r, c, d, score = move
    assert (word, r, c, d) == ("RIBLERAS", 1, 1, 'H')

    # Baseline main-word-only score + 2 points for cross word "OS" (O existing above, S newly placed).
    main_only = compute_main_word_score(board, "RIBLERAS", 1, 1, 'H', lang="FR")
    assert main_only is not None
    assert score == main_only + 2
