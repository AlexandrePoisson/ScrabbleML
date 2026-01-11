import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.scoring import compute_main_word_score
from scrabble.move_generator import best_move


def test_scoring_diff_en_fr_qi_plain_cells():
    board = Board.empty()
    # Place 'QI' at row 0, col 1 horizontally to avoid premiums (0,0 is TW)
    en = compute_main_word_score(board, 'QI', 0, 1, 'H', lang='EN')
    fr = compute_main_word_score(board, 'QI', 0, 1, 'H', lang='FR')
    assert en == 11  # Q10 + I1
    assert fr == 9   # Q8 + I1


def test_best_move_uses_lang_parameter():
    board = Board.empty()
    # Use a tiny dictionary where the language weighting matters
    dictionary = {"QI"}
    # In EN: score (without premiums at col 1) is 11; center DW can change absolute score but we just check non-None
    move_en = best_move(board, "QI", dictionary, lang='EN')
    move_fr = best_move(board, "QI", dictionary, lang='FR')
    assert move_en is not None and move_fr is not None
    # Both cover center due to first move rule; but FR should yield a lower score than EN due to Q value
    assert move_en[4] > move_fr[4]
