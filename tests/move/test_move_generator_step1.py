import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrabble.board import Board
from scrabble.move_generator import _candidate_starts_for_word, _occupied_positions_by_letter


BOARD_WITH_CAT = "\n".join([
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    ".......CAT.....",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
])


def test_candidate_starts_empty_board_cover_center():
    board = Board.empty()
    # For a 2-letter word on the first move, there are exactly 2 horizontal starts that cover center (7,7)
    starts_h = _candidate_starts_for_word(board, "QI", "H")
    assert set(starts_h) == {(7, 6), (7, 7)}
    # And 2 vertical starts
    starts_v = _candidate_starts_for_word(board, "QI", "V")
    assert set(starts_v) == {(6, 7), (7, 7)}


def test_candidate_starts_overlap_existing_letters():
    board = Board.from_string(BOARD_WITH_CAT)
    occ = _occupied_positions_by_letter(board)

    # 'SCAT' can only be started at (6,6) horizontally to overlap 'CAT'
    starts = _candidate_starts_for_word(board, "SCAT", "H", occ)
    assert (6, 6) in set(starts)

    # Every candidate start must overlap at least one existing cell with a matching letter
    word = "SCAT"
    ok = []
    for r0, c0 in starts:
        overlaps = False
        for i, ch in enumerate(word):
            existing = board.grid[r0][c0 + i]
            if existing is not None and existing.upper() == ch:
                overlaps = True
                break
        ok.append(overlaps)
    assert all(ok)
