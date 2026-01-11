from typing import Optional

from .board import Board, BOARD_SIZE

LETTER_SCORES_EN = {
    **{c: 1 for c in list("AEILNORSTU")},
    **{c: 2 for c in list("DG")},
    **{c: 3 for c in list("BCMP")},
    **{c: 4 for c in list("FHVWY")},
    "K": 5,
    **{c: 8 for c in list("JX")},
    **{c: 10 for c in list("QZ")},
}

# French Scrabble letter scores (no diacritics on tiles)
LETTER_SCORES_FR = {
    **{c: 1 for c in list("AEILNORSTU")},
    **{c: 2 for c in list("DGM")},
    **{c: 3 for c in list("BCP")},
    **{c: 4 for c in list("FHV")},
    **{c: 8 for c in list("JQ")},
    **{c: 10 for c in list("KWXYZ")},
}

def _scores_for_lang(lang: str) -> dict[str, int]:
    return LETTER_SCORES_EN if lang.upper() == 'EN' else LETTER_SCORES_FR


def _tile_score(ch: str, lang: str) -> int:
    # lowercase -> blank tile (0 points)
    if ch is None:
        return 0
    if 'a' <= ch <= 'z':
        return 0
    return _scores_for_lang(lang).get(ch, 0)


def compute_main_word_score(
    board: Board, word: str, row: int, col: int, direction: str, lang: str = 'EN'
) -> Optional[int]:
    # direction: 'H' or 'V'
    dr, dc = (0, 1) if direction == 'H' else (1, 0)
    # Validate bounds
    end_r = row + (len(word) - 1) * dr
    end_c = col + (len(word) - 1) * dc
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE):
        return None

    total = 0
    word_mult = 1
    used_any_new = False
    for i, ch in enumerate(word):
        r = row + i * dr
        c = col + i * dc
        existing = board.grid[r][c]
        if existing is not None:
            # Must match existing letter (case-insensitive)
            if existing.upper() != ch.upper():
                return None
            total += _tile_score(existing, lang)  # existing letters count, no multipliers
        else:
            used_any_new = True
            letter_score = _tile_score(ch, lang)
            premium = board.premium_at(r, c)
            if premium == 'DL':
                letter_score *= 2
            elif premium == 'TL':
                letter_score *= 3
            elif premium == 'DW':
                word_mult *= 2
            elif premium == 'TW':
                word_mult *= 3
            total += letter_score

    if not used_any_new:
        # Must place at least one new tile
        return None

    total *= word_mult
    # Note: cross-word scoring not included (baseline)
    return total


def compute_play_score(
    board: Board, word: str, row: int, col: int, direction: str, lang: str = 'EN'
) -> Optional[int]:
    """Compute the total score for a play: main word + all cross-words.

    Rules implemented:
    - Letter/word premiums apply only for newly placed tiles.
    - A premium on a newly placed tile applies to every word formed that includes that tile
      (main word and any cross-words).
    - Existing tiles contribute their face value (0 for blanks) and do not re-trigger premiums.
    """
    dr, dc = (0, 1) if direction == 'H' else (1, 0)
    pr, pc = (1, 0) if direction == 'H' else (0, 1)  # perpendicular direction

    end_r = row + (len(word) - 1) * dr
    end_c = col + (len(word) - 1) * dc
    if not (
        0 <= row < BOARD_SIZE
        and 0 <= col < BOARD_SIZE
        and 0 <= end_r < BOARD_SIZE
        and 0 <= end_c < BOARD_SIZE
    ):
        return None

    # Main word scoring + collect newly placed tiles.
    main_sum = 0
    main_word_mult = 1
    new_tiles: list[tuple[int, int, str]] = []  # (r, c, ch-from-word)
    for i, ch in enumerate(word):
        r = row + i * dr
        c = col + i * dc
        existing = board.grid[r][c]
        if existing is not None:
            if existing.upper() != ch.upper():
                return None
            main_sum += _tile_score(existing, lang)
        else:
            new_tiles.append((r, c, ch))
            letter_score = _tile_score(ch, lang)
            premium = board.premium_at(r, c)
            if premium == 'DL':
                letter_score *= 2
            elif premium == 'TL':
                letter_score *= 3
            elif premium == 'DW':
                main_word_mult *= 2
            elif premium == 'TW':
                main_word_mult *= 3
            main_sum += letter_score

    if not new_tiles:
        return None

    total = main_sum * main_word_mult

    # Cross-words formed by each newly placed tile.
    for r, c, placed_ch in new_tiles:
        # Walk backwards in perpendicular direction.
        before: list[str] = []
        rr, cc = r - pr, c - pc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            existing = board.grid[rr][cc]
            if existing is None:
                break
            before.append(existing)
            rr -= pr
            cc -= pc

        # Walk forwards in perpendicular direction.
        after: list[str] = []
        rr, cc = r + pr, c + pc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            existing = board.grid[rr][cc]
            if existing is None:
                break
            after.append(existing)
            rr += pr
            cc += pc

        if not before and not after:
            continue  # no cross-word

        cross_sum = 0
        # Existing tiles before
        for ch in reversed(before):
            cross_sum += _tile_score(ch, lang)

        # Newly placed tile (premiums apply)
        letter_score = _tile_score(placed_ch, lang)
        premium = board.premium_at(r, c)
        cross_word_mult = 1
        if premium == 'DL':
            letter_score *= 2
        elif premium == 'TL':
            letter_score *= 3
        elif premium == 'DW':
            cross_word_mult *= 2
        elif premium == 'TW':
            cross_word_mult *= 3
        cross_sum += letter_score

        # Existing tiles after
        for ch in after:
            cross_sum += _tile_score(ch, lang)

        total += cross_sum * cross_word_mult

    return total
