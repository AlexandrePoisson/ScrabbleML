from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .board import Board, BOARD_SIZE
from .scoring import compute_play_score


def load_dictionary(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip() and line[0].isalpha()}


@dataclass
class _TrieNode:
    children: Dict[str, "_TrieNode"] = field(default_factory=dict)
    is_word: bool = False


def _build_trie(dictionary: Iterable[str]) -> Tuple[_TrieNode, int]:
    root = _TrieNode()
    max_len = 0
    for w in dictionary:
        word = w.strip().upper()
        if not word or not word[0].isalpha():
            continue
        if len(word) > BOARD_SIZE:
            continue
        if len(word) < 2:
            continue
        max_len = max(max_len, len(word))
        node = root
        ok = True
        for ch in word:
            if not ('A' <= ch <= 'Z'):
                ok = False
                break
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
        if ok:
            node.is_word = True
    return root, max_len


def _can_supply_from_rack(required: Iterable[str], rack: str) -> bool:
    # rack may contain '?' representing a blank tile
    rack_counts = Counter(rack.upper())
    for ch in required:
        if rack_counts[ch] > 0:
            rack_counts[ch] -= 1
        elif rack_counts['?'] > 0:
            rack_counts['?'] -= 1
        else:
            return False
    return True


def _fits_and_uses_rack(board: Board, word: str, row: int, col: int, dir_: str, rack: str) -> Optional[str]:
    dr, dc = (0, 1) if dir_ == 'H' else (1, 0)
    needed: list[str] = []
    touched_existing = False
    for i, ch in enumerate(word):
        r = row + i * dr
        c = col + i * dc
        existing = board.grid[r][c]
        if existing is None:
            needed.append(ch)
        else:
            if existing.upper() != ch:
                return None
            touched_existing = True
    # If board not empty, require touching
    if not board.is_empty() and not touched_existing:
        return None
    if _can_supply_from_rack(needed, rack):
        # Return a mask representing whether each position uses rack (uppercase) or blank (lowercase)
        rack_counts = Counter(rack.upper())
        word_out = list(word)
        for i, ch in enumerate(word):
            r = row + i * dr
            c = col + i * dc
            if board.grid[r][c] is None:
                if rack_counts[ch] > 0:
                    rack_counts[ch] -= 1
                    word_out[i] = ch  # normal tile
                else:
                    rack_counts['?'] -= 1
                    word_out[i] = ch.lower()  # blank tile
        return "".join(word_out)
    return None


def _occupied_positions_by_letter(board: Board) -> Dict[str, List[Tuple[int, int]]]:
    by_letter: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            ch = board.grid[r][c]
            if ch is None:
                continue
            up = ch.upper()
            if 'A' <= up <= 'Z':
                by_letter[up].append((r, c))
    return dict(by_letter)


def _candidate_starts_for_word(
    board: Board,
    word: str,
    dir_: str,
    occupied_by_letter: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Sequence[Tuple[int, int]]:
    """Return candidate (row, col) starts that are worth checking.

    Step-1 optimization: instead of scanning every board start cell, generate starts that:
    - cover center when board is empty
    - overlap at least one existing tile (current baseline rule) when board is not empty
    """
    dr, dc = (0, 1) if dir_ == 'H' else (1, 0)
    L = len(word)
    starts: Set[Tuple[int, int]] = set()

    center_r = BOARD_SIZE // 2
    center_c = BOARD_SIZE // 2

    if board.is_empty():
        # Must cover center: align each word index to the center.
        for i in range(L):
            r0 = center_r - i * dr
            c0 = center_c - i * dc
            end_r = r0 + (L - 1) * dr
            end_c = c0 + (L - 1) * dc
            if 0 <= r0 < BOARD_SIZE and 0 <= c0 < BOARD_SIZE and 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE:
                starts.add((r0, c0))
        return sorted(starts)

    if not occupied_by_letter:
        return []

    # Baseline currently requires overlapping an existing tile; only consider starts
    # that align some word letter with a matching existing letter on the board.
    for i, ch in enumerate(word):
        positions = occupied_by_letter.get(ch)
        if not positions:
            continue
        for r_anchor, c_anchor in positions:
            r0 = r_anchor - i * dr
            c0 = c_anchor - i * dc
            end_r = r0 + (L - 1) * dr
            end_c = c0 + (L - 1) * dc
            if 0 <= r0 < BOARD_SIZE and 0 <= c0 < BOARD_SIZE and 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE:
                starts.add((r0, c0))
    return sorted(starts)


def _candidate_starts_any_word(board: Board, dir_: str) -> Sequence[Tuple[int, int]]:
    """Return candidate (row, col) starts for any word in direction.

    Mirrors the baseline legality rule in `_fits_and_uses_rack`:
    - first move must cover center
    - otherwise the placement must overlap (match) at least one existing tile
    """
    dr, dc = (0, 1) if dir_ == 'H' else (1, 0)
    starts: Set[Tuple[int, int]] = set()
    center_r = BOARD_SIZE // 2
    center_c = BOARD_SIZE // 2

    if board.is_empty():
        if dir_ == 'H':
            r = center_r
            for c in range(center_c + 1):
                starts.add((r, c))
        else:
            c = center_c
            for r in range(center_r + 1):
                starts.add((r, c))
        return sorted(starts)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            ch = board.grid[r][c]
            if ch is None:
                continue
            # For any word that overlaps (r, c), its start must be at most 14 cells "behind".
            for back in range(BOARD_SIZE):
                r0 = r - back * dr
                c0 = c - back * dc
                if not (0 <= r0 < BOARD_SIZE and 0 <= c0 < BOARD_SIZE):
                    break
                starts.add((r0, c0))
    return sorted(starts)


def _cross_words_valid(
    board: Board,
    word_with_blanks: str,
    row: int,
    col: int,
    dir_: str,
    dictionary: Set[str],
) -> bool:
    """Validate that all perpendicular words formed by newly placed tiles exist in dictionary.

    Note: main-word validity is guaranteed by generation from `dictionary` (trie).
    This only checks cross-words (length >= 2).
    """
    dr, dc = (0, 1) if dir_ == 'H' else (1, 0)
    pr, pc = (1, 0) if dir_ == 'H' else (0, 1)  # perpendicular direction

    for i, ch in enumerate(word_with_blanks):
        r = row + i * dr
        c = col + i * dc
        if board.grid[r][c] is not None:
            continue  # not a newly placed tile

        center = ch.upper()

        # Walk backwards
        left: List[str] = []
        rr, cc = r - pr, c - pc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            existing = board.grid[rr][cc]
            if existing is None:
                break
            left.append(existing.upper())
            rr -= pr
            cc -= pc

        # Walk forwards
        right: List[str] = []
        rr, cc = r + pr, c + pc
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            existing = board.grid[rr][cc]
            if existing is None:
                break
            right.append(existing.upper())
            rr += pr
            cc += pc

        if not left and not right:
            continue

        cross = "".join(reversed(left)) + center + "".join(right)
        if len(cross) >= 2 and cross.upper() not in dictionary:
            return False

    return True


def _full_main_word_formed(
    board: Board,
    word_with_blanks: str,
    row: int,
    col: int,
    dir_: str,
) -> Tuple[str, int, int]:
    """Return the full contiguous main word formed, including extensions through existing tiles.

    The returned word includes:
    - any existing tiles immediately before/after the placed segment in `dir_`
    - the placed segment `word_with_blanks` (including lowercase for rack blanks)

    Returns: (full_word_with_blanks, start_row, start_col)
    """
    dr, dc = (0, 1) if dir_ == 'H' else (1, 0)

    # Extend backwards from the placed segment start.
    start_r, start_c = row, col
    while True:
        rr = start_r - dr
        cc = start_c - dc
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            break
        if board.grid[rr][cc] is None:
            break
        start_r, start_c = rr, cc

    # Extend forwards from the placed segment end.
    end_r = row + (len(word_with_blanks) - 1) * dr
    end_c = col + (len(word_with_blanks) - 1) * dc
    while True:
        rr = end_r + dr
        cc = end_c + dc
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            break
        if board.grid[rr][cc] is None:
            break
        end_r, end_c = rr, cc

    # Build full word from (start_r, start_c) to (end_r, end_c)
    full: List[str] = []
    cur_r, cur_c = start_r, start_c
    while True:
        if dir_ == 'H' and cur_r == row and col <= cur_c <= col + len(word_with_blanks) - 1:
            idx = cur_c - col
            full.append(word_with_blanks[idx])
        elif dir_ == 'V' and cur_c == col and row <= cur_r <= row + len(word_with_blanks) - 1:
            idx = cur_r - row
            full.append(word_with_blanks[idx])
        else:
            existing = board.grid[cur_r][cur_c]
            if existing is None:
                # Should not happen: we only traverse through contiguous occupied cells.
                break
            full.append(existing)

        if cur_r == end_r and cur_c == end_c:
            break
        cur_r += dr
        cur_c += dc

    return "".join(full), start_r, start_c


def best_move(
    board: Board,
    rack: str,
    dictionary: Iterable[str],
    lang: str = 'EN',
) -> Optional[Tuple[str, int, int, str, int]]:
    # Returns (word_with_blanks, row, col, direction, score)
    best: Optional[Tuple[str, int, int, str, int]] = None

    dict_set: Set[str]
    if isinstance(dictionary, set):
        dict_set = {w.upper() for w in dictionary}
    else:
        dict_set = {str(w).strip().upper() for w in dictionary}

    trie_root, max_word_len = _build_trie(dict_set)
    if max_word_len < 2:
        return None

    rack_counts0 = Counter(rack.upper())
    center_r = BOARD_SIZE // 2
    center_c = BOARD_SIZE // 2

    def dfs_from_start(
        r0: int,
        c0: int,
        dir_: str,
        node: _TrieNode,
        rack_counts: Counter,
        word_chars: List[str],
        touched_existing: bool,
        used_new: bool,
        covered_center: bool,
        depth: int,
    ) -> None:
        nonlocal best
        if depth >= max_word_len:
            return

        dr, dc = (0, 1) if dir_ == 'H' else (1, 0)
        r = r0 + depth * dr
        c = c0 + depth * dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return

        existing = board.grid[r][c]
        if existing is not None:
            ch = existing.upper()
            nxt = node.children.get(ch)
            if nxt is None:
                return
            word_chars.append(ch)
            new_touched_existing = True
            new_used_new = used_new
            new_covered_center = covered_center or (r == center_r and c == center_c)

            if nxt.is_word and len(word_chars) >= 2:
                if board.is_empty():
                    ok = new_covered_center
                else:
                    ok = new_touched_existing and new_used_new
                if ok:
                    fitted = "".join(word_chars)
                    full_word, full_r0, full_c0 = _full_main_word_formed(board, fitted, r0, c0, dir_)
                    if full_word.upper() not in dict_set:
                        pass
                    elif not _cross_words_valid(board, fitted, r0, c0, dir_, dict_set):
                        pass
                    else:
                        score = compute_play_score(board, full_word, full_r0, full_c0, dir_, lang=lang)
                        if score is not None:
                            cand = (full_word, full_r0, full_c0, dir_, score)
                            if best is None or score > best[4]:
                                best = cand

            dfs_from_start(
                r0,
                c0,
                dir_,
                nxt,
                rack_counts,
                word_chars,
                new_touched_existing,
                new_used_new,
                new_covered_center,
                depth + 1,
            )
            word_chars.pop()
            return

        # Empty cell: try letters from rack or blank.
        child_letters = sorted(node.children.keys())

        # Use non-blank rack letters that are valid next trie edges.
        for ch in child_letters:
            if rack_counts.get(ch, 0) <= 0:
                continue
            rack_counts[ch] -= 1
            word_chars.append(ch)
            nxt = node.children[ch]
            new_covered_center = covered_center or (r == center_r and c == center_c)

            if nxt.is_word and len(word_chars) >= 2:
                if board.is_empty():
                    ok = new_covered_center
                else:
                    ok = touched_existing and True  # used_new is guaranteed when we place a tile
                if ok:
                    fitted = "".join(word_chars)
                    full_word, full_r0, full_c0 = _full_main_word_formed(board, fitted, r0, c0, dir_)
                    if full_word.upper() not in dict_set:
                        pass
                    elif _cross_words_valid(board, fitted, r0, c0, dir_, dict_set):
                        score = compute_play_score(board, full_word, full_r0, full_c0, dir_, lang=lang)
                        if score is not None:
                            cand = (full_word, full_r0, full_c0, dir_, score)
                            if best is None or score > best[4]:
                                best = cand

            dfs_from_start(
                r0,
                c0,
                dir_,
                nxt,
                rack_counts,
                word_chars,
                touched_existing,
                True,
                new_covered_center,
                depth + 1,
            )
            word_chars.pop()
            rack_counts[ch] += 1

        # Use blank as any next trie edge (lowercase in fitted word for scoring=0).
        if rack_counts.get('?', 0) > 0:
            for ch in child_letters:
                rack_counts['?'] -= 1
                word_chars.append(ch.lower())
                nxt = node.children[ch]
                new_covered_center = covered_center or (r == center_r and c == center_c)

                if nxt.is_word and len(word_chars) >= 2:
                    if board.is_empty():
                        ok = new_covered_center
                    else:
                        ok = touched_existing and True
                    if ok:
                        fitted = "".join(word_chars)
                        full_word, full_r0, full_c0 = _full_main_word_formed(board, fitted, r0, c0, dir_)
                        if full_word.upper() not in dict_set:
                            pass
                        elif _cross_words_valid(board, fitted, r0, c0, dir_, dict_set):
                            score = compute_play_score(board, full_word, full_r0, full_c0, dir_, lang=lang)
                            if score is not None:
                                cand = (full_word, full_r0, full_c0, dir_, score)
                                if best is None or score > best[4]:
                                    best = cand

                dfs_from_start(
                    r0,
                    c0,
                    dir_,
                    nxt,
                    rack_counts,
                    word_chars,
                    touched_existing,
                    True,
                    new_covered_center,
                    depth + 1,
                )
                word_chars.pop()
                rack_counts['?'] += 1

    for dir_ in ('H', 'V'):
        for r0, c0 in _candidate_starts_any_word(board, dir_):
            dfs_from_start(
                r0,
                c0,
                dir_,
                trie_root,
                rack_counts0.copy(),
                [],
                False,
                False,
                False,
                0,
            )

    return best
