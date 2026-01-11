from dataclasses import dataclass
from typing import List, Optional

BOARD_SIZE = 15

# Standard Scrabble premium squares layout
# Codes: ".." normal, "TW" triple word, "DW" double word, "TL" triple letter, "DL" double letter
PREMIUMS: List[List[str]] = [
    ["TW","..","..","DL","..","..","..","TW","..","..","..","DL","..","..","TW"],
    ["..","DW","..","..","..","TL","..","..","..","TL","..","..","..","DW",".."],
    ["..","..","DW","..","..","..","DL","..","DL","..","..","..","DW","..",".."],
    ["DL","..","..","DW","..","..","..","DL","..","..","..","DW","..","..","DL"],
    ["..","..","..","..","DW","..","..","..","..","..","DW","..","..","..",".."],
    ["..","TL","..","..","..","TL","..","..","..","TL","..","..","..","TL",".."],
    ["..","..","DL","..","..","..","DL","..","DL","..","..","..","DL","..",".."],
    ["TW","..","..","DL","..","..","..","DW","..","..","..","DL","..","..","TW"],
    ["..","..","DL","..","..","..","DL","..","DL","..","..","..","DL","..",".."],
    ["..","TL","..","..","..","TL","..","..","..","TL","..","..","..","TL",".."],
    ["..","..","..","..","DW","..","..","..","..","..","DW","..","..","..",".."],
    ["DL","..","..","DW","..","..","..","DL","..","..","..","DW","..","..","DL"],
    ["..","..","DW","..","..","..","DL","..","DL","..","..","..","DW","..",".."],
    ["..","DW","..","..","..","TL","..","..","..","TL","..","..","..","DW",".."],
    ["TW","..","..","DL","..","..","..","TW","..","..","..","DL","..","..","TW"],
]


@dataclass
class Board:
    # grid[r][c] is None for empty, 'A'-'Z' for letters, 'a'-'z' for blanks
    grid: List[List[Optional[str]]]

    @staticmethod
    def empty() -> "Board":
        return Board([[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)])

    @staticmethod
    def from_string(multiline: str) -> "Board":
        # 15 lines of 15 chars; '.' empty, 'A-Z' letter, 'a-z' blank
        rows = [line.strip() for line in multiline.strip().splitlines() if line.strip()]
        if len(rows) != BOARD_SIZE or any(len(r) != BOARD_SIZE for r in rows):
            raise ValueError("Board string must be 15 lines of 15 characters")
        grid: List[List[Optional[str]]] = []
        for r in rows:
            row: List[Optional[str]] = []
            for ch in r:
                if ch == '.':
                    row.append(None)
                elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
                    row.append(ch)
                else:
                    raise ValueError(f"Invalid board character: {ch}")
            grid.append(row)
        return Board(grid)

    def is_empty(self) -> bool:
        return all(cell is None for row in self.grid for cell in row)

    def premium_at(self, r: int, c: int) -> str:
        return PREMIUMS[r][c]
