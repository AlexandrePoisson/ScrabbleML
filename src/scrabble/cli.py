import argparse
import sys
from typing import Optional

from .board import Board
from .move_generator import best_move, load_dictionary
from .ocr import ocr_board_image


def _parse_board_string(board_string: Optional[str]) -> Board:
    if not board_string:
        return Board.empty()
    return Board.from_string(board_string)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scrabble OCR + Move Suggester")
    p.add_argument("--image", type=str, help="Path to board image (optional)")
    p.add_argument("--board-string", type=str, help="15 lines of 15 chars; '.' empty; A-Z tiles; a-z blank")
    p.add_argument("--rack", required=True, type=str, help="Your rack letters (use '?' for blanks)")
    p.add_argument("--dict", required=True, type=str, dest="dict_path", help="Path to dictionary file (one word per line)")
    p.add_argument("--ocr-engine", default="tesseract", choices=["tesseract", "cnn"], help="OCR engine")
    p.add_argument("--ocr-debug-dir", type=str, help="Directory to write per-cell debug images from OCR")
    p.add_argument("--ocr-inner-margin", type=float, default=0.18, help="Center crop margin fraction for each cell (0-0.4)")
    p.add_argument("--ocr-min-conf", type=float, default=75.0, help="Minimum Tesseract confidence (0-100) to accept a letter")
    p.add_argument("--ocr-detect", type=str, default="auto", choices=["auto", "contour", "rect", "hough", "proj"], help="Board detection method")
    p.add_argument("--ocr-min-area-frac", type=float, default=0.10, help="Min board area fraction to accept (0-1)")
    p.add_argument("--ocr-psm", type=int, default=10, help="Tesseract page segmentation mode (default 10: single char)")
    p.add_argument("--ocr-psm-fallback", type=int, default=8, help="Fallback PSM if primary yields nothing (set -1 to disable)")
    p.add_argument("--ocr-binarize", action="store_true", help="Apply extra binarization + cleanup before OCR")
    p.add_argument("--ocr-isolate-main", action="store_true", help="Keep only the largest contour per cell before OCR (drops small digits)")
    p.add_argument("--ocr-cnn-checkpoint", type=str, help="Path to cnn_ocr checkpoint (.pt) when using --ocr-engine=cnn")
    p.add_argument("--ocr-cnn-device", type=str, help="Torch device for cnn engine (cpu|cuda)")
    p.add_argument("--ocr-cnn-min-conf", type=float, default=0.0, help="Minimum probability threshold for CNN predictions (0-1)")
    p.add_argument("--ocr-cnn-image-size", type=int, default=32, help="Image size used during CNN training")
    p.add_argument("--ocr-corners", type=str, help="Manual corner hints: 'x1,y1;x2,y2;x3,y3;x4,y4' (order TL,TR,BL,BR)")
    p.add_argument("--lang", default="EN", choices=["EN", "FR"], help="Language for dictionary and scoring")

    args = p.parse_args(argv)

    cnn_model = None
    cnn_alphabet = None
    cnn_device = None
    if args.ocr_engine == "cnn":
        if not args.ocr_cnn_checkpoint:
            p.error("--ocr-cnn-checkpoint is required when --ocr-engine=cnn")
        try:
            from . import cnn_ocr
        except Exception as exc:
            p.error(f"cnn engine requested but torch/torchvision not available: {exc}")
        cnn_model, cnn_meta = cnn_ocr.load_checkpoint(args.ocr_cnn_checkpoint, device=args.ocr_cnn_device)
        cnn_alphabet = cnn_meta.get("alphabet")
        cnn_device = cnn_meta.get("device")

    if args.image:
        def _parse_corners(s: str | None) -> list[tuple[float, float]] | None:
            if not s:
                return None
            try:
                parts = s.strip().split(";")
                pts: list[tuple[float, float]] = []
                for p in parts:
                    x_str, y_str = p.strip().split(",")
                    pts.append((float(x_str), float(y_str)))
                if len(pts) != 4:
                    return None
                return pts
            except Exception:
                return None

        board_letters = ocr_board_image(
            args.image,
            engine=args.ocr_engine,
            debug_dir=args.ocr_debug_dir,
            inner_margin=args.ocr_inner_margin,
            min_conf=args.ocr_min_conf,
            detection_method=args.ocr_detect,
            min_area_frac=args.ocr_min_area_frac,
            corner_hints=_parse_corners(args.ocr_corners),
            psm=args.ocr_psm,
            psm_fallback=(None if args.ocr_psm_fallback < 0 else args.ocr_psm_fallback),
            apply_binarize=args.ocr_binarize,
            isolate_main_blob=args.ocr_isolate_main,
            cnn_model=cnn_model,
            cnn_alphabet=cnn_alphabet,
            cnn_device=cnn_device,
            cnn_min_conf=args.ocr_cnn_min_conf,
            cnn_image_size=args.ocr_cnn_image_size,
        )
        # Convert list of lists to Board (None for empty)
        # Use '.' for None when formatting; pass-through letters
        rows = []
        for row in board_letters:
            rows.append("".join(ch if ch is not None else '.' for ch in row))
        board = Board.from_string("\n".join(rows))
    else:
        board = _parse_board_string(args.board_string)

    dictionary = load_dictionary(args.dict_path)
    move = best_move(board, args.rack, dictionary, lang=args.lang)
    if move is None:
        print("No valid moves found.")
        return 1
    word, r, c, d, score = move

    placed = []
    dr, dc = (0, 1) if d == 'H' else (1, 0)
    for i, ch in enumerate(word):
        rr = r + i * dr
        cc = c + i * dc
        existing = board.grid[rr][cc]
        if existing is None:
            if 'a' <= ch <= 'z':
                placed.append((rr, cc, ch.upper(), True))
            else:
                placed.append((rr, cc, ch, False))

    print(f"Best: {word} at ({r},{c}) {d} score={score}")
    if placed:
        pretty = []
        for rr, cc, ch, is_blank in placed:
            pretty.append(f"{ch}{'*' if is_blank else ''}@({rr},{cc})")
        print("New tiles: " + ", ".join(pretty) + ("  (* = blank)" if any(p[3] for p in placed) else ""))

    # Print the board with the move applied ('.' empty, A-Z tile, a-z blank tile).
    out_grid = [[('.' if board.grid[rr][cc] is None else board.grid[rr][cc]) for cc in range(15)] for rr in range(15)]
    dr, dc = (0, 1) if d == 'H' else (1, 0)
    for i, ch in enumerate(word):
        rr = r + i * dr
        cc = c + i * dc
        if board.grid[rr][cc] is None:
            out_grid[rr][cc] = ch
    print("Board after move:")
    print("\n".join("".join(row) for row in out_grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
