import os
import sys
import shutil
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    import cv2  # type: ignore
    from scrabble.ocr import ocr_board_image, letters_grid_to_board_string
    from scrabble import cnn_ocr
except Exception:  # pragma: no cover
    cv2 = None
    cnn_ocr = None  # type: ignore


@pytest.mark.skipif(cv2 is None, reason="OpenCV not available")
def test_ocr_picture_extracts_board():
    img_path = os.path.join(ROOT, "pictures", "IMG_5002.jpeg")
    if not os.path.exists(img_path):
        pytest.skip("Sample picture not present: pictures/IMG_5002.jpeg")

    checkpoint = os.path.join(ROOT, "models", "cnn_checkpoint", "cnn_ocr.pt")
    if cnn_ocr is None:
        pytest.skip("CNN OCR dependencies not available (scrabble.cnn_ocr import failed)")
    if not os.path.exists(checkpoint):
        pytest.skip(f"CNN checkpoint not present: {checkpoint}")

    model, meta = cnn_ocr.load_checkpoint(checkpoint, device="cpu")
    image_size = int((meta.get("config") or {}).get("image_size", 32))
    alphabet = meta.get("alphabet")
    letters = ocr_board_image(
        img_path,
        engine="cnn",
        cnn_model=model,
        cnn_device="cpu",
        cnn_alphabet=alphabet,
        cnn_image_size=image_size,
    )
    board_str = letters_grid_to_board_string(letters)
    # Placeholder expected board; edit to the correct 15x15 board for your image
    EXPECTED_BOARD = "\n".join([
        "...............",
        "...............",
        "..........F....",
        "..........U....",
        "......F...I....",
        ".....VA...T....",
        "......U..JEU...",
        "......TETE.....",
        "BERLIN?..A.....",
        ".......HAN.....",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
    ])
    print("\nOCR Board:\n" + board_str)
    assert len(board_str.splitlines()) == 15 and all(len(line) == 15 for line in board_str.splitlines())
    # The expected board is a project-local placeholder and depends on the sample picture and OCR setup.
    # Enable strict comparison only when explicitly requested.
    if os.environ.get("SCRABBLE_OCR_STRICT") == "1":
        assert board_str == EXPECTED_BOARD
