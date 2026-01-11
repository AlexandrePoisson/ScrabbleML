# Scrabble ML

A webapp and CLI interface to showcasing how AI can help in finding the best move during a Scrabble play.


A first perspective is to use OCR to create a Scrabble board numerical representation
A second perspective is to to find the best move.


## Demo

See the video of OCR detection, using a CNN and then best move :

[![Demo of OCR and Best Move](https://img.youtube.com/vi/uEeuumiCYeQ/hqdefault.jpg)](https://youtu.be/uEeuumiCYeQ)


## Pipeline:

- Input: a Scrabble board photo or a board string.
- OCR: OpenCV grid extraction + OCR using a custom CNN - trained with tiles.
- Board: build a 15×15 internal representation with premium squares.
- Move: generate candidate plays using a dictionary, validate cross-words, and score all words formed.


## Language Support

- Scoring supports English (`EN`) and French (`FR`). Provide a matching dictionary and pass `--lang`.
- Board format: 15 lines × 15 chars; `.` empty, `A-Z` = letter, `a-z` = blank tile.


## Requirements
- Python 3.10+

Optional, for OCR using tesseract-ocr
- System Tesseract (for  OCR using tesseract-ocr):
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`

## Installation

Create Virtual Env
```bash
uv venv
```

Install
```bash
uv pip install -e .
```

Optional EasyOCR:
```bash
uv pip install "scrabble-ai[easyocr]"
```

Optional CNN OCR (PyTorch CPU by default):
```bash
uv pip install "scrabble-ai[cnn]"
```

## Using the Web Application
Run:
```bash
uv run python -m scrabble.webapp
```

Then open [webapp](http://127.0.0.1:8765)

Routes:
- `/play`: best-move UI (default landing page).
- `/build`: corner/warp/debug tool.
- `/label`: labeling + training UI for the CNN tile OCR.

Notes:
- Sessions/files are stored under `webdata/`.

### Load From Picture (Play)
On `/play`:
- Select an image.
- Adjust the four corners.
- Extract the board and load it into the play UI.

## CNN OCR (optional)
Label cell crops and train a lightweight classifier.

### Label + Train (Web UI)
Open http://127.0.0.1:8765/label.

Labeling:
- Pick a session (must have `extract_debug` with `cell_XX_YY.png`).
- Click labels (`.`, `?`, `A-Z`).
- Crops are copied into `webdata/label_store/images/<session>/`.
- Labels are appended to the single manifest: `webdata/label_store/label_manifest.jsonl`.

Training:
- Use the Train section and set `valFraction` (the trainer creates an internal validation split).
- Training progress (loss/accuracy) and logs are shown live.

Reload model:
- Click “Reload model” to reload `models/cnn_checkpoint/cnn_ocr.pt` into the running server (no restart needed).

Manifest cleanup:
- “Regenerate manifest” migrates older formats and archives stale manifests into `webdata/label_store/_archive_<timestamp>/`.


## Using the Command Line Interface
Dictionary file: plain text, one UPPERCASE word per line. See `dictionaries/en_small.txt` for format.


### Best move
1) Find best move on an empty board:
```bash
uv run scrabble --rack READING --dict dictionaries/en_small.txt
```

2) Find best move on a board string (15 lines × 15 chars; `.` empty, `A-Z` tile, `a-z` blank tile):
```bash
uv run scrabble --rack TEAR --dict dictionaries/en_small.txt --board-string """
...............
...............
...............
...............
...............
...............
.......CAT.....
.........H.....
.........E.....
...............
...............
...............
...............
...............
...............
"""
```

3) OCR from an image (assumes a roughly top-down board):
```bash
uv run scrabble --image path/to/board.jpg --rack READING --dict dictionaries/en_small.txt --ocr-engine tesseract --lang EN
```

4) French scoring and dictionary:
```bash
uv run scrabble --rack TAQUIN --dict dictionaries/fr_small.txt --lang FR
```

### Training a CNN
Train from the single label-store manifest:
```bash
python -m scrabble.cnn_ocr train \
  --manifest webdata/label_store/label_manifest.jsonl \
  --out models/cnn_checkpoint \
  --epochs 20 --batch-size 64 --val-fraction 0.2
```

Validate a checkpoint:
```bash
python -m scrabble.cnn_ocr validate \
  --manifest webdata/label_store/label_manifest.jsonl \
  --checkpoint models/cnn_checkpoint/cnn_ocr.pt \
  --batch-size 64
```

Optional sanity check over a directory of crops:
```bash
python -m scrabble.cnn_ocr infer \
  --checkpoint models/cnn_checkpoint/cnn_ocr.pt \
  --dir webdata/<session>/extract_debug \
  --min-conf 0.5
```

If you know a full board string for a session, you can also build a manifest directly from a session’s `extract_debug/` crops:
```bash
python -m scrabble.cnn_ocr build-manifest \
  --board-string path/to/board.txt \
  --session-dir webdata/<session>/extract_debug \
  --out data/cnn_manifest.jsonl
```

Use the CNN in the CLI instead of Tesseract:
```bash
uv run scrabble --image path/to/board.jpg --rack READING --dict dictionaries/en_small.txt \
  --ocr-engine cnn --ocr-cnn-checkpoint models/cnn_checkpoint/cnn_ocr.pt --ocr-cnn-min-conf 0.5
```


## Execute Tests

All tests

```bash
uv run pytest -q
```

OCR tests only
```bash
uv run pytest ./tests/ocr -q
```

Best move test only

OCR tests
```bash
uv run pytest ./tests/move -q
```


## Roadmap
- Board detection & perspective correction.
- Move generation optimizations (anchors, cross-checks, trie/DAWG).
- MCTS and RL pluggable strategies.

## Examples (Regression Cases)

### Adjacent word validation
When using the ODS8 dictionary, with initial board:

```
...............
...............
...............
...............
...............
...D...........
...A...........
...GRILL.......
...U...........
...o...........
...N...........
...S...........
...............
...............
...............
```

and rack: `DIR?AKN`

The engine must reject placements that create invalid adjacent cross-words (e.g. `DN`, `NR`, …).

### Cross-word scoring
Board:

```
.......MODELENT
.......A.....O.
.......L.....D.
..TAVELLE....A.
....I..E.....L.
....N........E.
....M..........
.AUREOLA.......
....S..........
...............
...............
...............
...............
...............
...............
```

Rack: `EIRRLSB`

Example expected best move: `RIBLERAS` at `(1,1)`; the computed score must include both the main word and any adjacent cross-words formed (e.g. `OS`).

### Main word extension validation
With ODS8, board:

```
...............
...............
...............
...............
...............
...............
...............
.......FANEZ...
...............
...............
...............
...............
...............
...............
...............
```

and rack: `A?MELNR`

The engine must validate the fully-extended main word (e.g. reject a play that would extend into `FANEZAMU` if it’s not in the dictionary).
