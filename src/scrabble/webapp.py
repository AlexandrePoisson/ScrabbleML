import json
import os
import shutil
import uuid
import time
import threading
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for

from .ocr import preprocess_board_image, _order_points, extract_board_from_warped_image
from .board import Board, PREMIUMS
from .move_generator import best_move, load_dictionary
from .scoring import LETTER_SCORES_EN, LETTER_SCORES_FR
import cv2  # type: ignore
import numpy as np  # type: ignore


def create_app() -> Flask:
    src_root = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        static_folder=str(src_root / "static"),
        template_folder=str(src_root / "templates"),
    )
    project_root = Path(__file__).resolve().parents[2]
    storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../webdata"))
    os.makedirs(storage_dir, exist_ok=True)
    label_root = os.path.join(storage_dir, "label_store")
    label_images_root = os.path.join(label_root, "images")
    os.makedirs(label_images_root, exist_ok=True)

    # CNN model cache (single-user/simple)
    _cnn_lock = threading.Lock()
    _cnn_cache: Dict[str, Any] = {
        "checkpoint": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/cnn_checkpoint/cnn_ocr.pt")),
        "device": "cpu",
        "model": None,
        "meta": None,
        "loaded_at": None,
        "error": None,
    }

    def _load_cnn_checkpoint(checkpoint_path: str, device: Optional[str] = None, force: bool = False):
        with _cnn_lock:
            if (
                not force
                and _cnn_cache.get("model") is not None
                and _cnn_cache.get("checkpoint") == checkpoint_path
                and (device is None or _cnn_cache.get("device") == device)
            ):
                return _cnn_cache["model"], _cnn_cache["meta"]

        from . import cnn_ocr

        model, meta = cnn_ocr.load_checkpoint(checkpoint_path, device=device or "cpu")
        with _cnn_lock:
            _cnn_cache["checkpoint"] = checkpoint_path
            _cnn_cache["device"] = device or "cpu"
            _cnn_cache["model"] = model
            _cnn_cache["meta"] = meta
            _cnn_cache["loaded_at"] = time.time()
            _cnn_cache["error"] = None
        return model, meta

    # Training state (in-memory; reset on server restart)
    _train_lock = threading.Lock()
    _train_proc: Optional[subprocess.Popen] = None
    _train_state: Dict[str, Any] = {
        "running": False,
        "history": [],
        "logs": [],
        "error": None,
        "exit_code": None,
        "started_at": None,
        "finished_at": None,
        "config": None,
    }

    def _train_log(line: str) -> None:
        with _train_lock:
            logs: List[str] = _train_state["logs"]
            logs.append(line)
            # Keep last ~300 lines.
            if len(logs) > 300:
                del logs[: len(logs) - 300]

    def _parse_train_metric_line(line: str) -> Optional[Dict[str, Any]]:
        # Expected format:
        # [cnn-train] epoch=1 train_loss=0.1234 train_acc=0.456 val_acc=0.789
        if not line.startswith("[cnn-train]"):
            return None
        parts = line.strip().split()
        out: Dict[str, Any] = {}
        for p in parts[1:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k == "epoch":
                try:
                    out["epoch"] = int(v)
                except Exception:
                    pass
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    pass
        return out if "epoch" in out else None

    def _training_thread(cmd: List[str]) -> None:
        nonlocal _train_proc
        try:
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            # Ensure src-layout imports work even if not installed editable.
            src_path = str(project_root / "src")
            if env.get("PYTHONPATH"):
                if src_path not in env["PYTHONPATH"].split(":"):
                    env["PYTHONPATH"] = src_path + ":" + env["PYTHONPATH"]
            else:
                env["PYTHONPATH"] = src_path
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            with _train_lock:
                _train_proc = proc
                _train_state["running"] = True
                _train_state["error"] = None
                _train_state["exit_code"] = None
                _train_state["finished_at"] = None

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                _train_log(line)
                metric = _parse_train_metric_line(line)
                if metric:
                    with _train_lock:
                        _train_state["history"].append(metric)

            code = proc.wait()
            with _train_lock:
                _train_state["exit_code"] = code
                _train_state["running"] = False
                _train_state["finished_at"] = time.time()
                if code != 0:
                    _train_state["error"] = f"Training process exited with code {code}"
        except Exception as exc:
            with _train_lock:
                _train_state["running"] = False
                _train_state["error"] = str(exc)
                _train_state["finished_at"] = time.time()
        finally:
            with _train_lock:
                _train_proc = None

    def _label_manifest_path() -> str:
        return os.path.join(label_root, "label_manifest.jsonl")

    def _archive_dir() -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = os.path.join(label_root, "_archive_" + ts)
        os.makedirs(p, exist_ok=True)
        return p

    def _archive_file(path: str, archive_dir: str) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        dest = os.path.join(archive_dir, os.path.basename(path))
        try:
            shutil.move(path, dest)
            return dest
        except Exception:
            return None

    def _migrate_label_manifest_to_cnn_format(archive_dir: Optional[str] = None) -> bool:
        """Ensure label manifest is a single CNN manifest: {path,label,(optional row/col/session/file)}.

        Removes any "split" fields and adds "path" when only "file" is present.
        Returns True if it rewrote the manifest.
        """
        manifest = _label_manifest_path()
        if not os.path.exists(manifest):
            return False

        rewritten = False
        out_entries: List[Dict[str, Any]] = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if "split" in data:
                    data.pop("split", None)
                    rewritten = True

                if "path" not in data:
                    key = data.get("file")
                    if key:
                        p = (Path(label_images_root) / str(key)).resolve()
                        if p.exists():
                            data["path"] = str(p)
                            rewritten = True

                if "path" in data and "label" in data:
                    out_entries.append(data)

        if not rewritten:
            return False

        if archive_dir is None:
            archive_dir = _archive_dir()
        _archive_file(manifest, archive_dir)
        with open(manifest, "w", encoding="utf-8") as out:
            for e in out_entries:
                out.write(json.dumps(e) + "\n")
        return True

    def _load_labels(session_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        manifest = _label_manifest_path()
        labels: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(manifest):
            return labels
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Support both:
                    # - New format: {path,label,session?,file?}
                    # - Old format: {file:"<sid>/<fname>",label,session,split}
                    key = data.get("file")
                    if not key and data.get("path"):
                        try:
                            p = Path(str(data.get("path")))
                            # Expect .../label_store/images/<sid>/<fname>
                            fname = p.name
                            sid = p.parent.name
                            key = f"{sid}/{fname}"
                            data.setdefault("file", key)
                            data.setdefault("session", sid)
                        except Exception:
                            key = None
                    if not key:
                        continue
                    if session_filter:
                        sess = data.get("session")
                        if sess and sess != session_filter:
                            continue
                        if not sess and not str(key).startswith(str(session_filter) + "/"):
                            continue
                    labels[str(key)] = data
                except Exception:
                    continue
        return labels

    def _save_labels(labels: Dict[str, Dict[str, Any]]) -> None:
        manifest = _label_manifest_path()
        os.makedirs(label_root, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            for entry in labels.values():
                # Ensure no split is written.
                if isinstance(entry, dict) and "split" in entry:
                    entry = dict(entry)
                    entry.pop("split", None)
                f.write(json.dumps(entry) + "\n")

    def _list_sessions_with_cells() -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(storage_dir)):
            sess_path = os.path.join(storage_dir, name)
            cells_dir = os.path.join(sess_path, "extract_debug")
            if not os.path.isdir(cells_dir):
                continue
            count = len([f for f in os.listdir(cells_dir) if f.startswith("cell_") and f.endswith(".png")])
            if count == 0:
                continue
            sessions.append({"id": name, "cells": count})
        return sessions

    @app.get("/")
    def index():
        return redirect(url_for("play_page"))

    @app.get("/build")
    def build_page():
        return render_template("index.html")

    @app.get("/label")
    def label_page():
        return render_template("labeler.html")

    dict_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dictionaries"))
    dict_whitelist = {
        "en_small": os.path.join(dict_dir, "en_small.txt"),
        "fr_small": os.path.join(dict_dir, "fr_small.txt"),
           "fr_ods8": os.path.join(dict_dir, "fr_ods8.txt"),
    }
    dict_cache: Dict[str, set[str]] = {}

    @app.get("/play")
    def play_page():
        return render_template(
            "play.html",
            premiums=PREMIUMS,
            letter_scores={"EN": LETTER_SCORES_EN, "FR": LETTER_SCORES_FR},
        )

    @app.post("/api/move/best")
    def api_best_move():
        data = request.get_json(silent=True) or {}
        board_string = data.get("boardString")
        rack = (data.get("rack") or "").strip()
        lang = (data.get("lang") or "FR").upper()
        dict_key = data.get("dictKey") or "fr_ods8"

        if not rack:
            return jsonify({"error": "rack is required"}), 400
        if lang not in {"EN", "FR"}:
            return jsonify({"error": "lang must be EN or FR"}), 400
        if dict_key not in dict_whitelist:
            return jsonify({"error": f"unknown dictKey: {dict_key}"}), 400

        try:
            board = Board.empty() if not board_string else Board.from_string(str(board_string))
        except Exception as exc:
            return jsonify({"error": f"invalid boardString: {exc}"}), 400

        dict_path = dict_whitelist[dict_key]
        if dict_path not in dict_cache:
            if not os.path.exists(dict_path):
                return jsonify({"error": f"dictionary file not found: {dict_path}"}), 400
            dict_cache[dict_path] = load_dictionary(dict_path)

        move = best_move(board, rack, dict_cache[dict_path], lang=lang)
        if move is None:
            return jsonify({"move": None})

        word, r, c, d, score = move
        placed = []
        dr, dc = (0, 1) if d == 'H' else (1, 0)
        for i, ch in enumerate(word):
            rr = r + i * dr
            cc = c + i * dc
            if board.grid[rr][cc] is None:
                placed.append({
                    "row": rr,
                    "col": cc,
                    "letter": ch.upper(),
                    "isBlank": ('a' <= ch <= 'z'),
                })

        return jsonify({
            "move": {
                "word": word,
                "row": r,
                "col": c,
                "dir": d,
                "score": score,
                "placed": placed,
            }
        })

    @app.post("/api/suggest")
    def suggest():
        if "image" not in request.files:
            return jsonify({"error": "image file missing"}), 400
        file = request.files["image"]
        sid = str(uuid.uuid4())
        sess_dir = os.path.join(storage_dir, sid)
        os.makedirs(sess_dir, exist_ok=True)
        src_path = os.path.join(sess_dir, "source.jpg")
        file.save(src_path)
        pre = preprocess_board_image(src_path, return_debug=True)
        warped, dbg = pre  # type: ignore
        suggest = None
        if isinstance(dbg, dict) and isinstance(dbg.get("quad_points"), np.ndarray):
            pts = dbg["quad_points"].astype(float).tolist()
            # Ensure order TL, TR, BR, BL (as returned by _order_points)
            suggest = pts
        # Save a display copy of the source and warped
        disp_src = os.path.join(sess_dir, "source_display.jpg")
        disp_warp = os.path.join(sess_dir, "warped_display.jpg")
        src_img = cv2.imread(src_path)
        if src_img is not None:
            cv2.imwrite(disp_src, src_img)
        cv2.imwrite(disp_warp, warped)
        return jsonify({
            "session": sid,
            "sourceUrl": f"/files/{sid}/source_display.jpg",
            "warpedUrl": f"/files/{sid}/warped_display.jpg",
            "suggestedCorners": suggest,
        })

    @app.post("/api/warp")
    def warp():
        data = request.get_json(silent=True) or {}
        sid = data.get("session")
        corners = data.get("corners")  # [[x,y], ...] TL,TR,BR,BL or TL,TR,BL,BR
        if not sid or not isinstance(corners, list) or len(corners) != 4:
            return jsonify({"error": "session or corners invalid"}), 400
        sess_dir = os.path.join(storage_dir, sid)
        src_path = os.path.join(sess_dir, "source.jpg")
        if not os.path.exists(src_path):
            return jsonify({"error": "session not found"}), 404
        src = cv2.imread(src_path)
        if src is None:
            return jsonify({"error": "failed to read source"}), 500
        pts = np.array(corners, dtype=np.float32)
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(src, M, (maxWidth, maxHeight))
        # Normalize to square and draw grid overlay
        side = max(maxWidth, maxHeight)
        warped = cv2.resize(warped, (side, side), interpolation=cv2.INTER_AREA)
        overlay = warped.copy()
        # Grid lines
        board_size = 15
        for i in range(board_size + 1):
            y = int(round(i * side / board_size))
            x = int(round(i * side / board_size))
            cv2.line(overlay, (0, min(y, side - 1)), (side - 1, min(y, side - 1)), (255, 0, 0), 1)
            cv2.line(overlay, (min(x, side - 1), 0), (min(x, side - 1), side - 1), (255, 0, 0), 1)
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        step = side / board_size
        for r in range(board_size):
            yy = int(round((r + 0.5) * step))
            cv2.putText(overlay, chr(ord('A') + r), (5, min(yy, side - 1)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        for c in range(board_size):
            xx = int(round((c + 0.5) * step))
            cv2.putText(overlay, str(c + 1), (min(xx, side - 1), 15), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        out_warp = os.path.join(sess_dir, "warp.png")
        out_grid = os.path.join(sess_dir, "warp_grid.png")
        cv2.imwrite(out_warp, warped)
        cv2.imwrite(out_grid, overlay)
        return jsonify({
            "warpedUrl": f"/files/{sid}/warp.png",
            "gridUrl": f"/files/{sid}/warp_grid.png",
        })

    @app.post("/api/extract")
    def extract():
        data = request.get_json(silent=True) or {}
        sid = data.get("session")
        engine = str(data.get("engine", "tesseract") or "tesseract").lower()
        inner_margin = float(data.get("innerMargin", 0.18))
        outer_margin = float(data.get("outerMargin", 0.0))
        min_conf = float(data.get("minConf", 75.0))
        psm = int(data.get("psm", 10))
        psm_fallback = int(data.get("psmFallback", 8)) if data.get("psmFallback", 8) is not None else 8
        apply_binarize = bool(data.get("binarize", False))
        isolate_main = bool(data.get("isolateMain", False))
        if not sid:
            return jsonify({"error": "session missing"}), 400
        sess_dir = os.path.join(storage_dir, sid)
        warp_path = os.path.join(sess_dir, "warp.png")
        if not os.path.exists(warp_path):
            return jsonify({"error": "warp not found; run /api/warp first"}), 400
        debug_dir = os.path.join(sess_dir, "extract_debug")
        
        # Load CNN model if needed
        cnn_model = None
        cnn_alphabet = None
        cnn_device = None
        if engine == "cnn":
            checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/cnn_checkpoint/cnn_ocr.pt"))
            if not os.path.exists(checkpoint_path):
                return jsonify({
                    "error": f"CNN checkpoint not found at {checkpoint_path}",
                    "code": "cnn_checkpoint_missing",
                }), 400
            try:
                cnn_model, meta = _load_cnn_checkpoint(checkpoint_path, device="cpu")
                cnn_alphabet = meta.get("alphabet") if isinstance(meta, dict) else None
                cnn_device = meta.get("device") if isinstance(meta, dict) else None
            except ImportError as exc:
                return jsonify({
                    "error": f"CNN OCR engine unavailable (missing dependency): {exc}. Install with scrabble-ai[cnn].",
                    "code": "cnn_unavailable",
                }), 400
            except Exception as exc:
                return jsonify({
                    "error": f"Failed to load CNN checkpoint: {type(exc).__name__}: {exc}",
                    "code": "cnn_load_failed",
                }), 500
        
        try:
            t_start = time.perf_counter()
            letters, board_str, preview_last, preview_first, previews_all, timing_cells = extract_board_from_warped_image(
                warp_path,
                inner_margin=inner_margin,
                outer_margin=outer_margin,
                min_conf=min_conf,
                psm=psm,
                psm_fallback=(None if psm_fallback < 0 else psm_fallback),
                apply_binarize=apply_binarize,
                isolate_main_blob=isolate_main,
                debug_dir=debug_dir,
                collect_timings=True,
                engine=engine,
                cnn_model=cnn_model,
                cnn_alphabet=cnn_alphabet,
                cnn_device=cnn_device,
                cnn_min_conf=min_conf / 100.0,
                cnn_image_size=32,
            )
            total_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"[extract] session={sid} engine={engine} total_ms={total_ms:.1f} cells={len(timing_cells)}")
        except Exception as exc:
            return jsonify({
                "error": f"Extraction failed: {type(exc).__name__}: {exc}",
                "code": "extract_failed",
            }), 500
        def _pack_preview(p: Optional[Dict[str, Any]]):
            if p and isinstance(p, dict) and "file" in p:
                return {
                    "row": p.get("row"),
                    "col": p.get("col"),
                    "char": p.get("char"),
                    "url": f"/files/{sid}/extract_debug/{p['file']}",
                }
            return None
        preview_out = _pack_preview(preview_first) or _pack_preview(preview_last)
        previews_seq = []
        if previews_all:
            for p in previews_all:
                if not isinstance(p, dict):
                    continue
                entry = {
                    "row": p.get("row"),
                    "col": p.get("col"),
                    "char": p.get("char"),
                }
                fname = p.get("file")
                if fname:
                    entry["url"] = f"/files/{sid}/extract_debug/{fname}"
                previews_seq.append(entry)
        return jsonify({
            "boardString": board_str,
            "letters": letters,
            "debugDir": f"/files/{sid}/extract_debug",
            "lastPreview": preview_out,
            "previews": previews_seq,
            "timings": {
                "total_ms": total_ms,
                "cells": timing_cells,
            },
        })

    @app.get("/files/<sid>/<path:name>")
    def files(sid: str, name: str):
        base = os.path.join(storage_dir, sid)
        if not os.path.exists(os.path.join(base, name)):
            # Debug aid: list directory contents when missing
            contents = []
            for root, _, files in os.walk(base):
                for f in files:
                    contents.append(os.path.relpath(os.path.join(root, f), base))
            return jsonify({"error": "file not found", "requested": name, "available": contents}), 404
        return send_from_directory(base, name)

    @app.get("/files/label_store/images/<sid>/<fname>")
    def label_store_files(sid: str, fname: str):
        base = os.path.join(label_images_root, sid)
        if not os.path.exists(os.path.join(base, fname)):
            return jsonify({"error": "file not found in label_store", "path": f"{sid}/{fname}"}), 404
        return send_from_directory(base, fname)

    @app.get("/api/label/sessions")
    def label_sessions():
        return jsonify({"sessions": _list_sessions_with_cells()})

    @app.post("/api/label/next")
    def label_next():
        data = request.get_json(silent=True) or {}
        sid = data.get("session")
        review_list = data.get("reviewList")  # Optional list of specific files to review
        if not sid:
            return jsonify({"error": "session required"}), 400
        cells_dir = os.path.join(storage_dir, sid, "extract_debug")
        if not os.path.isdir(cells_dir):
            return jsonify({"error": "extract_debug not found for session"}), 404
        labels = _load_labels(session_filter=sid)
        
        # If review list provided, use it; otherwise use all files
        if review_list:
            files = [f for f in review_list if os.path.exists(os.path.join(cells_dir, f))]
        else:
            files = sorted([f for f in os.listdir(cells_dir) if f.startswith("cell_") and f.endswith(".png")])
        
        labeled_names = {os.path.basename(k) for k in labels.keys()}
        remaining = [f for f in files if f not in labeled_names]
        if not remaining:
            return jsonify({"done": True, "total": len(files), "labeled": len(labels)})
        fname = remaining[0]
        
        # Include current label if exists for review mode
        current_label = None
        key = f"{sid}/{fname}"
        if key in labels:
            current_label = labels[key].get("label")
        
        return jsonify({
            "file": fname,
            "imageUrl": f"/files/{sid}/extract_debug/{fname}",
            "labeled": len(labels),
            "remaining": len(remaining),
            "total": len(files),
            "currentLabel": current_label,
        })

    @app.post("/api/label/save")
    def label_save():
        data = request.get_json(silent=True) or {}
        sid = data.get("session")
        fname = data.get("file")
        label = data.get("label")
        if not sid or not fname or label is None:
            return jsonify({"error": "session, file, and label are required"}), 400
        
        # Check both extract_debug and label_store/images locations
        cells_dir = os.path.join(storage_dir, sid, "extract_debug")
        label_store_dir = os.path.join(label_images_root, sid)
        
        source_path = None
        if os.path.exists(os.path.join(cells_dir, fname)):
            source_path = os.path.join(cells_dir, fname)
        elif os.path.exists(os.path.join(label_store_dir, fname)):
            source_path = os.path.join(label_store_dir, fname)
        else:
            return jsonify({"error": "cell file not found in extract_debug or label_store"}), 404
        
        label = str(label).strip()
        if not label:
            return jsonify({"error": "label cannot be empty"}), 400
        labels = _load_labels()
        key = f"{sid}/{fname}"
        
        # Copy image into central label store if not already there
        dest_dir = os.path.join(label_images_root, sid)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, fname)
        if not os.path.exists(dest_path):
            shutil.copyfile(source_path, dest_path)

        # Single manifest format: always include a usable CNN "path" and omit any split.
        labels[key] = {"file": key, "session": sid, "label": label, "path": os.path.abspath(dest_path)}
        _save_labels(labels)
        labeled_count = len([v for v in labels.values() if v.get("session") == sid])
        return jsonify({"ok": True, "labeled": labeled_count})

    @app.post("/api/label/detect-mislabels")
    def detect_mislabels_api():
        """Run mislabel detection using the CNN model."""
        try:
            from .cnn_ocr import Sample, load_manifest, normalize_label, predict_array
            from PIL import Image
            
            data = request.get_json(silent=True) or {}
            # Default to the label-store manifest produced by the labeling UI.
            # (Older versions expected a CNN training manifest like webdata/train_manifest.jsonl.)
            manifest_path = data.get("manifest") or _label_manifest_path()
            checkpoint_path = data.get("checkpoint", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/cnn_checkpoint/cnn_ocr.pt")))
            min_conf = float(data.get("minConf", 0.5))
            top_n = int(data.get("topN", 50))
            
            print(f"[detect-mislabels] manifest={manifest_path}, checkpoint={checkpoint_path}, min_conf={min_conf}")
            
            if not os.path.exists(checkpoint_path):
                return jsonify({"error": f"Checkpoint not found: {checkpoint_path}"}), 400
            if not os.path.exists(manifest_path):
                return jsonify({"error": f"Manifest not found: {manifest_path}"}), 400

            # Use cached model so a reload button affects inference without restarting.
            model, meta = _load_cnn_checkpoint(checkpoint_path, device="cpu")
            alphabet = meta.get("alphabet", [])
            device_name = meta.get("device")

            # Support both formats:
            # - CNN manifest: JSONL with {path,label,...}
            # - Label-store manifest: JSONL with {file:"<sid>/<fname>",label,split}
            samples = []
            try:
                samples = load_manifest(manifest_path)
            except Exception:
                samples = []
            if not samples:
                # Try label-store format.
                base_images = Path(label_images_root)
                with Path(manifest_path).open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue
                        key = entry.get("file")
                        lab = normalize_label(str(entry.get("label") or ""))
                        if not key or not lab:
                            continue
                        # key is like "<sid>/<fname>".
                        p = (base_images / str(key)).resolve()
                        if not p.exists():
                            continue
                        samples.append(Sample(path=str(p), label=lab))
            
            print(f"[detect-mislabels] Loaded {len(samples)} samples from manifest")
            
            suspicious: List[Dict[str, Any]] = []
            for i, sample in enumerate(samples):
                try:
                    img = Image.open(sample.path).convert("L")
                    arr = np.array(img)
                    pred_label, prob = predict_array(arr, model, alphabet=alphabet, device=device_name, image_size=32)
                    
                    if pred_label and pred_label != sample.label and prob >= min_conf:
                        # Extract session and filename from path
                        # Path format: .../label_store/images/{session_id}/{filename}
                        path_obj = Path(sample.path)
                        fname = path_obj.name
                        session_id = path_obj.parent.name
                        
                        suspicious.append({
                            "session": session_id,
                            "file": fname,
                            "path": sample.path,
                            "currentLabel": sample.label,
                            "predictedLabel": pred_label,
                            "confidence": float(prob),
                        })
                except Exception as e:
                    print(f"[detect-mislabels] Error processing {sample.path}: {e}")
                    continue
            
            suspicious.sort(key=lambda x: x["confidence"], reverse=True)
            suspicious = suspicious[:top_n]
            
            print(f"[detect-mislabels] Found {len(suspicious)} suspicious cases")
            
            return jsonify({
                "total": len(suspicious),
                "cases": suspicious,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.post("/api/label/reload-model")
    def reload_cnn_model_api():
        """Reload the CNN checkpoint into memory (no server restart)."""
        try:
            data = request.get_json(silent=True) or {}
            checkpoint_path = str(data.get("checkpoint") or _cnn_cache.get("checkpoint"))
            device = str(data.get("device") or "cpu")
            if not os.path.exists(checkpoint_path):
                return jsonify({"error": f"Checkpoint not found: {checkpoint_path}"}), 400

            model, meta = _load_cnn_checkpoint(checkpoint_path, device=device, force=True)
            cfg = meta.get("config") or {}
            return jsonify({
                "ok": True,
                "checkpoint": checkpoint_path,
                "device": meta.get("device"),
                "alphabetSize": len(meta.get("alphabet") or []),
                "imageSize": cfg.get("image_size"),
                "loadedAt": time.time(),
                "mtime": os.path.getmtime(checkpoint_path),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            with _cnn_lock:
                _cnn_cache["error"] = str(e)
            return jsonify({"error": str(e)}), 500

    @app.post("/api/label/regenerate-manifests")
    def regenerate_manifests_api():
        """Migrate label manifest to single-manifest format and archive old manifest files."""
        try:
            archive_dir = _archive_dir()

            archived: List[str] = []
            for old_name in ("train_manifest.jsonl", "val_manifest.jsonl", "manifest.jsonl"):
                moved = _archive_file(os.path.join(storage_dir, old_name), archive_dir)
                if moved:
                    archived.append(moved)

            migrated = _migrate_label_manifest_to_cnn_format(archive_dir=archive_dir)

            # Also remove any split fields when saving the next label.
            return jsonify({
                "ok": True,
                "archived": archived,
                "migrated": migrated,
                "labelManifest": _label_manifest_path(),
                "archiveDir": archive_dir,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.get("/api/train/status")
    def train_status_api():
        with _train_lock:
            return jsonify({
                "running": bool(_train_state.get("running")),
                "history": list(_train_state.get("history") or []),
                "logs": list(_train_state.get("logs") or []),
                "error": _train_state.get("error"),
                "exitCode": _train_state.get("exit_code"),
                "startedAt": _train_state.get("started_at"),
                "finishedAt": _train_state.get("finished_at"),
                "config": _train_state.get("config"),
            })

    @app.post("/api/train/start")
    def train_start_api():
        nonlocal _train_proc
        data = request.get_json(silent=True) or {}
        with _train_lock:
            if _train_proc is not None and _train_state.get("running"):
                return jsonify({"error": "Training already running"}), 400
            _train_state["history"] = []
            _train_state["logs"] = []
            _train_state["error"] = None
            _train_state["exit_code"] = None
            _train_state["started_at"] = time.time()
            _train_state["finished_at"] = None
            # Mark running immediately to avoid UI polling race.
            _train_state["running"] = True
            _train_state["logs"].append("[train] starting…")

        # Default to the single label manifest; valFraction is used to create an internal validation split.
        manifest = str(data.get("manifest") or _label_manifest_path())
        out_dir = str(data.get("outDir") or os.path.abspath(os.path.join(project_root, "models/cnn_checkpoint")))
        epochs = int(data.get("epochs", 15))
        batch_size = int(data.get("batchSize", 64))
        lr = float(data.get("lr", 1e-3))
        val_fraction = float(data.get("valFraction", 0.2))
        device = data.get("device")
        seed = int(data.get("seed", 1337))
        num_workers = int(data.get("numWorkers", 0))
        image_size = int(data.get("imageSize", 32))
        no_augment = bool(data.get("noAugment", False))
        aug_rotation = float(data.get("augRotation", 7.0))
        aug_translate_x = float(data.get("augTranslateX", 0.05))
        aug_translate_y = float(data.get("augTranslateY", 0.05))
        aug_scale_min = float(data.get("augScaleMin", 0.9))
        aug_scale_max = float(data.get("augScaleMax", 1.1))
        aug_multiplier = int(data.get("augMultiplier", 1))

        if not os.path.exists(manifest):
            return jsonify({"error": f"Manifest not found: {manifest}. Start labeling first."}), 400

        cmd: List[str] = [
            sys.executable,
            "-u",
            "-m",
            "scrabble.cnn_ocr",
            "train",
            "--manifest",
            manifest,
            "--out",
            out_dir,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--val-fraction",
            str(val_fraction),
            "--seed",
            str(seed),
            "--num-workers",
            str(num_workers),
            "--image-size",
            str(image_size),
            "--aug-rotation",
            str(aug_rotation),
            "--aug-translate-x",
            str(aug_translate_x),
            "--aug-translate-y",
            str(aug_translate_y),
            "--aug-scale-min",
            str(aug_scale_min),
            "--aug-scale-max",
            str(aug_scale_max),
            "--aug-multiplier",
            str(aug_multiplier),
        ]
        if device:
            cmd.extend(["--device", str(device)])
        if no_augment:
            cmd.append("--no-augment")

        with _train_lock:
            _train_state["config"] = {
                "manifest": manifest,
                "outDir": out_dir,
                "epochs": epochs,
                "batchSize": batch_size,
                "lr": lr,
                "valFraction": val_fraction,
                "device": device,
                "seed": seed,
                "numWorkers": num_workers,
                "imageSize": image_size,
                "noAugment": no_augment,
                "augRotation": aug_rotation,
                "augTranslateX": aug_translate_x,
                "augTranslateY": aug_translate_y,
                "augScaleMin": aug_scale_min,
                "augScaleMax": aug_scale_max,
                "augMultiplier": aug_multiplier,
            }

        t = threading.Thread(target=_training_thread, args=(cmd,), daemon=True)
        t.start()
        return jsonify({"ok": True, "cmd": cmd, "config": _train_state["config"]})

    @app.post("/api/train/stop")
    def train_stop_api():
        nonlocal _train_proc
        with _train_lock:
            proc = _train_proc
        if proc is None:
            return jsonify({"ok": True, "stopped": False, "message": "No training process running"})
        try:
            proc.terminate()
            return jsonify({"ok": True, "stopped": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8765, debug=True)
