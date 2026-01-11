import time
from typing import List, Optional, Sequence, Tuple, Dict, Any

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


def _order_points(pts_: np.ndarray) -> np.ndarray:
    # Order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_.sum(axis=1)
    rect[0] = pts_[np.argmin(s)]
    rect[2] = pts_[np.argmax(s)]
    diff = np.diff(pts_, axis=1)
    rect[1] = pts_[np.argmin(diff)]
    rect[3] = pts_[np.argmax(diff)]
    return rect


def _edges_for_grid(length: int, board_size: int = 15) -> List[int]:
    # Compute integer edges using rounding to distribute remainder evenly
    return [int(round(i * (length / board_size))) for i in range(board_size + 1)]


def _slice_board_grid(img: np.ndarray, board_size: int = 15, inner_margin: float = 0.18, outer_margin: float = 0.0) -> List[List[np.ndarray]]:
    # Slice the warped board into cells using rounded edges across the full extent
    h, w = img.shape[:2]
    y_edges = _edges_for_grid(h, board_size)
    x_edges = _edges_for_grid(w, board_size)
    cells: List[List[np.ndarray]] = []
    for r in range(board_size):
        row: List[np.ndarray] = []
        y0, y1 = y_edges[r], y_edges[r + 1]
        ch = y1 - y0
        for c in range(board_size):
            x0, x1 = x_edges[c], x_edges[c + 1]
            cw = x1 - x0
            # Apply outer margin (expand beyond cell boundary)
            outer_h = int(ch * outer_margin)
            outer_w = int(cw * outer_margin)
            y0_outer = max(0, y0 - outer_h)
            y1_outer = min(h, y1 + outer_h)
            x0_outer = max(0, x0 - outer_w)
            x1_outer = min(w, x1 + outer_w)
            cell = img[y0_outer:y1_outer, x0_outer:x1_outer]
            # Center-crop by inner_margin percentage on each side
            ch_cell, cw_cell = cell.shape[:2]
            m_h = int(ch_cell * inner_margin)
            m_w = int(cw_cell * inner_margin)
            crop = cell[m_h : max(m_h, ch_cell - m_h), m_w : max(m_w, cw_cell - m_w)]
            row.append(crop)
        cells.append(row)
    return cells


def _prep_cell_image(
    cell: np.ndarray,
    apply_binarize: bool = False,
    isolate_main_blob: bool = False,
    debug_images: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if isolate_main_blob:
        # Keep only the largest contour (assumed main character) to drop smaller digits/marks
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main)
            if w > 0 and h > 0:
                mask = np.zeros_like(bin_inv)
                cv2.drawContours(mask, [main], -1, 255, thickness=cv2.FILLED)
                masked = cv2.bitwise_and(gray, gray, mask=mask)
                gray = masked[y:y + h, x:x + w]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    scale = 2
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    if apply_binarize:
        _, gray_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray_bin = cv2.morphologyEx(gray_bin, cv2.MORPH_OPEN, kernel)
        gray = gray_bin
    if debug_images is not None:
        debug_images["processed"] = gray.copy()
    return gray


def _ocr_cell_tesseract(
    cell: np.ndarray,
    min_conf: float = 75.0,
    psm: int = 10,
    psm_fallback: Optional[int] = 8,
    apply_binarize: bool = False,
    isolate_main_blob: bool = False,
    debug_images: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[str]:
    if pytesseract is None:
        return None
    gray = _prep_cell_image(
        cell,
        apply_binarize=apply_binarize,
        isolate_main_blob=isolate_main_blob,
        debug_images=debug_images,
    )

    # Try multiple thresholding methods; pick best by confidence
    variants = []
    _, th_otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th_mean_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    th_gauss_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    variants.extend([th_otsu_inv, th_mean_inv, th_gauss_inv])

    def _try_psm(img: np.ndarray, p: int) -> Tuple[Optional[str], float]:
        cfg = f"--psm {p} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        best_char: Optional[str] = None
        best_conf: float = -1.0
        try:
            data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
            texts = data.get("text", [])
            confs = data.get("conf", [])
            for t, c in zip(texts, confs):
                if not t:
                    continue
                t = t.strip().upper()
                try:
                    cval = float(c)
                except Exception:
                    cval = -1.0
                if len(t) == 1 and 'A' <= t <= 'Z' and cval >= min_conf and cval > best_conf:
                    best_char = t
                    best_conf = cval
        except Exception:
            # Fallback to image_to_string with relaxed conf
            try:
                txt = pytesseract.image_to_string(img, config=cfg).strip().upper()
                if len(txt) == 1 and 'A' <= txt <= 'Z' and min_conf <= 60.0:
                    best_char = txt
                    best_conf = max(best_conf, 60.0)
            except Exception:
                pass
        return best_char, best_conf

    best_char: Optional[str] = None
    best_conf: float = -1.0
    for img in variants:
        ch, conf = _try_psm(img, psm)
        if conf > best_conf:
            best_char, best_conf = ch, conf
        if best_char is None and psm_fallback is not None:
            ch2, conf2 = _try_psm(img, psm_fallback)
            if conf2 > best_conf:
                best_char, best_conf = ch2, conf2
    return best_char


def _ocr_cell_cnn(
    cell: np.ndarray,
    cnn_model: Any,
    cnn_device: Optional[str] = None,
    cnn_alphabet: Optional[Sequence[str]] = None,
    min_conf: float = 0.0,
    apply_binarize: bool = False,
    isolate_main_blob: bool = False,
    debug_images: Optional[Dict[str, np.ndarray]] = None,
    cnn_image_size: int = 32,
) -> Optional[str]:
    if cnn_model is None:
        raise ValueError("cnn_model is required when engine='cnn'")
    try:
        from . import cnn_ocr
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("cnn_ocr module not available; install with `python -m pip install \"scrabble-ai[cnn]\"`") from exc
    processed = _prep_cell_image(
        cell,
        apply_binarize=apply_binarize,
        isolate_main_blob=isolate_main_blob,
        debug_images=debug_images,
    )
    alphabet = list(cnn_alphabet) if cnn_alphabet is not None else cnn_ocr.ALPHABET
    label, prob = cnn_ocr.predict_array(
        processed,
        cnn_model,
        alphabet=alphabet,
        device=cnn_device,
        min_confidence=min_conf,
        image_size=cnn_image_size,
    )
    if label == '.':
        return None
    return label


def preprocess_board_image(
    path: str,
    board_size: int = 15,
    max_side: int = 1500,
    return_debug: bool = False,
    detection_method: str = "auto",
    min_area_frac: float = 0.10,
    aspect_tol: float = 0.35,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    # Resize for speed if very large
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (img, {"source_with_quad": img.copy()}) if return_debug else img

    H, W = img.shape[:2]
    img_area = float(H * W)

    def _try_contour_quad() -> Optional[np.ndarray]:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < min_area_frac * img_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Check aspect ratio near square
                rect = cv2.minAreaRect(approx)
                (w_box, h_box) = rect[1]
                if w_box == 0 or h_box == 0:
                    continue
                ratio = max(w_box, h_box) / max(1e-6, min(w_box, h_box))
                if (1 - aspect_tol) <= ratio <= (1 + aspect_tol):
                    return approx
        return None

    def _try_min_area_rect() -> Optional[np.ndarray]:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        area = cv2.contourArea(box)
        if area < min_area_frac * img_area:
            return None
        return box.reshape(-1, 1, 2)

    def _try_hough_lines() -> Optional[np.ndarray]:
        # Robust Hough lines (rho, theta) approach with binarization and morphological cleanup
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -5)
        # Emphasize grid lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        lines_h_img = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel_h)
        lines_v_img = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel_v)

        def _hough_extremes(bin_img: np.ndarray, orient: str) -> Optional[Tuple[float, float]]:
            lines = cv2.HoughLines(bin_img, 1, np.pi / 180, threshold=150)
            if lines is None:
                return None
            ys = []
            xs = []
            for rho_theta in lines:
                rho, theta = rho_theta[0]
                # Normalize angles
                if orient == 'h':
                    # near horizontal -> theta around 0 or pi
                    if not (theta < np.deg2rad(15) or theta > np.pi - np.deg2rad(15)):
                        continue
                    # y = rho/sin(theta) at x=0
                    s = np.sin(theta)
                    if abs(s) < 1e-6:
                        continue
                    y = rho / s
                    ys.append(y)
                else:
                    # near vertical -> theta around pi/2
                    if not (abs(theta - np.pi / 2) < np.deg2rad(15)):
                        continue
                    c = np.cos(theta)
                    if abs(c) < 1e-6:
                        continue
                    x = rho / c
                    xs.append(x)
            if orient == 'h' and ys:
                return (min(ys), max(ys))
            if orient == 'v' and xs:
                return (min(xs), max(xs))
            return None

        h_ext = _hough_extremes(lines_h_img, 'h')
        v_ext = _hough_extremes(lines_v_img, 'v')
        if h_ext is None or v_ext is None:
            return None
        top_y, bot_y = h_ext
        left_x, right_x = v_ext
        # Clamp to image bounds
        top_y = float(max(0, min(H - 1, int(round(top_y)))))
        bot_y = float(max(0, min(H - 1, int(round(bot_y)))))
        left_x = float(max(0, min(W - 1, int(round(left_x)))))
        right_x = float(max(0, min(W - 1, int(round(right_x)))))
        pts = np.array([[left_x, top_y], [right_x, top_y], [right_x, bot_y], [left_x, bot_y]], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    quad = None
    chosen_method = None

    methods = []
    if detection_method in ("auto", "contour"):
        methods.append(("contour", _try_contour_quad))
    if detection_method in ("auto", "rect"):
        methods.append(("rect", _try_min_area_rect))
    if detection_method in ("auto", "hough"):
        methods.append(("hough", _try_hough_lines))
    if detection_method in ("auto", "proj"):
        def _try_projection_profiles() -> Optional[np.ndarray]:
            grayp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thp = cv2.adaptiveThreshold(grayp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -5)
            # Sum along axes
            col_sum = thp.sum(axis=0)
            row_sum = thp.sum(axis=1)
            # Threshold as top percentile to find grid lines
            import numpy as _np
            cs_thr = _np.percentile(col_sum, 95)
            rs_thr = _np.percentile(row_sum, 95)
            cols = [i for i, v in enumerate(col_sum) if v >= cs_thr]
            rows = [i for i, v in enumerate(row_sum) if v >= rs_thr]
            if not cols or not rows:
                return None
            left_x, right_x = float(min(cols)), float(max(cols))
            top_y, bot_y = float(min(rows)), float(max(rows))
            pts = np.array([[left_x, top_y], [right_x, top_y], [right_x, bot_y], [left_x, bot_y]], dtype=np.float32)
            return pts.reshape(-1, 1, 2)

        methods.append(("proj", _try_projection_profiles))

    for name, fn in methods:
        quad = fn()
        if quad is not None:
            chosen_method = name
            break
    if quad is None:
        return (img, {"source_with_quad": img.copy(), "detection_method": "none"}) if return_debug else img

    pts = quad.reshape(4, 2).astype(np.float32)

    rect = _order_points(pts)
    src_with_quad = img.copy()
    try:
        cv2.polylines(src_with_quad, [quad.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)
    except Exception:
        pass
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
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # Normalize to a square for consistent slicing
    side = max(maxWidth, maxHeight)
    warped = cv2.resize(warped, (side, side), interpolation=cv2.INTER_AREA)
    if return_debug:
        return warped, {
            "source_with_quad": src_with_quad,
            "detection_method": chosen_method or "unknown",
            "quad_points": rect.astype(np.float32),
        }
    return warped


def _refine_corners_with_hints(img: np.ndarray, hints: List[Tuple[float, float]], max_dist: float = 60.0) -> np.ndarray:
    # Try to snap user-provided corner hints to nearest strong corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=7)
    pts = []
    if corners is None:
        return np.array(hints, dtype=np.float32)
    cand = corners.reshape(-1, 2)
    for (hx, hy) in hints:
        d = np.sqrt(((cand[:, 0] - hx) ** 2) + ((cand[:, 1] - hy) ** 2))
        idx = int(np.argmin(d))
        if d[idx] <= max_dist:
            pts.append((float(cand[idx, 0]), float(cand[idx, 1])))
        else:
            pts.append((float(hx), float(hy)))
    return np.array(pts, dtype=np.float32)


def ocr_board_image(
    path: str,
    engine: str = "tesseract",
    debug_dir: Optional[str] = None,
    inner_margin: float = 0.18,
    min_conf: float = 75.0,
    detection_method: str = "auto",
    min_area_frac: float = 0.10,
    aspect_tol: float = 0.35,
    corner_hints: Optional[List[Tuple[float, float]]] = None,
    psm: int = 10,
    psm_fallback: Optional[int] = 8,
    apply_binarize: bool = False,
    isolate_main_blob: bool = False,
    cnn_model: Any = None,
    cnn_alphabet: Optional[Sequence[str]] = None,
    cnn_device: Optional[str] = None,
    cnn_min_conf: float = 0.0,
    cnn_image_size: int = 32,
) -> List[List[Optional[str]]]:
    # Preprocess (find and warp board), then slice
    dbg = None
    if corner_hints:
        src = cv2.imread(path)
        if src is None:
            raise FileNotFoundError(path)
        refined = _refine_corners_with_hints(src, corner_hints)
        rect = _order_points(refined)
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
        side = max(maxWidth, maxHeight)
        img = cv2.resize(warped, (side, side), interpolation=cv2.INTER_AREA)
        if debug_dir:
            try:
                import os
                os.makedirs(debug_dir, exist_ok=True)
                src_copy = src.copy()
                for (x, y) in refined.astype(int):
                    cv2.circle(src_copy, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.polylines(src_copy, [refined.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
                cv2.imwrite(os.path.join(debug_dir, "source_with_hints.png"), src_copy)
                with open(os.path.join(debug_dir, "detection_method.txt"), "w", encoding="utf-8") as f:
                    f.write("hints_refined")
            except Exception:
                pass
    else:
        if debug_dir:
            pre = preprocess_board_image(
                path,
                return_debug=True,
                detection_method=detection_method,
                min_area_frac=min_area_frac,
                aspect_tol=aspect_tol,
            )
            img, dbg = pre  # type: ignore[assignment]
        else:
            img = preprocess_board_image(
                path,
                return_debug=False,
                detection_method=detection_method,
                min_area_frac=min_area_frac,
                aspect_tol=aspect_tol,
            )
    cells = _slice_board_grid(img, 15, inner_margin=inner_margin)
    letters: List[List[Optional[str]]] = []
    proc_images: List[List[Optional[np.ndarray]]] = []
    for row in cells:
        out_row: List[Optional[str]] = []
        proc_row: List[Optional[np.ndarray]] = []
        for cell in row:
            ch: Optional[str] = None
            cell_dbg: Optional[Dict[str, Any]] = {} if debug_dir else None
            if engine == "tesseract":
                ch = _ocr_cell_tesseract(
                    cell,
                    min_conf=min_conf,
                    psm=psm,
                    psm_fallback=psm_fallback,
                    apply_binarize=apply_binarize,
                    isolate_main_blob=isolate_main_blob,
                    debug_images=cell_dbg,
                )
            elif engine == "cnn":
                ch = _ocr_cell_cnn(
                    cell,
                    cnn_model=cnn_model,
                    cnn_device=cnn_device,
                    cnn_alphabet=cnn_alphabet,
                    min_conf=cnn_min_conf,
                    apply_binarize=apply_binarize,
                    isolate_main_blob=isolate_main_blob,
                    debug_images=cell_dbg,
                    cnn_image_size=cnn_image_size,
                )
            else:
                raise ValueError(f"Unsupported OCR engine: {engine}")
            out_row.append(ch)
            proc_row.append(cell_dbg.get("processed") if cell_dbg is not None else None)
        letters.append(out_row)
        proc_images.append(proc_row)
    if debug_dir:
        try:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            for r, row in enumerate(cells):
                for c, cell in enumerate(row):
                    proc = proc_images[r][c]
                    cv2.imwrite(os.path.join(debug_dir, f"cell_{r:02d}_{c:02d}.png"), proc if proc is not None else cell)
            # Save an overview image as well
            cv2.imwrite(os.path.join(debug_dir, "warped_board.png"), img)
            # Save the source image annotated with detected quadrilateral
            if isinstance(dbg, dict) and "source_with_quad" in dbg:
                cv2.imwrite(os.path.join(debug_dir, "source_with_quad.png"), dbg["source_with_quad"])
                # Also record which detection method was used
                try:
                    with open(os.path.join(debug_dir, "detection_method.txt"), "w", encoding="utf-8") as f:
                        f.write(str(dbg.get("detection_method", "unknown")))
                except Exception:
                    pass
            # Save a grid overlay for visual verification
            overlay = img.copy()
            h, w = overlay.shape[:2]
            y_edges = _edges_for_grid(h, 15)
            x_edges = _edges_for_grid(w, 15)
            for y in y_edges:
                y = min(max(0, y), h - 1)
                cv2.line(overlay, (0, y), (w - 1, y), (255, 0, 0), 1)
            for x in x_edges:
                x = min(max(0, x), w - 1)
                cv2.line(overlay, (x, 0), (x, h - 1), (255, 0, 0), 1)
            # Add labels: rows A-O on left, cols 1-15 on top
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, y in enumerate(y_edges[:-1]):
                yy = min(max(0, int(round((y + y_edges[i + 1]) / 2))), h - 1)
                cv2.putText(overlay, chr(ord('A') + i), (5, yy), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            for j, x in enumerate(x_edges[:-1]):
                xx = min(max(0, int(round((x + x_edges[j + 1]) / 2))), w - 1)
                cv2.putText(overlay, str(j + 1), (xx, 15), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(debug_dir, "grid_overlay.png"), overlay)
        except Exception:
            pass
    return letters


def letters_grid_to_board_string(letters: List[List[Optional[str]]]) -> str:
    rows: List[str] = []
    for row in letters:
        rows.append("".join(ch if ch is not None else '.' for ch in row))
    return "\n".join(rows)


def extract_board_from_warped_image(
    path: str,
    inner_margin: float = 0.18,
    outer_margin: float = 0.0,
    min_conf: float = 75.0,
    psm: int = 10,
    psm_fallback: Optional[int] = 8,
    apply_binarize: bool = False,
    isolate_main_blob: bool = False,
    debug_dir: Optional[str] = None,
    collect_timings: bool = False,
    engine: str = "tesseract",
    cnn_model: Any = None,
    cnn_alphabet: Optional[Sequence[str]] = None,
    cnn_device: Optional[str] = None,
    cnn_min_conf: float = 0.0,
    cnn_image_size: int = 32,
) -> Tuple[List[List[Optional[str]]], str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """OCR a board from an already-warped image path and return grid + board string.

    The input image should be a top-down warp of the Scrabble board. This function
    slices it into a 15x15 grid, OCRs each cell (Tesseract by default), and returns
    both the letters grid and the 15-line board string representation.
    
    Args:
        inner_margin: Crop margin inside each cell (0-0.45, removes edges/grid lines)
        outer_margin: Expand margin outside each cell (0-0.2, captures more context)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    # Normalize to square for consistent slicing
    h, w = img.shape[:2]
    side = max(h, w)
    if (h != side) or (w != side):
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    cells = _slice_board_grid(img, 15, inner_margin=inner_margin, outer_margin=outer_margin)
    letters: List[List[Optional[str]]] = []
    last_preview: Optional[Dict[str, Any]] = None
    first_non_empty: Optional[Dict[str, Any]] = None
    preview_list: List[Dict[str, Any]] = []
    proc_images: List[List[Optional[np.ndarray]]] = []
    timing_cells: List[Dict[str, Any]] = []
    for r_idx, row in enumerate(cells):
        out_row: List[Optional[str]] = []
        proc_row: List[Optional[np.ndarray]] = []
        for c_idx, cell in enumerate(row):
            cell_dbg: Dict[str, Any] = {}
            t0 = time.perf_counter()
            if engine == "tesseract":
                ch = _ocr_cell_tesseract(
                    cell,
                    min_conf=min_conf,
                    psm=psm,
                    psm_fallback=psm_fallback,
                    apply_binarize=apply_binarize,
                    isolate_main_blob=isolate_main_blob,
                    debug_images=cell_dbg,
                )
            elif engine == "cnn":
                ch = _ocr_cell_cnn(
                    cell,
                    cnn_model=cnn_model,
                    cnn_device=cnn_device,
                    cnn_alphabet=cnn_alphabet,
                    min_conf=cnn_min_conf,
                    apply_binarize=apply_binarize,
                    isolate_main_blob=isolate_main_blob,
                    debug_images=cell_dbg,
                    cnn_image_size=cnn_image_size,
                )
            else:
                raise ValueError(f"Unsupported OCR engine: {engine}")
            if collect_timings:
                timing_cells.append({
                    "row": r_idx,
                    "col": c_idx,
                    "ms": (time.perf_counter() - t0) * 1000.0,
                })
            last_preview = {"row": r_idx, "col": c_idx, "char": ch}
            if ch and first_non_empty is None:
                first_non_empty = {"row": r_idx, "col": c_idx, "char": ch}
            out_row.append(ch)
            preview_list.append({"row": r_idx, "col": c_idx, "char": ch})
            proc_row.append(cell_dbg.get("processed"))
        letters.append(out_row)
        proc_images.append(proc_row)
    if debug_dir:
        try:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "warped_board.png"), img)
            for r, row in enumerate(cells):
                for c, cell in enumerate(row):
                    fname = f"cell_{r:02d}_{c:02d}.png"
                    img_to_save = proc_images[r][c] if proc_images[r][c] is not None else cell
                    cv2.imwrite(os.path.join(debug_dir, fname), img_to_save)
                    if last_preview and last_preview.get("row") == r and last_preview.get("col") == c:
                        last_preview["file"] = fname
                    if first_non_empty and first_non_empty.get("row") == r and first_non_empty.get("col") == c:
                        first_non_empty["file"] = fname
                    # Attach filenames to preview list entries
                    for p in preview_list:
                        if p.get("row") == r and p.get("col") == c:
                            p["file"] = fname
        except Exception:
            pass
    board_str = letters_grid_to_board_string(letters)
    return letters, board_str, last_preview, first_non_empty, preview_list, timing_cells

