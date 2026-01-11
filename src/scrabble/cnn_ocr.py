from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image

try:  # Optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception:  # pragma: no cover - handled lazily
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore
    transforms = None  # type: ignore


ALPHABET: List[str] = ['.'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ?")
LABEL_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_LABEL: Dict[int, str] = {i: c for i, c in enumerate(ALPHABET)}


def _require_torch() -> None:
    if torch is None or nn is None or Dataset is None or transforms is None:  # type: ignore
        raise ImportError("Torch not available; install with `python -m pip install \"scrabble-ai[cnn]\"`")


@dataclass
class Sample:
    path: str
    label: str
    row: Optional[int] = None
    col: Optional[int] = None


def normalize_label(ch: str) -> Optional[str]:
    if not ch:
        return None
    ch = ch.strip()
    if not ch:
        return None
    if ch == '.':
        return '.'
    if ch == '?':
        return '?'
    if len(ch) == 1:
        up = ch.upper()
        if 'A' <= up <= 'Z':
            return up
    return None


def board_string_to_grid(board_string: str) -> List[List[str]]:
    rows_raw = [r for r in board_string.strip().splitlines() if r.strip()]
    if len(rows_raw) != 15:
        raise ValueError(f"Board string must have 15 rows, got {len(rows_raw)}")
    grid: List[List[str]] = []
    for idx, row in enumerate(rows_raw):
        if len(row) != 15:
            raise ValueError(f"Row {idx} has length {len(row)}; expected 15")
        grid.append(list(row))
    return grid


def build_manifest_from_board_string(
    board_string: str,
    session_dir: str,
    manifest_path: str,
    include_empty: bool = False,
) -> Dict[str, Any]:
    grid = board_string_to_grid(board_string)
    session = Path(session_dir)
    entries: List[Dict[str, Any]] = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            norm = normalize_label(ch)
            if norm is None:
                continue
            if norm == '.' and not include_empty:
                continue
            fname = f"cell_{r:02d}_{c:02d}.png"
            fpath = session / fname
            if not fpath.exists():
                continue
            entry = {
                "path": str(fpath.resolve()),
                "label": norm,
                "row": r,
                "col": c,
            }
            entries.append(entry)
    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return {"written": len(entries), "path": str(out)}


def _resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(manifest_path: str) -> List[Sample]:
    base = Path(manifest_path).parent
    samples: List[Sample] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            path = _resolve_path(base, data["path"])
            label = normalize_label(str(data.get("label", "")) or '')
            if label is None:
                continue
            samples.append(Sample(path=str(path), label=label, row=data.get("row"), col=data.get("col")))
    return samples


def export_label_splits(
    label_manifest: str,
    images_dir: str,
    out_train: str,
    out_val: str,
    val_fraction: float = 0.2,
    seed: int = 1337,
    append_existing: bool = True,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    images_path = Path(images_dir)
    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []
    # If appending, preload existing manifests to allow incremental labeling
    if append_existing:
        for path_out, bucket in ((out_train, train_entries), (out_val, val_entries)):
            p = Path(path_out)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            bucket.append(json.loads(line))
                        except Exception:
                            continue
    # Track seen paths to avoid duplicates when extending
    seen_train = {e.get("path") for e in train_entries}
    seen_val = {e.get("path") for e in val_entries}
    total = 0
    with Path(label_manifest).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            fname = data.get("file")
            label = normalize_label(str(data.get("label", "")) or '')
            if fname is None or label is None:
                continue
            split = (data.get("split") or "").lower()
            path = images_path / fname
            if not path.exists():
                continue
            resolved_path = str(path.resolve())
            entry = {
                "path": resolved_path,
                "label": label,
                "row": data.get("row"),
                "col": data.get("col"),
            }
            target_list = train_entries
            seen = seen_train
            if split == "val":
                target_list = val_entries
                seen = seen_val
            elif split == "train":
                target_list = train_entries
                seen = seen_train
            else:
                target_list = val_entries if rng.random() < val_fraction else train_entries
                seen = seen_val if target_list is val_entries else seen_train
            if resolved_path in seen:
                continue
            target_list.append(entry)
            seen.add(resolved_path)
    def _write(out_path: str, entries: List[Dict[str, Any]]) -> None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fw:
            for e in entries:
                fw.write(json.dumps(e) + "\n")
    _write(out_train, train_entries)
    _write(out_val, val_entries)
    return {
        "total_label_lines": total,
        "train": len(train_entries),
        "val": len(val_entries),
        "train_manifest": str(Path(out_train).resolve()),
        "val_manifest": str(Path(out_val).resolve()),
    }


class CellDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 32,
        augment: bool = False,
        aug_rotation: float = 7.0,
        aug_translate: Tuple[float, float] = (0.05, 0.05),
        aug_scale: Tuple[float, float] = (0.9, 1.1),
        aug_multiplier: int = 1,
    ) -> None:
        _require_torch()
        self.samples = list(samples) * aug_multiplier  # Repeat samples for augmentation
        self.image_size = image_size
        aug: List[Any] = []
        if augment:
            aug = [
                transforms.RandomRotation(aug_rotation),
                transforms.RandomAffine(degrees=0, translate=aug_translate, scale=aug_scale),
            ]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("L")
        tensor = self.transform(img)
        label_idx = LABEL_TO_IDX[sample.label]
        return tensor, label_idx


def _stratified_split(samples: Sequence[Sample], val_fraction: float, seed: int = 1337) -> Tuple[List[Sample], List[Sample]]:
    random.seed(seed)
    buckets: Dict[str, List[Sample]] = {c: [] for c in ALPHABET}
    for s in samples:
        buckets.setdefault(s.label, []).append(s)
    train: List[Sample] = []
    val: List[Sample] = []
    for bucket in buckets.values():
        if not bucket:
            continue
        bucket_copy = bucket[:]
        random.shuffle(bucket_copy)
        if len(bucket_copy) == 1:
            train.extend(bucket_copy)
            continue
        cut = max(1, int(round(len(bucket_copy) * val_fraction)))
        if cut >= len(bucket_copy):
            cut = len(bucket_copy) - 1
        val.extend(bucket_copy[:cut])
        train.extend(bucket_copy[cut:])
    if not train or not val:
        raise ValueError("Not enough samples to split into train/val; add more labeled cells")
    random.shuffle(train)
    random.shuffle(val)
    return train, val


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Any) -> Any:
        feats = self.net(x)
        return self.head(feats)


def _pick_device(preferred: Optional[str] = None) -> str:
    _require_torch()
    if preferred:
        return preferred
    return "cuda" if torch and torch.cuda.is_available() else "cpu"  # type: ignore


def train_from_manifest(
    manifest_path: str,
    out_dir: str,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    device: Optional[str] = None,
    seed: int = 1337,
    num_workers: int = 0,
    image_size: int = 32,
    augment: bool = True,
    aug_rotation: float = 7.0,
    aug_translate: Tuple[float, float] = (0.05, 0.05),
    aug_scale: Tuple[float, float] = (0.9, 1.1),
    aug_multiplier: int = 1,
) -> Dict[str, Any]:
    _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    samples = load_manifest(manifest_path)
    if not samples:
        raise ValueError("Manifest has no labeled samples")
    train_samples, val_samples = _stratified_split(samples, val_fraction, seed=seed)
    train_ds = CellDataset(
        train_samples,
        image_size=image_size,
        augment=augment,
        aug_rotation=aug_rotation,
        aug_translate=aug_translate,
        aug_scale=aug_scale,
        aug_multiplier=aug_multiplier,
    )
    val_ds = CellDataset(val_samples, image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SmallCNN(num_classes=len(ALPHABET))
    device_name = _pick_device(device)
    model.to(device_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "cnn_ocr.pt"
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device_name)
            targets = targets.to(device_name)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device_name)
                targets = targets.to(device_name)
                logits = model(inputs)
                _, preds = torch.max(logits, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()
        val_acc = val_correct / max(1, val_total)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"[cnn-train] epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "alphabet": ALPHABET,
                "config": {
                    "image_size": image_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                },
            }, ckpt_path)
    return {"best_acc": best_acc, "checkpoint": str(ckpt_path), "history": history}


def load_checkpoint(path: str, device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    _require_torch()
    device_name = _pick_device(device)
    data = torch.load(path, map_location=device_name)
    alphabet = data.get("alphabet", ALPHABET)
    model = SmallCNN(num_classes=len(alphabet))
    model.load_state_dict(data["model_state"])
    model.to(device_name)
    model.eval()
    return model, {"alphabet": alphabet, "config": data.get("config", {}), "device": device_name}


def tensor_from_array(arr: np.ndarray, image_size: int = 32) -> Any:
    _require_torch()
    if arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))  # type: ignore[name-defined]
    else:
        img = Image.fromarray(arr)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(img).unsqueeze(0)


def predict_array(
    arr: np.ndarray,
    model: Any,
    alphabet: Sequence[str] = ALPHABET,
    device: Optional[str] = None,
    min_confidence: float = 0.0,
    image_size: int = 32,
) -> Tuple[Optional[str], float]:
    _require_torch()
    device_name = _pick_device(device)
    tensor = tensor_from_array(arr, image_size=image_size).to(device_name)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        prob, idx = torch.max(probs, dim=1)
        p_val = prob.item()
        label = alphabet[idx.item()]
        if p_val < min_confidence:
            return None, p_val
        return label, p_val


def predict_paths(
    paths: Iterable[str],
    checkpoint: str,
    device: Optional[str] = None,
    min_confidence: float = 0.0,
    image_size: int = 32,
) -> List[Dict[str, Any]]:
    model, meta = load_checkpoint(checkpoint, device=device)
    alphabet = meta.get("alphabet", ALPHABET)
    device_name = meta.get("device")
    results: List[Dict[str, Any]] = []
    for p in paths:
        img = Image.open(p).convert("L")
        arr = np.array(img)
        label, prob = predict_array(arr, model, alphabet=alphabet, device=device_name, min_confidence=min_confidence, image_size=image_size)
        results.append({"path": p, "label": label, "prob": prob})
    return results


def _cmd_build_manifest(args: argparse.Namespace) -> None:
    with open(args.board_string, "r", encoding="utf-8") as f:
        board_str = f.read()
    res = build_manifest_from_board_string(board_str, args.session_dir, args.out, include_empty=args.include_empty)
    print(json.dumps(res, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    res = train_from_manifest(
        args.manifest,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=not args.no_augment,
        aug_rotation=args.aug_rotation,
        aug_translate=(args.aug_translate_x, args.aug_translate_y),
        aug_scale=(args.aug_scale_min, args.aug_scale_max),
        aug_multiplier=args.aug_multiplier,
    )
    print(json.dumps(res, indent=2))


def _cmd_validate(args: argparse.Namespace) -> None:
    _require_torch()
    samples = load_manifest(args.manifest)
    model, meta = load_checkpoint(args.checkpoint, device=args.device)
    alphabet = meta.get("alphabet", ALPHABET)
    device_name = meta.get("device")
    ds = CellDataset(samples, image_size=args.image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device_name)
            targets = targets.to(device_name)
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
    acc = correct / max(1, total)
    print(json.dumps({"accuracy": acc, "samples": total, "alphabet": alphabet}, indent=2))


def _cmd_infer(args: argparse.Namespace) -> None:
    paths: List[str] = []
    if args.path:
        paths.append(args.path)
    if args.dir:
        for p in sorted(Path(args.dir).glob("*.png")):
            paths.append(str(p))
    if not paths:
        raise SystemExit("No input provided; use --path or --dir")
    results = predict_paths(paths, args.checkpoint, device=args.device, min_confidence=args.min_conf, image_size=args.image_size)
    print(json.dumps(results, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Tiny CNN for Scrabble tile OCR")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser("build-manifest", help="Write JSONL manifest from board string + session dir")
    p_manifest.add_argument("--board-string", required=True, help="Path to text file with 15x15 board string")
    p_manifest.add_argument("--session-dir", required=True, help="Directory containing cell_XX_YY.png files")
    p_manifest.add_argument("--out", required=True, help="Output manifest path (JSONL)")
    p_manifest.add_argument("--include-empty", action="store_true", help="Include '.' cells labeled as '.'")
    p_manifest.set_defaults(func=_cmd_build_manifest)

    p_train = sub.add_parser("train", help="Train CNN from manifest")
    p_train.add_argument("--manifest", required=True, help="JSONL manifest path")
    p_train.add_argument("--out", required=True, help="Output directory for checkpoint")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--device", type=str, help="Torch device, e.g., cpu or cuda")
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--image-size", type=int, default=32)
    p_train.add_argument("--no-augment", action="store_true", help="Disable training augmentations")
    p_train.add_argument("--aug-rotation", type=float, default=7.0, help="Max rotation degrees for augmentation (default: 7.0)")
    p_train.add_argument("--aug-translate-x", type=float, default=0.05, help="Max horizontal translation fraction (default: 0.05)")
    p_train.add_argument("--aug-translate-y", type=float, default=0.05, help="Max vertical translation fraction (default: 0.05)")
    p_train.add_argument("--aug-scale-min", type=float, default=0.9, help="Min scale factor for augmentation (default: 0.9)")
    p_train.add_argument("--aug-scale-max", type=float, default=1.1, help="Max scale factor for augmentation (default: 1.1)")
    p_train.add_argument("--aug-multiplier", type=int, default=1, help="Number of augmented copies per sample (default: 1)")
    p_train.set_defaults(func=_cmd_train)

    p_val = sub.add_parser("validate", help="Validate a checkpoint against a manifest")
    p_val.add_argument("--manifest", required=True)
    p_val.add_argument("--checkpoint", required=True)
    p_val.add_argument("--batch-size", type=int, default=64)
    p_val.add_argument("--device", type=str)
    p_val.add_argument("--num-workers", type=int, default=0)
    p_val.add_argument("--image-size", type=int, default=32)
    p_val.set_defaults(func=_cmd_validate)

    p_infer = sub.add_parser("infer", help="Run inference on a path or directory of cell images")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--path", help="Single image path")
    p_infer.add_argument("--dir", help="Directory of cell_XX_YY.png images")
    p_infer.add_argument("--device", type=str)
    p_infer.add_argument("--min-conf", type=float, default=0.0)
    p_infer.add_argument("--image-size", type=int, default=32)
    p_infer.set_defaults(func=_cmd_infer)

    p_split = sub.add_parser("split-labels", help="Split labeler outputs into train/val manifests")
    p_split.add_argument("--label-manifest", required=True, help="Path to label_manifest.jsonl from the labeler")
    p_split.add_argument("--images-dir", required=True, help="Directory containing cell_XX_YY.png images")
    p_split.add_argument("--out-train", required=True, help="Output JSONL for train split")
    p_split.add_argument("--out-val", required=True, help="Output JSONL for val split")
    p_split.add_argument("--val-fraction", type=float, default=0.2, help="Val fraction for entries without a split")
    p_split.add_argument("--seed", type=int, default=1337)
    p_split.add_argument("--no-append", action="store_true", help="Do not merge with existing train/val manifests; overwrite instead")
    def _cmd_split(args: argparse.Namespace) -> None:
        res = export_label_splits(
            args.label_manifest,
            args.images_dir,
            args.out_train,
            args.out_val,
            val_fraction=args.val_fraction,
            seed=args.seed,
            append_existing=not args.no_append,
        )
        print(json.dumps(res, indent=2))
    p_split.set_defaults(func=_cmd_split)

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
