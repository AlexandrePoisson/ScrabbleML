#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass(frozen=True)
class ManifestEntry:
	file: str
	label: str
	split: str
	session: Optional[str] = None
	row: Optional[int] = None
	col: Optional[int] = None


def _iter_jsonl(path: Path) -> Iterable[dict]:
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				yield json.loads(line)
			except Exception:
				continue


def _is_letter_label(label: str) -> bool:
	return len(label) == 1 and label.isalpha()


def _normalize_letter(label: str) -> Optional[str]:
	if not _is_letter_label(label):
		return None
	return label.upper()


def _compute_median(values: Sequence[int]) -> int:
	if not values:
		return 0
	values_sorted = sorted(values)
	return values_sorted[len(values_sorted) // 2]


def _select_rare_letters(letter_counts: Dict[str, int], bottom_n: int) -> List[str]:
	if bottom_n <= 0:
		return []
	items = sorted(letter_counts.items(), key=lambda kv: (kv[1], kv[0]))
	if not items:
		return []
	cutoff_index = min(bottom_n - 1, len(items) - 1)
	cutoff_count = items[cutoff_index][1]
	# Include ties at cutoff
	return [k for k, v in items if v <= cutoff_count]


def _random_params(
	rng: random.Random,
	max_rotation_deg: float,
	max_scale_delta: float,
	max_shift_px: int,
) -> Tuple[float, float, int, int]:
	angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
	scale = rng.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta)
	dx = rng.randint(-max_shift_px, max_shift_px)
	dy = rng.randint(-max_shift_px, max_shift_px)
	return angle, scale, dx, dy


def _augment_image(img: np.ndarray, angle: float, scale: float, dx: int, dy: int) -> np.ndarray:
	h, w = img.shape[:2]
	center = (w / 2.0, h / 2.0)
	m = cv2.getRotationMatrix2D(center, angle, scale)
	m[0, 2] += float(dx)
	m[1, 2] += float(dy)
	return cv2.warpAffine(
		img,
		m,
		dsize=(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REPLICATE,
	)


def main(argv: Optional[Sequence[str]] = None) -> int:
	p = argparse.ArgumentParser(
		description=(
			"Identify least-represented letters in label_manifest.jsonl and generate small image augmentations "
			"(±2° rotation, tiny scale, ±2px shift)."
		)
	)
	p.add_argument("--manifest", default="webdata/label_store/label_manifest.jsonl")
	p.add_argument("--images-dir", default="webdata/label_store/images")
	p.add_argument(
		"--out-subdir",
		default="augmented",
		help="Subdirectory created under --images-dir to store augmented images (default: augmented)",
	)
	p.add_argument(
		"--out-manifest",
		default="webdata/label_store/augmented_label_manifest.jsonl",
		help="JSONL manifest containing ONLY augmented entries",
	)
	p.add_argument(
		"--out-combined-manifest",
		default="webdata/label_store/label_manifest_with_aug.jsonl",
		help="JSONL manifest combining original + augmented entries",
	)
	p.add_argument(
		"--only-split",
		default="train",
		choices=["train", "val", "any"],
		help="Only augment samples from this split (default: train)",
	)
	p.add_argument("--bottom-n", type=int, default=6, help="Select bottom-N rare letters (ties included)")
	p.add_argument(
		"--target-per-letter",
		default="median",
		help="Target count per selected letter (int or 'median')",
	)
	p.add_argument("--max-rotation-deg", type=float, default=2.0)
	p.add_argument(
		"--max-scale-delta",
		type=float,
		default=0.02,
		help="Scale range is [1-delta, 1+delta] (default: 0.02 -> 0.98..1.02)",
	)
	p.add_argument("--max-shift-px", type=int, default=2)
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument(
		"--dry-run",
		action="store_true",
		help="Print what would be done without writing images/manifests",
	)

	args = p.parse_args(list(argv) if argv is not None else None)

	manifest_path = Path(args.manifest)
	images_dir = Path(args.images_dir)
	out_dir = images_dir / args.out_subdir
	out_manifest_path = Path(args.out_manifest)
	out_combined_path = Path(args.out_combined_manifest)

	if not manifest_path.exists():
		raise SystemExit(f"Manifest not found: {manifest_path}")
	if not images_dir.exists():
		raise SystemExit(f"Images dir not found: {images_dir}")

	letter_counts: Counter[str] = Counter()
	label_counts: Counter[str] = Counter()
	by_letter: Dict[str, List[ManifestEntry]] = defaultdict(list)

	for data in _iter_jsonl(manifest_path):
		label_raw = str(data.get("label", "") or "")
		label_counts[label_raw] += 1

		letter = _normalize_letter(label_raw)
		if letter is None:
			continue

		split = str(data.get("split", "") or "").lower() or ""
		if args.only_split != "any" and split != args.only_split:
			continue

		fname = data.get("file")
		if not fname:
			continue

		letter_counts[letter] += 1
		by_letter[letter].append(
			ManifestEntry(
				file=str(fname),
				label=letter,
				split=split,
				session=data.get("session"),
				row=data.get("row"),
				col=data.get("col"),
			)
		)

	if not letter_counts:
		print("No letter labels found to augment (check --only-split / label format).")
		return 0

	rare_letters = _select_rare_letters(dict(letter_counts), args.bottom_n)

	values = list(letter_counts.values())
	if isinstance(args.target_per_letter, str) and args.target_per_letter.lower() == "median":
		target = _compute_median(values)
	else:
		target = int(args.target_per_letter)

	print("Label summary (top 10):", label_counts.most_common(10))
	print("Letter counts (A-Z) seen:", len(letter_counts), "total:", sum(letter_counts.values()))
	print("Rarest letters selected:", ",".join(rare_letters))
	print("Target per selected letter:", target)

	rng = random.Random(args.seed)

	augmented_entries: List[dict] = []
	total_written = 0

	for letter in rare_letters:
		sources = by_letter.get(letter, [])
		have = letter_counts.get(letter, 0)
		if have <= 0 or not sources:
			print(f"{letter}: no sources found (count={have})")
			continue

		need = max(0, target - have)
		print(f"{letter}: have={have}, need={need}, sources={len(sources)}")
		if need == 0:
			continue

		sources_shuffled = list(sources)
		rng.shuffle(sources_shuffled)

		for i in range(need):
			src = sources_shuffled[i % len(sources_shuffled)]
			src_path = images_dir / src.file
			if not src_path.exists():
				continue

			out_letter_dir = out_dir / letter
			src_stem = Path(src.file).stem
			src_session = src.session or (Path(src.file).parts[0] if Path(src.file).parts else "unknown")
			out_name = f"{src_session}_{src_stem}_aug{i:04d}.png"
			out_rel = str(Path(args.out_subdir) / letter / out_name)
			out_path = images_dir / out_rel

			if args.dry_run:
				augmented_entries.append(
					{
						"file": out_rel,
						"session": "augmented",
						"label": letter,
						"split": src.split,
						"source_file": src.file,
					}
				)
				continue

			out_letter_dir.mkdir(parents=True, exist_ok=True)

			img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
			if img is None:
				continue

			angle, scale, dx, dy = _random_params(
				rng,
				max_rotation_deg=float(args.max_rotation_deg),
				max_scale_delta=float(args.max_scale_delta),
				max_shift_px=int(args.max_shift_px),
			)
			aug_img = _augment_image(img, angle=angle, scale=scale, dx=dx, dy=dy)

			out_path.parent.mkdir(parents=True, exist_ok=True)
			ok = cv2.imwrite(str(out_path), aug_img)
			if not ok:
				continue

			augmented_entries.append(
				{
					"file": out_rel,
					"session": "augmented",
					"label": letter,
					"split": src.split,
					"source_file": src.file,
					"aug": {"rot_deg": angle, "scale": scale, "dx": dx, "dy": dy},
				}
			)
			total_written += 1

	print("Augmented samples created:", total_written if not args.dry_run else len(augmented_entries))
	if args.dry_run:
		print("Dry-run: not writing manifests.")
		return 0

	out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
	with out_manifest_path.open("w", encoding="utf-8") as f:
		for e in augmented_entries:
			f.write(json.dumps(e) + "\n")

	with out_combined_path.open("w", encoding="utf-8") as f:
		with manifest_path.open("r", encoding="utf-8") as orig:
			for line in orig:
				line = line.strip("\n")
				if line.strip():
					f.write(line + "\n")
		for e in augmented_entries:
			f.write(json.dumps(e) + "\n")

	print("Wrote:")
	print("- augmented images dir:", str(out_dir))
	print("- augmented manifest:", str(out_manifest_path))
	print("- combined manifest:", str(out_combined_path))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())