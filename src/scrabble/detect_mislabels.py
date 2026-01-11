#!/usr/bin/env python3
"""
Detect potentially mislabeled images by comparing CNN predictions with labels.
Outputs a list of suspicious cases for manual review.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scrabble.cnn_ocr import load_checkpoint, load_manifest, predict_array
from PIL import Image
import numpy as np


def detect_mislabels(
    manifest_path: str,
    checkpoint_path: str,
    min_confidence: float = 0.7,
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    """
    Detect mislabeled images by finding cases where:
    - Model has high confidence (>min_confidence)
    - Model prediction differs from label
    
    Returns sorted list of suspicious cases by confidence.
    """
    model, meta = load_checkpoint(checkpoint_path)
    alphabet = meta.get("alphabet", [])
    device_name = meta.get("device")
    
    samples = load_manifest(manifest_path)
    suspicious: List[Dict[str, Any]] = []
    
    print(f"Checking {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  {i}/{len(samples)}...")
        
        try:
            img = Image.open(sample.path).convert("L")
            arr = np.array(img)
            pred_label, prob = predict_array(arr, model, alphabet=alphabet, device=device_name, image_size=32)
            
            # Check if prediction differs from label and model is confident
            if pred_label and pred_label != sample.label and prob >= min_confidence:
                suspicious.append({
                    "path": sample.path,
                    "true_label": sample.label,
                    "pred_label": pred_label,
                    "confidence": prob,
                    "row": sample.row,
                    "col": sample.col,
                })
        except Exception as e:
            print(f"Error processing {sample.path}: {e}")
            continue
    
    # Sort by confidence (highest first)
    suspicious.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Return top N
    return suspicious[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Detect mislabeled images in training data")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--checkpoint", required=True, help="Path to CNN checkpoint")
    parser.add_argument("--min-conf", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top suspicious cases to return")
    parser.add_argument("--out", help="Output JSON file (default: print to stdout)")
    
    args = parser.parse_args()
    
    suspicious = detect_mislabels(
        args.manifest,
        args.checkpoint,
        min_confidence=args.min_conf,
        top_n=args.top_n,
    )
    
    print(f"\nFound {len(suspicious)} suspicious cases:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Min confidence: {args.min_conf}")
    print("\nTop 10:")
    for i, case in enumerate(suspicious[:10], 1):
        print(f"  {i}. {Path(case['path']).name}: {case['true_label']} → {case['pred_label']} ({case['confidence']:.3f})")
    
    output = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "min_confidence": args.min_conf,
        "total_suspicious": len(suspicious),
        "cases": suspicious,
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.out}")
    else:
        print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
