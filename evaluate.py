#!/usr/bin/env python3
"""Evaluate OCR results against ground truth."""

import argparse
from pathlib import Path

from src.evaluate import evaluate_ocr, print_evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OCR output against ground truth'
    )
    parser.add_argument('ocr_file', help='Path to OCR output file')
    parser.add_argument('--gt', default='ground-truth/ocr_test_document.txt',
                        help='Path to ground truth file')

    args = parser.parse_args()

    ocr_path = Path(args.ocr_file)
    gt_path = Path(args.gt)

    if not ocr_path.exists():
        print(f"Error: OCR file not found: {ocr_path}")
        return 1

    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        return 1

    results = evaluate_ocr(ocr_path, gt_path)
    print_evaluation(results, str(ocr_path), str(gt_path))

    return 0


if __name__ == '__main__':
    exit(main())
