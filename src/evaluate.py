"""Evaluate OCR results against ground truth."""

from pathlib import Path


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(ocr_text: str, gt_text: str) -> float:
    """Calculate Character Error Rate.

    CER = (insertions + deletions + substitutions) / total_gt_characters
    """
    distance = levenshtein_distance(ocr_text, gt_text)
    if len(gt_text) == 0:
        return 0.0 if len(ocr_text) == 0 else 1.0
    return distance / len(gt_text)


def calculate_wer(ocr_text: str, gt_text: str) -> float:
    """Calculate Word Error Rate.

    WER = (insertions + deletions + substitutions) / total_gt_words
    """
    ocr_words = ocr_text.split()
    gt_words = gt_text.split()

    distance = levenshtein_distance(ocr_words, gt_words)
    if len(gt_words) == 0:
        return 0.0 if len(ocr_words) == 0 else 1.0
    return distance / len(gt_words)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Normalize whitespace
    lines = text.strip().split('\n')
    lines = [' '.join(line.split()) for line in lines]
    # Remove empty lines
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def evaluate_ocr(ocr_path: str | Path, gt_path: str | Path) -> dict:
    """Evaluate OCR output against ground truth.

    Args:
        ocr_path: Path to OCR output file.
        gt_path: Path to ground truth file.

    Returns:
        Dictionary with CER, WER, and other metrics.
    """
    ocr_path = Path(ocr_path)
    gt_path = Path(gt_path)

    ocr_text = ocr_path.read_text(encoding='utf-8')
    gt_text = gt_path.read_text(encoding='utf-8')

    # Normalize texts
    ocr_normalized = normalize_text(ocr_text)
    gt_normalized = normalize_text(gt_text)

    cer = calculate_cer(ocr_normalized, gt_normalized)
    wer = calculate_wer(ocr_normalized, gt_normalized)

    return {
        'cer': cer,
        'wer': wer,
        'cer_percent': cer * 100,
        'wer_percent': wer * 100,
        'ocr_chars': len(ocr_normalized),
        'gt_chars': len(gt_normalized),
        'ocr_words': len(ocr_normalized.split()),
        'gt_words': len(gt_normalized.split()),
    }


def print_evaluation(results: dict, ocr_path: str = "", gt_path: str = ""):
    """Print evaluation results."""
    print(f"OCR Evaluation Results")
    if ocr_path:
        print(f"  OCR file: {ocr_path}")
    if gt_path:
        print(f"  GT file:  {gt_path}")
    print("-" * 40)
    print(f"  CER: {results['cer_percent']:.2f}%")
    print(f"  WER: {results['wer_percent']:.2f}%")
    print(f"  Characters: {results['ocr_chars']} (OCR) vs {results['gt_chars']} (GT)")
    print(f"  Words: {results['ocr_words']} (OCR) vs {results['gt_words']} (GT)")
