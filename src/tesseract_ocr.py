"""Tesseract OCR with OpenCV text region detection and parallel processing."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract


def detect_text_regions_mser(image: np.ndarray, min_area: int = 100, max_area: int = 50000) -> List[Tuple[int, int, int, int]]:
    """Detect text regions using MSER algorithm.

    Args:
        image: BGR image as numpy array.
        min_area: Minimum region area to consider.
        max_area: Maximum region area to consider.

    Returns:
        List of bounding boxes as (x, y, w, h) tuples.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)

    regions, _ = mser.detectRegions(gray)

    # Convert regions to bounding boxes
    bboxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        bboxes.append((x, y, w, h))

    # Merge overlapping boxes
    return _merge_overlapping_boxes(bboxes)


def detect_text_regions_morphology(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect text regions using morphological operations.

    This method is often more reliable for document OCR than MSER.

    Args:
        image: BGR image as numpy array.

    Returns:
        List of bounding boxes as (x, y, w, h) tuples.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Create horizontal and vertical kernels for text line detection
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))

    # Dilate to connect text characters into lines
    dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
    dilated_v = cv2.dilate(binary, kernel_v, iterations=1)

    # Combine and dilate further to create text blocks
    combined = cv2.bitwise_or(dilated_h, dilated_v)
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(combined, kernel_block, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to bounding boxes, filter small regions
    bboxes = []
    min_area = 500  # Minimum area for a text block
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area and w > 20 and h > 10:
            bboxes.append((x, y, w, h))

    return bboxes


def _merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Merge overlapping bounding boxes.

    Args:
        boxes: List of (x, y, w, h) tuples.
        overlap_thresh: Overlap threshold for merging.

    Returns:
        Merged list of bounding boxes.
    """
    if not boxes:
        return []

    # Convert to numpy array for easier manipulation
    boxes_array = np.array(boxes)

    # Sort by y-coordinate (top to bottom)
    sorted_indices = np.argsort(boxes_array[:, 1])
    boxes_array = boxes_array[sorted_indices]

    merged = []
    used = set()

    for i, box in enumerate(boxes_array):
        if i in used:
            continue

        x1, y1, w1, h1 = box
        current = [x1, y1, x1 + w1, y1 + h1]

        # Check for overlaps with remaining boxes
        for j in range(i + 1, len(boxes_array)):
            if j in used:
                continue

            x2, y2, w2, h2 = boxes_array[j]
            box2 = [x2, y2, x2 + w2, y2 + h2]

            # Check if boxes overlap
            if _boxes_overlap(current, box2, overlap_thresh):
                # Merge boxes
                current[0] = min(current[0], box2[0])
                current[1] = min(current[1], box2[1])
                current[2] = max(current[2], box2[2])
                current[3] = max(current[3], box2[3])
                used.add(j)

        used.add(i)
        merged.append((current[0], current[1], current[2] - current[0], current[3] - current[1]))

    return merged


def _boxes_overlap(box1: List[int], box2: List[int], thresh: float) -> bool:
    """Check if two boxes overlap beyond threshold."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return False

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    min_area = min(area1, area2)
    if min_area == 0:
        return False

    return intersection / min_area > thresh


def _ocr_region(args: Tuple[np.ndarray, Tuple[int, int, int, int], str]) -> Tuple[int, int, str]:
    """OCR a single region (worker function for multiprocessing).

    Args:
        args: Tuple of (image, bbox, language).

    Returns:
        Tuple of (y, x, recognized_text) for sorting.
    """
    image, bbox, lang = args
    x, y, w, h = bbox

    # Add padding around the region
    pad = 5
    y1 = max(0, y - pad)
    y2 = min(image.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(image.shape[1], x + w + pad)

    roi = image[y1:y2, x1:x2]

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply slight preprocessing
    roi = cv2.GaussianBlur(roi, (1, 1), 0)

    # Run Tesseract
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(roi, lang=lang, config=config)

    return (y, x, text.strip())


def parallel_ocr(
    image: np.ndarray,
    regions: List[Tuple[int, int, int, int]],
    lang: str = 'hun',
    workers: int = None
) -> str:
    """Perform parallel OCR on detected regions.

    Args:
        image: BGR image as numpy array.
        regions: List of bounding boxes (x, y, w, h).
        lang: Tesseract language code.
        workers: Number of worker processes. Defaults to CPU count.

    Returns:
        Concatenated text from all regions, sorted by position.
    """
    if not regions:
        return ""

    if workers is None:
        workers = min(cpu_count(), len(regions))

    # Prepare arguments for workers
    args = [(image, region, lang) for region in regions]

    # Run OCR in parallel
    with Pool(processes=workers) as pool:
        results = pool.map(_ocr_region, args)

    # Sort by position (top to bottom, left to right)
    results.sort(key=lambda r: (r[0], r[1]))

    # Combine text
    texts = [r[2] for r in results if r[2]]
    return '\n\n'.join(texts)


def full_page_ocr(image: np.ndarray, lang: str = 'hun') -> str:
    """Perform OCR on the full page (for comparison).

    Args:
        image: BGR image as numpy array.
        lang: Tesseract language code.

    Returns:
        Recognized text.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Run Tesseract on full page
    config = '--oem 3 --psm 3'
    text = pytesseract.image_to_string(gray, lang=lang, config=config)

    return text.strip()


def _ocr_page_worker(args: Tuple[np.ndarray, str]) -> str:
    """Worker function for parallel page OCR."""
    image, lang = args
    return full_page_ocr(image, lang)


def parallel_page_ocr(
    images: List[np.ndarray],
    lang: str = 'hun',
    workers: int = None
) -> List[str]:
    """Perform OCR on multiple pages in parallel.

    Args:
        images: List of BGR images as numpy arrays.
        lang: Tesseract language code.
        workers: Number of worker processes.

    Returns:
        List of recognized texts, one per page.
    """
    if not images:
        return []

    if workers is None:
        workers = min(cpu_count(), len(images))

    args = [(img, lang) for img in images]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_ocr_page_worker, args))

    return results


def process_image(
    image: np.ndarray,
    lang: str = 'hun',
    method: str = 'morphology',
    parallel: bool = True,
    workers: int = None
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """Process a single image with text detection and OCR.

    Args:
        image: BGR image as numpy array.
        lang: Tesseract language code.
        method: Detection method - 'mser', 'morphology', or 'full'.
        parallel: Use parallel processing for regions.
        workers: Number of worker processes.

    Returns:
        Tuple of (recognized_text, detected_regions).
    """
    if method == 'full':
        text = full_page_ocr(image, lang)
        return text, []

    # Detect text regions
    if method == 'mser':
        regions = detect_text_regions_mser(image)
    else:  # morphology
        regions = detect_text_regions_morphology(image)

    if not regions:
        # Fall back to full page OCR if no regions detected
        text = full_page_ocr(image, lang)
        return text, []

    # Perform OCR on regions
    if parallel and len(regions) > 1:
        text = parallel_ocr(image, regions, lang, workers)
    else:
        # Sequential processing for single region or when parallel disabled
        results = []
        for region in regions:
            result = _ocr_region((image, region, lang))
            results.append(result)
        results.sort(key=lambda r: (r[0], r[1]))
        text = '\n\n'.join(r[2] for r in results if r[2])

    return text, regions
