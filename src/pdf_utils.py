"""PDF to image conversion utilities."""

from pathlib import Path
from typing import List

import numpy as np
from pdf2image import convert_from_path
from PIL import Image


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> List[np.ndarray]:
    """Convert PDF pages to numpy arrays (BGR format for OpenCV).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for conversion. Higher = better quality but slower.
             Recommended: 150 for fast processing, 300 for accuracy.

    Returns:
        List of numpy arrays, one per page, in BGR format.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Convert PDF to PIL Images
    pil_images = convert_from_path(str(pdf_path), dpi=dpi)

    # Convert to numpy arrays in BGR format (OpenCV standard)
    images = []
    for pil_img in pil_images:
        # Convert to RGB numpy array
        rgb_array = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        bgr_array = rgb_array[:, :, ::-1].copy()
        images.append(bgr_array)

    return images


def pdf_to_pil_images(pdf_path: str | Path, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for conversion.

    Returns:
        List of PIL Image objects, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return convert_from_path(str(pdf_path), dpi=dpi)
