"""LightOnOCR-2 neural OCR module using mlx-vlm."""

import re
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def strip_markdown(text: str) -> str:
    """Remove markdown and HTML formatting from text.

    Args:
        text: Text with potential markdown/HTML formatting.

    Returns:
        Plain text without formatting.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Remove markdown links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove trailing spaces used for line breaks
    text = re.sub(r"  +$", "", text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LightOnOCR:
    """LightOnOCR-2 wrapper for high-quality document OCR."""

    def __init__(self, model_id: str = "mlx-community/LightOnOCR-2-1B-4bit"):
        """Initialize LightOnOCR with the specified model.

        Args:
            model_id: HuggingFace model ID. Defaults to 4-bit quantized version
                     optimized for Apple Silicon. Use 'lightonai/LightOnOCR-2-1B'
                     for full precision.
        """
        from mlx_vlm import load

        self.model_id = model_id
        self.model, self.processor = load(model_id)
        self._temp_dir = tempfile.mkdtemp(prefix="lighton_ocr_")

    def ocr(self, image: Image.Image, prompt: str = "", output_format: str = "markdown") -> str:
        """Extract text from a PIL Image.

        Args:
            image: PIL Image to process.
            prompt: Custom prompt for text extraction. Empty string triggers automatic OCR.
            output_format: Output format - 'markdown' (default) or 'plain'.

        Returns:
            Extracted text from the image.
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # mlx-vlm expects image as path string
        # Save to temp file
        temp_path = Path(self._temp_dir) / "temp_image.png"
        image.save(temp_path, format='PNG')

        # Apply chat template (required for proper tokenization)
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=1,
            num_audios=0
        )

        # LightOnOCR-2 works best with empty prompt for pure OCR
        result = generate(
            self.model, self.processor, formatted_prompt,
            image=str(temp_path),
            max_tokens=4096,
            temperature=0.0
        )
        text = result.text if hasattr(result, 'text') else str(result)

        if output_format == "plain":
            text = strip_markdown(text)

        return text

    def ocr_page(self, image_array: np.ndarray, output_format: str = "markdown") -> str:
        """Extract text from a numpy array image (BGR format).

        Args:
            image_array: OpenCV image as numpy array (BGR format).
            output_format: Output format - 'markdown' (default) or 'plain'.

        Returns:
            Extracted text from the image.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_array = image_array[:, :, ::-1]
        pil_image = Image.fromarray(rgb_array)
        return self.ocr(pil_image, output_format=output_format)


def parallel_page_ocr_lighton(
    images: List[np.ndarray],
    model_id: str = "mlx-community/LightOnOCR-2-1B-4bit",
    workers: int = None,
    output_format: str = "markdown"
) -> List[str]:
    """Perform OCR on multiple pages using LightOnOCR-2.

    Note: LightOnOCR uses GPU (Metal) so parallel CPU workers don't help much.
    Processing is done sequentially on the GPU for best efficiency.

    Args:
        images: List of BGR images as numpy arrays.
        model_id: HuggingFace model ID for the OCR model.
        workers: Ignored (kept for API compatibility with Tesseract).
        output_format: Output format - 'markdown' (default) or 'plain'.

    Returns:
        List of recognized texts, one per page.
    """
    if not images:
        return []

    # Initialize OCR engine once (loads model)
    ocr = LightOnOCR(model_id)

    # Process pages sequentially (GPU-bound, parallelism doesn't help)
    results = []
    for i, image in enumerate(images):
        print(f"  Processing page {i + 1}/{len(images)}...")
        text = ocr.ocr_page(image, output_format=output_format)
        results.append(text)

    return results
