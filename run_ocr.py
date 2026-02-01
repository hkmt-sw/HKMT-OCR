#!/usr/bin/env python3
"""Main OCR script for Hungarian document processing."""

import argparse
import time
from pathlib import Path

from src.pdf_utils import pdf_to_images
from src.tesseract_ocr import parallel_page_ocr, process_image


def main():
    parser = argparse.ArgumentParser(
        description='Hungarian OCR POC - Tesseract + OpenCV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for PDF conversion (150=fast, 300=quality)')
    parser.add_argument('--method', choices=['morphology', 'mser', 'full', 'fast'], default='fast',
                        help='Text detection method (fast=parallel full-page OCR)')
    parser.add_argument('--lang', default='hun', help='Tesseract language code')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--engine', choices=['tesseract', 'lighton'], default='tesseract',
                        help='OCR engine: tesseract (fast, CPU) or lighton (neural, GPU)')
    parser.add_argument('--format', choices=['markdown', 'plain'], default='markdown',
                        help='Output format for LightOnOCR: markdown (structured) or plain (no formatting)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    parallel_mode = not args.no_parallel
    print(f"Processing: {pdf_path.name}")
    print(f"DPI: {args.dpi}, Method: {args.method}, Engine: {args.engine}, Parallel: {parallel_mode}")
    print("-" * 50)

    # Convert PDF to images
    start_time = time.perf_counter()
    print("Converting PDF to images...")
    images = pdf_to_images(pdf_path, dpi=args.dpi)
    convert_time = time.perf_counter() - start_time
    print(f"  Converted {len(images)} page(s) in {convert_time:.2f}s")

    # Process pages
    ocr_start = time.perf_counter()

    if args.engine == 'lighton':
        # LightOnOCR-2 neural OCR (GPU-accelerated)
        from src.lighton_ocr import parallel_page_ocr_lighton
        print(f"Running LightOnOCR-2 on {len(images)} pages (format: {args.format})...")
        all_text = parallel_page_ocr_lighton(images, output_format=args.format)
    elif args.method == 'fast' and parallel_mode and len(images) > 1:
        # Fast parallel page processing with Tesseract
        print(f"Running parallel OCR on {len(images)} pages...")
        all_text = parallel_page_ocr(images, args.lang, args.workers)
    else:
        # Sequential or region-based processing with Tesseract
        all_text = []
        for i, image in enumerate(images):
            page_start = time.perf_counter()
            if args.verbose:
                print(f"Processing page {i + 1}/{len(images)}...")

            method = 'full' if args.method == 'fast' else args.method
            text, regions = process_image(
                image,
                lang=args.lang,
                method=method,
                parallel=parallel_mode,
                workers=args.workers
            )

            if args.verbose:
                page_time = time.perf_counter() - page_start
                print(f"  Detected {len(regions)} text region(s), processed in {page_time:.2f}s")

            all_text.append(text)

    ocr_time = time.perf_counter() - ocr_start
    total_time = time.perf_counter() - start_time

    # Combine all pages
    full_text = '\n\n---PAGE---\n\n'.join(all_text)

    # Save output
    output_file = output_dir / f"{pdf_path.stem}_ocr.txt"
    output_file.write_text(full_text, encoding='utf-8')

    print("-" * 50)
    print(f"OCR completed in {ocr_time:.2f}s (total: {total_time:.2f}s)")
    print(f"Output saved to: {output_file}")

    # Print sample of recognized text
    print("\n--- Sample output (first 500 chars) ---")
    print(full_text[:500])
    if len(full_text) > 500:
        print("...")

    # Check for Hungarian characters
    hun_chars = set('áéíóöőúüűÁÉÍÓÖŐÚÜŰ')
    found_hun = [c for c in full_text if c in hun_chars]
    if found_hun:
        unique_hun = set(found_hun)
        print(f"\nHungarian characters found: {' '.join(sorted(unique_hun))}")
    else:
        print("\nWarning: No Hungarian special characters detected!")

    return 0


if __name__ == '__main__':
    exit(main())
