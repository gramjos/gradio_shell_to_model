"""Utilities for document question answering using Pix2Struct.

Usage: ``python doc_vqa.py`` will load the model and run a simple demo.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple

from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id: str = "google/pix2struct-docvqa-large") -> Tuple[Pix2StructForConditionalGeneration, Pix2StructProcessor]:
    """Load Pix2Struct DocVQA model and processor.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.

    Returns
    -------
    tuple
        The loaded model and processor.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    processor = Pix2StructProcessor.from_pretrained(model_id)
    return model, processor


def convert_to_image(input_source: str | bytes | Path, page_no: int = 1) -> Image.Image:
    """Convert a PDF page or image file to :class:`PIL.Image`.

    Parameters
    ----------
    input_source:
        Path to a PDF/image or raw bytes.
    page_no:
        1-based page number for PDFs.

    Returns
    -------
    PIL.Image.Image
    """
    if isinstance(input_source, (bytes, bytearray)):
        return Image.open(BytesIO(input_source))

    path = Path(input_source)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = convert_from_path(str(path))
        if not 1 <= page_no <= len(pages):
            raise ValueError(f"page_no {page_no} out of range for {len(pages)} page PDF")
        return pages[page_no - 1]
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        if page_no != 1:
            raise ValueError("page_no > 1 not supported for image inputs")
        return Image.open(path)
    raise ValueError(f"Unsupported file extension '{suffix}'")


def answer_questions(
    file_path: str | Path,
    questions: Iterable[str],
    model: Pix2StructForConditionalGeneration | None = None,
    processor: Pix2StructProcessor | None = None,
    page_no: int = 1,
) -> List[Tuple[str, str]]:
    """Generate answers for the provided questions against the given document.

    Parameters
    ----------
    file_path:
        Path to a PDF or image file.
    questions:
        Questions to ask.
    model, processor:
        Optionally supply pre-loaded model objects.
    page_no:
        Page number for PDFs.

    Returns
    -------
    list of tuples
        ``[(question, answer), ...]``
    """
    if model is None or processor is None:
        model, processor = load_model()

    image = convert_to_image(file_path, page_no)
    question_list = list(questions)
    inputs = processor(images=[image] * len(question_list), text=question_list, return_tensors="pt").to(DEVICE)
    predictions = model.generate(**inputs, max_new_tokens=1028)
    answers = processor.batch_decode(predictions, skip_special_tokens=True)
    return list(zip(question_list, answers))


if __name__ == "__main__":
    MODEL, PROCESSOR = load_model()
    demo_questions = ["Example question?"]
    for q, a in answer_questions("example.png", demo_questions, MODEL, PROCESSOR):
        print(f"{q}: {a}")
