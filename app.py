"""Gradio interface for Pix2Struct document question answering.

Usage: ``python app.py``
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator, List, Tuple

import gradio as gr

from doc_vqa import answer_questions, load_model

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

MODEL: Pix2StructForConditionalGeneration | None = None
PROCESSOR: Pix2StructProcessor | None = None


def collect_inputs(file_obj: gr.File, texts: str) -> Generator[Tuple[str, str], None, None]:
    """Handle file upload and questions from the Gradio interface with logging."""
    logs: List[str] = []

    def log(message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append(f"[{timestamp}] {message}")
        return "\n".join(logs)

    if file_obj is None:
        yield "", log("No file uploaded.")
        return

    file_path = Path(file_obj.name)
    yield "", log(f"Uploaded file: {file_path.name}")
    if file_path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        yield "", log("Invalid file type. Please upload PDF or image.")
        return

    questions: List[str] = [line.strip() for line in texts.splitlines() if line.strip()]
    yield "", log(f"Parsed {len(questions)} question(s)")
    if not questions:
        yield "", log("No questions provided.")
        return

    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None:
        yield "", log("Loading DocVQA model…")
        MODEL, PROCESSOR = load_model()
        yield "", log("Model loaded")

    yield "", log("Running inference…")
    results = answer_questions(str(file_path), questions, MODEL, PROCESSOR)
    result_text = "\n".join(f"{q}: {a}" for q, a in results)
    yield result_text, log("Inference complete")


def build_demo() -> gr.Blocks:
    """Create and return the Gradio Blocks demo."""
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Document QA"):
                gr.Markdown("### \U0001F4C4 Upload a file (PDF, PNG, or JPG):")
                file_input = gr.File(
                    label="File Upload",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )

                gr.Markdown("### \u2753 Enter your question(s), one per line:")
                questions = gr.Textbox(
                    label="Questions",
                    placeholder="Type each question on its own line…",
                    lines=5,
                    max_lines=10,
                )
                submit = gr.Button("Submit")
                output = gr.Textbox(label="Results", lines=10, show_copy_button=True)

            with gr.TabItem("Log"):
                gr.Markdown("### \U0001F4DD Activity Log")
                log_output = gr.Textbox(
                    label="Log",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                )

        submit.click(
            fn=collect_inputs,
            inputs=[file_input, questions],
            outputs=[output, log_output],
        )
    return demo


def main() -> None:
    """Launch the Gradio demo."""
    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
