# Document Question Answering Demo

This repository provides a minimal Gradio interface for asking questions about
PDF or image documents using the Pix2Struct DocVQA model from Hugging Face.

## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demo
   ```bash
   python app.py
   ```

The interface allows you to upload a document and type one or more questions.
Results are produced using the Pix2Struct model.  A second tab shows verbose
logs for each processing step with timestamps.  Use the copy button above the
log to easily capture the output.

The code is designed to run in Google Colab and can be deployed to a Hugging
Face Space.
