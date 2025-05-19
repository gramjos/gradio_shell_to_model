"""
 collect user input: document and questions
    - document: pdf, png, jpg, jpeg
    - question: delineated by new line in the text box
"""
import gradio as gr
import os

def collect_inputs(file_obj, texts):
    """
    Process the uploaded file and questions.
    """
    if file_obj is None:
        return "Please upload a file first."

    # Get file extension
    file_path = file_obj.name
    file_ext = os.path.splitext(file_path)[1].lower()

    # Validate file type
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    if file_ext not in allowed_extensions:
        return f"Invalid file type. Please upload one of: {', '.join(allowed_extensions)}"

    # Process questions
    questions = [line.strip() for line in texts.splitlines() if line.strip()]
    if not questions:
        return "Please enter at least one question."

    # results = run_doc_vqa(file_path, questions, page_no=1)
    # return "\n".join(f"{q}: {a}" for q, a in results)

    return f"File uploaded: {file_path}\n\nQuestions:\n" + "\n".join(questions)


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Document QA"):
            gr.Markdown("### 📄 Upload a file (PDF, PNG, or JPG):")
            file_input = gr.File(
                label="File Upload",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath"
            )

            gr.Markdown("### ❓ Enter your question(s), one per line:")
            questions = gr.Textbox(
                label="Questions",
                placeholder="Type each question on its own line…",
                lines=5,               # show 5 rows by default
                max_lines=10          # allow up to 10 lines
            )
            submit = gr.Button("Submit")
            output = gr.Textbox(label="Results", lines=10)

            submit.click(
                fn=collect_inputs,
                inputs=[file_input, questions],
                outputs=output
            )
        
        with gr.TabItem("Log"):
            gr.Markdown("### 📝 Activity Log")
            log_output = gr.Textbox(
                label="Log",
                lines=15,
                interactive=False
            )

demo.launch()
