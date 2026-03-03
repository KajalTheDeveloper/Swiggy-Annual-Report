"""
Gradio Web UI for the Swiggy Annual Report RAG Application
Provides a user-friendly chat interface for asking questions.
"""

import os
import glob
import gradio as gr
from document_processor import process_pdf
from embedding_store import create_or_load_vector_store
from rag_engine import RAGEngine


# ============================================================
# Global state
# ============================================================
rag_engine = None


def initialize_rag(pdf_path):
    """Initialize the RAG pipeline with a PDF."""
    global rag_engine

    try:
        chunks = process_pdf(pdf_path)
        if not chunks:
            return "No text could be extracted from the PDF."

        vector_store, embeddings = create_or_load_vector_store(
            chunks=chunks,
            force_rebuild=True
        )

        rag_engine = RAGEngine(vector_store)

        return f"RAG system initialized! {len(chunks)} chunks from PDF. Ready to answer questions."
    except Exception as e:
        return f"Error initializing RAG: {str(e)}"


def answer_question(question):
    """Process a user question and return the answer with sources."""
    if rag_engine is None:
        return "RAG system is still initializing. Please wait..."

    if not question.strip():
        return "Please enter a question."

    try:
        result = rag_engine.query(question, k=5)

        answer = result["answer"]
        output = f"**Answer:**\n\n{answer}\n\n"
        output += f"---\n\n**Sources ({result['num_sources']} chunks used):**\n\n"
        for i, src in enumerate(result["sources"]):
            score = src['score']
            output += f"**[{i+1}]** Page {src['page']} | {src['section']} | Score: {score:.4f}\n\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


def create_ui():
    """Create and return the Gradio interface."""

    with gr.Blocks(
        title="Swiggy Annual Report RAG Q&A",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # Swiggy Annual Report - RAG Q&A System
            ### Ask any question about the Swiggy Annual Report FY 2023-24

            This system uses **Retrieval-Augmented Generation (RAG)** to answer your questions
            strictly based on the content of the Swiggy Annual Report. It will not hallucinate
            or make up information.

            **Tech Stack:** LangChain | FAISS | Sentence-Transformers | HuggingFace Flan-T5 | Gradio
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is Swiggy's total revenue for FY 2023-24?",
                    lines=2
                )
                ask_btn = gr.Button("Ask", variant="primary")

        answer_output = gr.Markdown(label="Answer")

        gr.Markdown("### Sample Questions (click to try)")
        gr.Examples(
            examples=[
                ["What is Swiggy's total revenue for FY 2023-24?"],
                ["Who are the board of directors of Swiggy?"],
                ["How many cities does Swiggy's food delivery operate in?"],
                ["What is the net loss of the company for FY 2023-24?"],
                ["What is Instamart and how does it work?"],
                ["Who is the CEO of Swiggy?"],
                ["Tell me about Swiggy's subsidiaries"],
                ["What is the company's registered address?"],
                ["Did Swiggy declare any dividend?"],
                ["What are the CSR activities of Swiggy?"],
            ],
            inputs=question_input,
        )

        ask_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

        question_input.submit(
            fn=answer_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

    return demo


if __name__ == "__main__":
    # Auto-initialize if PDF exists in data/
    data_pdfs = glob.glob("data/*.pdf")
    if data_pdfs:
        print(f"Found PDF: {data_pdfs[0]}")
        print("Auto-initializing RAG pipeline...")
        status = initialize_rag(data_pdfs[0])
        print(status)
    else:
        print("No PDF found in data/ folder. Please add the Swiggy Annual Report PDF.")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
