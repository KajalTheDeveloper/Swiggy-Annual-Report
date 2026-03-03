"""
CLI Interface for the Swiggy Annual Report RAG Q&A System
Run this to interact with the system via command line.
"""

import sys
import os
from document_processor import process_pdf
from embedding_store import create_or_load_vector_store
from rag_engine import RAGEngine


def print_header():
    print("\n" + "=" * 65)
    print("  SWIGGY ANNUAL REPORT — RAG Q&A SYSTEM (CLI)")
    print("  Based on Swiggy Annual Report FY 2023-24")
    print("=" * 65)


def print_result(result: dict):
    print(f"\n Answer:")
    print(f"   {result['answer']}")
    print(f"\n Sources ({result['num_sources']} chunks used):")
    for i, src in enumerate(result["sources"]):
        print(f"   [{i+1}] Page {src['page']} | {src['section']} | Score: {src['score']:.4f}")
    print()


def main():
    print_header()

    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for PDF in data/ directory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        pdf_files = []
        if os.path.exists(data_dir):
            pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

        if pdf_files:
            pdf_path = os.path.join(data_dir, pdf_files[0])
            print(f"\n Found PDF: {pdf_path}")
        else:
            print("\n No PDF found in data/ directory.")
            pdf_path = input("   Enter the path to the Swiggy Annual Report PDF: ").strip()
            if not pdf_path or not os.path.exists(pdf_path):
                print("Invalid path. Exiting.")
                sys.exit(1)

    # Initialize RAG pipeline
    print(f"\n Initializing RAG pipeline...")
    print(f"   PDF: {pdf_path}\n")

    chunks = process_pdf(pdf_path)
    if not chunks:
        print(" No text extracted from PDF. Exiting.")
        sys.exit(1)

    vector_store, embeddings = create_or_load_vector_store(chunks, force_rebuild=True)
    rag = RAGEngine(vector_store)

    print("\n System ready! Type your questions below.")
    print("   Type 'quit' or 'exit' to stop.\n")

    # Interactive Q&A loop
    while True:
        try:
            question = input(" You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n Goodbye!")
            break

        result = rag.query(question, k=5)
        print_result(result)


if __name__ == "__main__":
    main()
