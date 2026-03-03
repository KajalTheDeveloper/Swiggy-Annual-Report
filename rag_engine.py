"""
RAG (Retrieval-Augmented Generation) Engine
Retrieves relevant context and generates answers using a local HuggingFace LLM.
The system strictly answers from document context to prevent hallucination.
"""

import re
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ============================================================
# PROMPT TEMPLATE — Grounding the LLM to document context only
# ============================================================

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions ONLY based on the provided context from the Swiggy Annual Report FY 2023-24. 

RULES:
1. Answer ONLY using information from the context below.
2. If the context does not contain enough information to answer, say: "I could not find sufficient information in the Annual Report to answer this question."
3. Do NOT make up or assume any information.
4. Be specific — include numbers, dates, and names when available.
5. Keep answers concise but complete.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


class RAGEngine:
    """
    RAG Engine that retrieves context from FAISS and generates answers
    using a local HuggingFace model (no API key required).
    """

    def __init__(self, vector_store: FAISS, model_name: str = "google/flan-t5-base"):
        """
        Initialize the RAG Engine.
        
        Args:
            vector_store: FAISS vector store with embedded document chunks
            model_name: HuggingFace model for text generation (default: flan-t5-base)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.generator = None
        self._load_model()

    def _load_model(self):
        """Load the text generation model."""
        print(f"🤖 Loading LLM: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=False,
                device=-1,  # CPU
            )
            print("   LLM loaded successfully")
        except Exception as e:
            print(f"    Failed to load model: {e}")
            print("   Falling back to extractive QA mode.")
            self.generator = None

    def retrieve_context(self, query: str, k: int = 5) -> list:
        """
        Retrieve the most relevant document chunks for a given query.
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        documents = []
        for doc, score in results:
            doc.metadata["similarity_score"] = float(score)
            documents.append(doc)
        return documents

    def format_context(self, documents: list) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            page = doc.metadata.get("page_number", "?")
            section = doc.metadata.get("section", "Unknown")
            context_parts.append(
                f"[Source: Page {page}, Section: {section}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM based on retrieved context.
        Falls back to extractive mode if LLM is not available.
        """
        if self.generator is None:
            return self._extractive_answer(query, context)

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

        # Truncate prompt if too long for the model
        max_input_length = 512
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_input_length)
        truncated_prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            result = self.generator(truncated_prompt)[0]["generated_text"]
            answer = result.strip()
            # If LLM gives a very short/unhelpful answer, fall back to extractive
            if len(answer) < 20 or answer.lower() in ("annual report", "swiggy", "n/a", "yes", "no"):
                return self._extractive_answer(query, context)
            return answer
        except Exception as e:
            print(f" Generation failed: {e}")
            return self._extractive_answer(query, context)

    def _extractive_answer(self, query: str, context: str) -> str:
        """
        Fallback: Extract the most relevant sentences from context.
        Uses keyword matching and sentence scoring.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        sentences = re.split(r'(?<=[.!?])\s+', context)

        scored = []
        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            overlap = len(query_words & sent_words)
            # Bonus for sentences with numbers (likely factual)
            has_numbers = bool(re.search(r'\d+', sent))
            score = overlap + (1 if has_numbers else 0)
            if score > 0:
                scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            top_sentences = [s for _, s in scored[:5]]
            return " ".join(top_sentences)
        else:
            return "I could not find sufficient information in the Annual Report to answer this question."

    def query(self, question: str, k: int = 5) -> dict:
        """
        Full RAG pipeline: Retrieve → Format → Generate.
        
        Returns:
            dict with 'answer', 'sources', and 'context'
        """
        # Step 1: Retrieve relevant chunks
        documents = self.retrieve_context(question, k=k)

        # Step 2: Format context
        context = self.format_context(documents)

        # Step 3: Generate answer
        answer = self.generate_answer(question, context)

        # Step 4: Prepare source references
        sources = []
        for doc in documents:
            sources.append({
                "page": doc.metadata.get("page_number", "?"),
                "section": doc.metadata.get("section", "Unknown"),
                "score": doc.metadata.get("similarity_score", 0),
                "preview": doc.page_content[:150] + "..."
            })

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_sources": len(documents)
        }


if __name__ == "__main__":
    from document_processor import process_pdf
    from embedding_store import create_or_load_vector_store
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Swiggy_Annual_Report_FY2023-24.pdf"

    # Build pipeline
    chunks = process_pdf(pdf_path)
    vector_store, embeddings = create_or_load_vector_store(chunks, force_rebuild=True)
    rag = RAGEngine(vector_store)

    # Test queries
    test_questions = [
        "What is Swiggy's total revenue for FY 2023-24?",
        "Who is the CEO of Swiggy?",
        "How many cities does Swiggy operate in?",
        "What is the net loss of the company?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f" {q}")
        result = rag.query(q)
        print(f" {result['answer']}")
        print(f" Sources: {len(result['sources'])} chunks used")
