"""
Document Processing Module
Handles PDF loading, text extraction, cleaning, chunking, and metadata attachment.
"""

import os
import re
import pdfplumber
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF using pdfplumber (primary) with PyPDF2 fallback.
    Returns a list of dicts: {page_number, text}
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages_data = []
    print(f" Extracting text from: {os.path.basename(pdf_path)}")

    # Primary extraction with pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        for row in table:
                            if row:
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                table_text += " | ".join(cleaned_row) + "\n"

                combined_text = text
                if table_text and table_text.strip() not in text:
                    combined_text += "\n\n[TABLE DATA]\n" + table_text

                pages_data.append({
                    "page_number": i + 1,
                    "text": combined_text,
                    "source": "pdfplumber"
                })
    except Exception as e:
        print(f" pdfplumber failed: {e}. Falling back to PyPDF2...")

    # Fallback / supplement with PyPDF2 for pages with no text
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if i < len(pages_data):
                if len(pages_data[i]["text"].strip()) < 20 and len(text.strip()) > 20:
                    pages_data[i]["text"] = text
                    pages_data[i]["source"] = "PyPDF2"
            else:
                pages_data.append({
                    "page_number": i + 1,
                    "text": text,
                    "source": "PyPDF2"
                })
    except Exception as e:
        print(f" PyPDF2 also failed: {e}")

    pages_with_text = sum(1 for p in pages_data if len(p["text"].strip()) > 20)
    print(f"  Extracted text from {pages_with_text}/{len(pages_data)} pages")
    return pages_data


def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
    text = text.strip()
    return text


def detect_section(text):
    """Detect the section/heading of a chunk based on its content."""
    section_patterns = [
        (r'(?i)\bboard.?s?\s+report\b', "Board's Report"),
        (r'(?i)\bfinancial\s+summary\b', 'Financial Summary'),
        (r'(?i)\bindependent\s+auditor', "Auditor's Report"),
        (r'(?i)\bbalance\s+sheet\b', 'Balance Sheet'),
        (r'(?i)\bstatement\s+of\s+profit\b', 'Profit & Loss Statement'),
        (r'(?i)\bcash\s+flow\b', 'Cash Flow Statement'),
        (r'(?i)\bcorporate\s+information\b', 'Corporate Information'),
        (r'(?i)\brisk\s+management\b', 'Risk Management'),
        (r'(?i)\brelated\s+party\b', 'Related Party Transactions'),
        (r'(?i)\bshare\s+capital\b', 'Share Capital'),
        (r'(?i)\bfood\s+delivery\b', 'Food Delivery Segment'),
        (r'(?i)\bquick\s+commerce\b|instamart', 'Quick Commerce / Instamart'),
        (r'(?i)\bdineout\b|out.of.home', 'Out-of-Home Consumption'),
        (r'(?i)\bsubsidiar', 'Subsidiaries'),
        (r'(?i)\besop\b|stock\s+option', 'ESOP / Stock Options'),
        (r'(?i)\bcsr\b|corporate\s+social', 'CSR'),
        (r'(?i)\bnotes\s+to\b.*financial', 'Notes to Financial Statements'),
    ]
    for pattern, section in section_patterns:
        if re.search(pattern, text[:500]):
            return section
    return "General"


def chunk_documents(pages_data, chunk_size=1000, chunk_overlap=200):
    """Split extracted pages into meaningful chunks with metadata."""
    print(f" Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False,
    )

    all_chunks = []
    for page_data in pages_data:
        text = clean_text(page_data["text"])
        if len(text.strip()) < 30:
            continue

        page_num = page_data["page_number"]
        section = detect_section(text)
        chunks = text_splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 20:
                continue
            metadata = {
                "page_number": page_num,
                "chunk_index": i,
                "section": section,
                "source_file": "Swiggy_Annual_Report_FY2023-24",
                "char_count": len(chunk_text),
            }
            doc = Document(page_content=chunk_text, metadata=metadata)
            all_chunks.append(doc)

    print(f" Created {len(all_chunks)} chunks from {len(pages_data)} pages")
    return all_chunks


def process_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Complete pipeline: Extract -> Clean -> Chunk -> Attach Metadata."""
    print("=" * 60)
    print("DOCUMENT PROCESSING PIPELINE")
    print("=" * 60)

    pages_data = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages_data, chunk_size, chunk_overlap)

    print(f"\n Processing Summary:")
    print(f"   Total pages in PDF: {len(pages_data)}")
    print(f"   Pages with text: {sum(1 for p in pages_data if len(p['text'].strip()) > 20)}")
    print(f"   Total chunks created: {len(chunks)}")
    sections = set(c.metadata.get('section', 'Unknown') for c in chunks)
    print(f"   Sections detected: {', '.join(sorted(sections))}")
    print("=" * 60)

    return chunks


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Annual-Report-FY-2023-24.pdf"
    chunks = process_pdf(pdf_path)
    print("\n Sample Chunks:\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ---")
        print(f"Page: {chunk.metadata['page_number']}, Section: {chunk.metadata['section']}")
        print(f"Text: {chunk.page_content[:200]}...\n")
