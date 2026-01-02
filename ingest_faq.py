#!/usr/bin/env python
import os
import pickle
import argparse
from typing import List, Dict, Any, Optional

from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Multilingual model: works for English + French
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def load_docx(path: str) -> Document:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Check the --doc argument.")
    return Document(path)


def is_heading(paragraph) -> bool:
    """
    Detect section headings via Word style and a simple ALL-CAPS heuristic.
    """
    try:
        style_name = paragraph.style.name or ""
    except Exception:
        style_name = ""

    if style_name.startswith("Heading"):
        return True

    text = paragraph.text.strip()
    if text and text.upper() == text and len(text.split()) <= 8:
        return True

    return False


def get_question_prefixes(language: str):
    """
    Language-specific 'question line' prefixes.
    Adapt if your formatting is slightly different.
    """
    if language == "fr":
        # e.g. "Q :", "Question :", "QUESTION :", etc.
        return ("q :", "q.", "question :", "question-")
    else:
        # e.g. "Q:", "Q.", "Question:"
        return ("q:", "q.", "question:")


def detect_question(text: str, language: str) -> Optional[str]:
    """
    Identify if a line is a question; return cleaned question text or None.
    """
    if not text:
        return None

    # Strip simple numbering like "1. ", "2) " etc.
    stripped = text.lstrip("0123456789).").strip()
    lower = stripped.lower()

    # Prefix patterns
    for pfx in get_question_prefixes(language):
        if lower.startswith(pfx):
            q = stripped[len(pfx):].strip()
            return q or None

    # Fallback: anything ending with '?' is treated as a question
    if stripped.endswith("?"):
        return stripped

    return None


def parse_docx_into_qa_sections(doc: Document, language: str) -> List[Dict[str, Any]]:
    """
    Build a list of:
      { "section": ..., "question": ..., "answer": ... }
    from the Word document.
    """
    qa_items: List[Dict[str, Any]] = []

    current_section = "General" if language == "en" else "Général"
    current_question: Optional[str] = None
    current_answer_parts: List[str] = []

    def flush_current():
        nonlocal current_question, current_answer_parts
        if current_question:
            answer_text = "\n\n".join(current_answer_parts).strip()
            qa_items.append(
                {
                    "section": current_section,
                    "question": current_question.strip(),
                    "answer": answer_text,
                }
            )
        current_question = None
        current_answer_parts = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        # New section?
        if is_heading(p):
            flush_current()
            current_section = text
            continue

        # New question?
        maybe_q = detect_question(text, language)
        if maybe_q is not None:
            flush_current()
            current_question = maybe_q
            continue

        # Answer paragraph
        if current_question:
            current_answer_parts.append(text)
        else:
            # Paragraph outside Q/A; treat as section intro if you want.
            # For now, we ignore it for retrieval.
            pass

    # Flush last question
    flush_current()
    return qa_items


def build_embeddings_and_index(qa_items: List[Dict[str, Any]]):
    """
    Create embeddings for each Q/A (with section context) and build a FAISS index.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = []
    for item in qa_items:
        combined = (
            f"Section: {item['section']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
        texts.append(combined)

    print(f"Encoding {len(texts)} Q/A chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a Word FAQ and build a FAISS index of Q/A pairs."
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "fr"],
        default="en",
        help="Language of the FAQ (affects question parsing and defaults).",
    )
    parser.add_argument(
        "--doc",
        "-d",
        default=None,
        help="Path to the .docx FAQ file. "
             "Default: faq_adjuvant_<language>.docx",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Output path for FAISS index. "
             "Default: faq_index_<language>.faiss",
    )
    parser.add_argument(
        "--qa",
        default=None,
        help="Output path for Q/A pickle. "
             "Default: faq_qa_<language>.pkl",
    )

    args = parser.parse_args()

    language = args.language
    docx_path = args.doc or f"faq_adjuvant_{language}.docx"
    index_path = args.index or f"faq_index_{language}.faiss"
    qa_path = args.qa or f"faq_qa_{language}.pkl"

    print(f"Language: {language}")
    print(f"Input DOCX: {docx_path}")
    print(f"Output index: {index_path}")
    print(f"Output QA pickle: {qa_path}\n")

    print("Loading document...")
    doc = load_docx(docx_path)

    print("Parsing document into section + Q/A items...")
    qa_items = parse_docx_into_qa_sections(doc, language)
    print(f"Parsed {len(qa_items)} Q/A entries.")

    if not qa_items:
        print(
            "WARNING: No Q/A pairs detected. You may need to tweak detect_question() "
            "and/or the formatting of your FAQ."
        )

    print("Building embeddings and FAISS index...")
    index = build_embeddings_and_index(qa_items)

    print(f"Saving index → {index_path}")
    faiss.write_index(index, index_path)

    print(f"Saving Q/A items → {qa_path}")
    with open(qa_path, "wb") as f:
        pickle.dump(
            {
                "items": qa_items,
                "embedding_model_name": EMBEDDING_MODEL_NAME,
                "language": language,
                "docx_path": docx_path,
            },
            f,
        )

    print("Done.")


if __name__ == "__main__":
    main()
