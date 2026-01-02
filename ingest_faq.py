#!/usr/bin/env python
import os
import pickle
import argparse
import re
from typing import List, Dict, Any, Optional

from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Stronger multilingual model (helps EN + FR retrieval)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_docx(path: str) -> Document:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Check the --doc argument.")
    return Document(path)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def is_heading(paragraph) -> bool:
    """
    Detect section headings via:
    1) Word Heading styles
    2) ALL-CAPS heuristic
    3) Pattern: "Section 6, ..." or "Section 6: ..."
    4) Pattern: leading "Section <number>" even if not styled
    """
    try:
        style_name = (paragraph.style.name or "").strip()
    except Exception:
        style_name = ""

    text = (paragraph.text or "").strip()
    if not text:
        return False

    # 1) Word heading styles
    if style_name.lower().startswith("heading"):
        return True

    # 2) ALL CAPS short lines
    if text.upper() == text and 1 <= len(text.split()) <= 10:
        return True

    # 3) "Section 6, Title" / "Section 6: Title" / "Section 6 - Title"
    if re.match(r"^section\s+\d+\s*[,:\-–]\s*\S+", text, flags=re.IGNORECASE):
        return True

    # 4) Sometimes it's "Section 6" alone
    if re.match(r"^section\s+\d+\s*$", text, flags=re.IGNORECASE):
        return True

    return False


def detect_question_prefix(text: str) -> Optional[str]:
    """
    Detect 'Q:' / 'Q :' / 'Question:' / etc.
    """
    if not text:
        return None
    t = text.strip()

    m = re.match(r"^(?:\d+\s*[\)\.\-]\s*)?(q|question)\s*[:\.\-–]\s*(.+)$", t, flags=re.IGNORECASE)
    if not m:
        return None
    qtext = m.group(2).strip()
    return qtext if qtext else None


def split_multi_questions(text: str) -> List[str]:
    """
    Split a line that contains multiple questions into separate question strings.

    Example:
      "Is it possible ...? Does pregnancy increase the risk ...?"
    """
    t = normalize_spaces(text)
    if not t:
        return []

    # Split on '? ' boundaries while keeping '?'
    parts = re.split(r"\?\s+", t)
    out = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i < len(parts) - 1 and not p.endswith("?"):
            p = p + "?"
        out.append(p)
    return out


def looks_like_question_sentence(text: str) -> bool:
    """
    STRICT: treat as a question only if it ends with '?'.
    This avoids misclassifying answer lines like:
      'Do not double up...'
      'However, ... recommended:'
    """
    t = (text or "").strip()
    return bool(t) and t.endswith("?")


def parse_docx_into_qa(doc: Document, language: str) -> List[Dict[str, Any]]:
    """
    Robust FAQ parser:
    - Supports section headings
    - Supports 'Q:' / 'Question:' prefixes
    - Supports multiple questions on the same Q line, all sharing the same answer

    CRITICAL BEHAVIOR for your case:
    If a 'Q:' line contains two questions, we store TWO entries with identical answer.
    """
    items: List[Dict[str, Any]] = []
    current_section = "General" if language == "en" else "Général"

    pending_questions: List[str] = []
    answer_parts: List[str] = []

    def flush():
        nonlocal pending_questions, answer_parts
        if not pending_questions:
            answer_parts = []
            return

        answer_text = "\n".join([p.strip() for p in answer_parts if p.strip()]).strip()

        # Only flush if we have an answer
        if answer_text:
            for q in pending_questions:
                items.append(
                    {
                        "section": current_section,
                        "question": normalize_spaces(q),
                        "answer": answer_text,
                    }
                )

        pending_questions = []
        answer_parts = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        if is_heading(p):
            flush()
            current_section = normalize_spaces(text)
            continue

        pref_q = detect_question_prefix(text)
        if pref_q is not None:
            flush()
            pending_questions = split_multi_questions(pref_q)
            continue

        # Sometimes the 2nd question is on a new line without Q:
        if looks_like_question_sentence(text):
            if pending_questions and not answer_parts:
                # treat as continuation of the question block
                pending_questions[-1] = normalize_spaces(pending_questions[-1] + " " + text)
                # re-split last question if it now contains multiple questions
                last = pending_questions.pop()
                pending_questions.extend(split_multi_questions(last))
            else:
                flush()
                pending_questions = split_multi_questions(text)
            continue

        # Otherwise: answer content
        if pending_questions:
            answer_parts.append(text)

    flush()
    return items


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Simple BM25 tokenizer:
    - lowercase
    - keep word characters
    """
    text = (text or "").lower()
    return re.findall(r"\b\w+\b", text)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    index.add(embeddings.astype("float32"))
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a Word FAQ and build hybrid retrieval artifacts (FAISS Q + FAISS QA + BM25)."
    )
    parser.add_argument("--language", "-l", choices=["en", "fr"], default="en")
    parser.add_argument(
        "--doc",
        "-d",
        default=None,
        help="Path to the .docx FAQ file. Default: faq_adjuvant_<language>.docx",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Prefix for outputs. Default: faq_<language>",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print first 15 parsed questions for a quick sanity check.",
    )
    parser.add_argument(
        "--grep",
        default=None,
        help="Search parsed Q/A for a keyword (case-insensitive) and print matches.",
    )

    args = parser.parse_args()
    language = args.language
    docx_path = args.doc or f"faq_adjuvant_{language}.docx"
    out_prefix = args.out_prefix or f"faq_{language}"

    index_q_path = f"{out_prefix}_index_q.faiss"
    index_qa_path = f"{out_prefix}_index_qa.faiss"
    qa_path = f"{out_prefix}_qa.pkl"
    bm25_path = f"{out_prefix}_bm25.pkl"

    print(f"Language: {language}")
    print(f"Input DOCX: {docx_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Output Q index:  {index_q_path}")
    print(f"Output QA index: {index_qa_path}")
    print(f"Output QA data:  {qa_path}")
    print(f"Output BM25:     {bm25_path}\n")

    print("Loading document...")
    doc = load_docx(docx_path)

    print("Parsing document into section + Q/A items...")
    qa_items = parse_docx_into_qa(doc, language)
    print(f"Parsed {len(qa_items)} Q/A entries.")

    if args.inspect:
        print("\n[INSPECT] First 15 questions:")
        for i, it in enumerate(qa_items[:15], start=1):
            print(f"{i:02d}. [{it['section']}] {it['question']}")
        print()

    if args.grep:
        kw = args.grep.lower()
        matches = [it for it in qa_items if kw in (it["question"] + "\n" + it["answer"]).lower()]
        print(f"\n[GREP] Matches for '{args.grep}': {len(matches)}")
        for it in matches[:25]:
            print(f"\n- SECTION: {it['section']}\n  Q: {it['question']}\n  A: {it['answer'][:350]}{'…' if len(it['answer'])>350 else ''}")
        print()

    if not qa_items:
        print(
            "WARNING: No Q/A pairs detected. You may need to tweak parsing heuristics "
            "and/or the formatting of your FAQ."
        )
        return

    print("Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    # --- Build separate corpora ---
    q_texts: List[str] = []
    qa_texts: List[str] = []
    bm25_docs: List[List[str]] = []

    for item in qa_items:
        sec = item["section"]
        q = item["question"]
        a = item["answer"]

        q_text = f"Section: {sec}\nQuestion: {q}"
        qa_text = f"Section: {sec}\nQuestion: {q}\nAnswer: {a}"

        q_texts.append(q_text)
        qa_texts.append(qa_text)

        bm25_docs.append(tokenize_for_bm25(f"{sec} {q} {a}"))

    print("Encoding Q-only embeddings...")
    q_emb = model.encode(q_texts, convert_to_numpy=True, show_progress_bar=True)

    print("Encoding QA embeddings...")
    qa_emb = model.encode(qa_texts, convert_to_numpy=True, show_progress_bar=True)

    print("Building FAISS indexes...")
    index_q = build_faiss_index(q_emb)
    index_qa = build_faiss_index(qa_emb)

    print("Building BM25...")
    bm25 = BM25Okapi(bm25_docs)

    print(f"Saving FAISS Q index → {index_q_path}")
    faiss.write_index(index_q, index_q_path)

    print(f"Saving FAISS QA index → {index_qa_path}")
    faiss.write_index(index_qa, index_qa_path)

    print(f"Saving QA items → {qa_path}")
    with open(qa_path, "wb") as f:
        pickle.dump(
            {
                "items": qa_items,
                "embedding_model_name": args.embedding_model,
                "language": language,
                "docx_path": docx_path,
                "out_prefix": out_prefix,
            },
            f,
        )

    print(f"Saving BM25 → {bm25_path}")
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "bm25_docs": bm25_docs,
                "language": language,
                "docx_path": docx_path,
                "out_prefix": out_prefix,
                "bm25": bm25,
            },
            f,
        )

    print("Done.")


if __name__ == "__main__":
    main()
