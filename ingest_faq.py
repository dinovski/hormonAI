#!/usr/bin/env python
import os
import pickle
import argparse
import re
from typing import List, Dict, Any, Optional

import json
import urllib.request

from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Stronger multilingual model (helps EN + FR retrieval)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_docx(path: str) -> Document:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DOCX not found: {path}")
    return Document(path)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def is_heading(paragraph) -> bool:
    """
    Detect section headings using:
    1) Word heading styles
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

    # 2) ALL CAPS, not too long
    if text.isupper() and len(text) <= 120:
        return True

    # 3) "Section 6: ..." / "Section 6, ..."
    if re.match(r"^section\s+\d+\s*[:,-]", text, flags=re.IGNORECASE):
        return True

    # 4) "Section 6" at the beginning
    if re.match(r"^section\s+\d+\b", text, flags=re.IGNORECASE):
        return True

    return False


def detect_question_prefix(text: str) -> Optional[str]:
    """
    Return the text after question prefix if present, else None.
    Supports:
    - "Q:" / "Q :" / "Question:" / "Question :"
    - French "Q :" as well
    """
    t = (text or "").strip()
    m = re.match(r"^(Q|Question)\s*[:：]\s*(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()
    return None


def looks_like_question_sentence(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    # Sometimes questions are written without ?, so use soft heuristics
    if re.match(r"^(can|do|does|is|are|should|what|when|why|how)\b", t, flags=re.IGNORECASE):
        return True
    if re.match(r"^(puis|dois|est-ce|comment|pourquoi|quand|quoi)\b", t, flags=re.IGNORECASE):
        return True
    return False


def split_multi_questions(qtext: str) -> List[str]:
    """
    Split lines with multiple questions into separate questions.
    Heuristics:
    - split on '?' keeping it
    - split on ';' if it appears to join two questions
    """
    q = normalize_spaces(qtext)
    if not q:
        return []

    # First split by '?'
    parts = re.split(r"(\?)", q)
    out: List[str] = []
    buf = ""
    for p in parts:
        if p == "?":
            buf = (buf + "?").strip()
            if buf:
                out.append(buf)
            buf = ""
        else:
            buf += " " + p
            buf = buf.strip()

    if buf:
        # leftover without '?'
        out.append(buf.strip())

    # If still only 1 and has ';', maybe split
    if len(out) == 1 and ";" in out[0]:
        semi = [normalize_spaces(x) for x in out[0].split(";") if normalize_spaces(x)]
        if len(semi) >= 2:
            return semi

    # De-dup
    seen = set()
    dedup = []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup


def parse_docx_into_qa(doc: Document, language: str) -> List[Dict[str, Any]]:
    """
    Robust FAQ parser:
    - Supports section headings
    - Supports 'Q:' / 'Question:' prefixes
    - Supports multiple questions on the same Q line, all sharing the same answer

    CRITICAL BEHAVIOR:
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


def _ollama_paraphrases(
    question: str,
    language: str,
    n: int,
    host: str,
    model: str,
    timeout_s: int,
) -> List[str]:
    """
    Generate paraphrases of a QUESTION ONLY, for retrieval augmentation.
    Safety: must not add medical facts; only rewrite the intent.
    Returns a list of strings (may be empty if generation fails).
    """
    q = (question or "").strip()
    if not q or n <= 0:
        return []

    lang = (language or "en").lower()

    sys = (
        "You generate paraphrases of the user's QUESTION ONLY. "
        "Do NOT add any medical facts, advice, statistics, timelines, examples, or new entities. "
        "Do NOT mention drugs or conditions unless already present in the question. "
        "Return ONLY valid JSON: an array of strings."
    )
    if lang.startswith("fr"):
        sys = (
            "Tu génères uniquement des reformulations de la QUESTION. "
            "N'ajoute AUCUN fait médical, conseil, statistique, durée, exemple, ni nouvelle entité. "
            "Ne mentionne pas de médicaments ou pathologies sauf s'ils sont déjà dans la question. "
            "Retourne UNIQUEMENT du JSON valide : une liste de chaînes."
        )

    prompt = (
        f"Question: {q}\n"
        f"Generate {n} different paraphrases in everyday language. Keep meaning exactly the same."
    )
    if lang.startswith("fr"):
        prompt = (
            f"Question: {q}\n"
            f"Génère {n} reformulations différentes en langage courant. Garde exactement le même sens."
        )

    base = (host or "http://localhost:11434").rstrip("/")
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": sys,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 300},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        raw = (out.get("response") or "").strip()
    except Exception:
        return []

    parsed: List[str] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            parsed = [str(x) for x in obj if isinstance(x, (str, int, float))]
    except Exception:
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    parsed = [str(x) for x in obj if isinstance(x, (str, int, float))]
            except Exception:
                parsed = []

    cleaned: List[str] = []
    seen = set()
    for s in parsed:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if not s:
            continue
        # reject unsafe additions: numbers/percent
        if re.search(r"(\d|%)", s):
            continue
        # avoid overly long paraphrases
        if len(s) > 180:
            continue
        # avoid identical
        if s.lower() == q.lower():
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        cleaned.append(s)

    return cleaned[:n]


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
        "--augment-questions",
        action="store_true",
        help="Generate paraphrases of each FAQ question (offline, for retrieval only) and build an extra FAISS index.",
    )
    parser.add_argument(
        "--paraphrase-n",
        type=int,
        default=5,
        help="Number of paraphrases to generate per question when --augment-questions is enabled.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL for paraphrase generation (default: env OLLAMA_HOST or http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name for paraphrase generation (default: llama3.2).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for each Ollama request during paraphrase generation.",
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
    index_qp_path = f"{out_prefix}_index_qp.faiss"  # optional paraphrase-augmented question index
    qa_path = f"{out_prefix}_qa.pkl"
    bm25_path = f"{out_prefix}_bm25.pkl"

    print(f"Language: {language}")
    print(f"Input DOCX: {docx_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Output Q index:  {index_q_path}")
    print(f"Output QA index: {index_qa_path}")
    print(f"Output Q+Paraphrase index: {index_qp_path}")
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
            print(
                f"\n- SECTION: {it['section']}\n  Q: {it['question']}\n  A: {it['answer'][:350]}"
                f"{'…' if len(it['answer'])>350 else ''}"
            )
        print()

    if not qa_items:
        print(
            "WARNING: No Q/A pairs detected. You may need to tweak parsing heuristics "
            "and/or the formatting of your FAQ."
        )
        return

    if args.augment_questions:
        print(
            f"Paraphrase augmentation ENABLED: n={args.paraphrase_n} "
            f"(Ollama host={args.ollama_host}, model={args.ollama_model})"
        )

    print("Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    # --- Build separate corpora ---
    q_texts: List[str] = []
    qa_texts: List[str] = []
    qp_texts: List[str] = []  # question + paraphrases (retrieval-only)
    bm25_docs: List[List[str]] = []

    for item in qa_items:
        sec = item["section"]
        q = item["question"]
        a = item["answer"]

        # Optional: generate paraphrases of the QUESTION for retrieval augmentation
        q_paras: List[str] = []
        if args.augment_questions:
            q_paras = _ollama_paraphrases(
                question=q,
                language=language,
                n=max(0, int(args.paraphrase_n)),
                host=args.ollama_host,
                model=args.ollama_model,
                timeout_s=int(args.ollama_timeout),
            )
        item["q_paraphrases"] = q_paras

        q_text = f"Section: {sec}\nQuestion: {q}"
        qa_text = f"Section: {sec}\nQuestion: {q}\nAnswer: {a}"

        # Retrieval-only text: section + question + paraphrases (if any)
        if item.get("q_paraphrases"):
            paras = "\n".join([f"- {p}" for p in item["q_paraphrases"]])
            qp_text = f"Section: {sec}\nQuestion: {q}\nParaphrases:\n{paras}"
        else:
            qp_text = q_text
        item["retrieval_text"] = qp_text

        q_texts.append(q_text)
        qa_texts.append(qa_text)
        qp_texts.append(qp_text)

        bm25_docs.append(tokenize_for_bm25(f"{sec} {q} {a}"))

    print("Encoding Q-only embeddings...")
    q_emb = model.encode(q_texts, convert_to_numpy=True, show_progress_bar=True)

    print("Encoding QA embeddings...")
    qa_emb = model.encode(qa_texts, convert_to_numpy=True, show_progress_bar=True)

    qp_emb = None
    if args.augment_questions:
        print("Encoding Q+Paraphrase embeddings (retrieval-only augmentation)...")
        qp_emb = model.encode(qp_texts, convert_to_numpy=True, show_progress_bar=True)

    print("Building FAISS indexes...")
    index_q = build_faiss_index(q_emb)
    index_qa = build_faiss_index(qa_emb)
    index_qp = build_faiss_index(qp_emb) if qp_emb is not None else None

    print("Building BM25...")
    bm25 = BM25Okapi(bm25_docs)

    print(f"Saving FAISS Q index → {index_q_path}")
    faiss.write_index(index_q, index_q_path)

    print(f"Saving FAISS QA index → {index_qa_path}")
    faiss.write_index(index_qa, index_qa_path)

    if index_qp is not None:
        print(f"Saving FAISS Q+Paraphrase index → {index_qp_path}")
        faiss.write_index(index_qp, index_qp_path)

    print(f"Saving QA items → {qa_path}")
    with open(qa_path, "wb") as f:
        pickle.dump(
            {
                "items": qa_items,
                "embedding_model_name": args.embedding_model,
                "language": language,
                "docx_path": docx_path,
                "out_prefix": out_prefix,
                "augment_questions": bool(args.augment_questions),
                "paraphrase_n": int(args.paraphrase_n),
                "ollama_model": str(args.ollama_model),
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
