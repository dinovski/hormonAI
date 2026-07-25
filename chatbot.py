#!/usr/bin/env python3
"""
chatbot.py

CLI for hormonAI.

Key rules:
- Retrieval is restricted to the knowledge base (FAQs + articles).
- LLM (if enabled) rephrases ONLY the retrieved source text (no added facts).
- Answers are grounded strictly in the retrieved sources (verbatim or grounded
  rephrase), with sources always cited.
"""

from __future__ import annotations

import os
import argparse

from rag_core import HybridFAQRetriever, answer_query, print_debug
from audit_logger import AuditLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hormonAI CLI chatbot (knowledge-base-restricted RAG).")
    p.add_argument("--language", "-l", choices=["en", "fr"], default="en")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--embedding-model",
                   default=os.getenv("HORMONAI_EMBEDDING_MODEL",
                                     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
                   help="Must match the model used at ingestion (e.g. BAAI/bge-m3).")

    # Reranking is ON by default: it reorders candidates so the coverage/stats
    # gates operate on better-ordered results. Use --no-rerank to disable.
    # The cross-encoder loads lazily and degrades gracefully if unavailable.
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                   help="CrossEncoder reranking (better ordering, slower). Default: on. Disable with --no-rerank.")
    p.add_argument("--rerank-model",
                   default=os.getenv("HORMONAI_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
                   help="Cross-encoder reranker (e.g. BAAI/bge-reranker-v2-m3 for stronger multilingual ranking).")

    # Shared multilingual index (faq_all_*): same-language preferred, cross-lingual
    # fallback. --language then selects the active QUERY language. Default: off
    # (loads the per-language faq_<lang>_* index).
    p.add_argument("--shared", action="store_true",
                   help="Use the combined faq_all_* index (cross-lingual fallback).")

    # Gate tuning (calibrate on tests/eval_set.jsonl for the deployed models).
    # Semantic-first: rerank score is primary when --rerank is on, else dense cosine.
    p.add_argument("--sem-threshold", type=float, default=0.62,
                   help="Dense cosine accept threshold when reranking is OFF (default: 0.62).")
    p.add_argument("--rerank-threshold", type=float, default=-1.0,
                   help="Cross-encoder accept threshold when reranking is ON (default: -1.0; calibrate on eval set).")
    p.add_argument("--dense-floor", type=float, default=0.50,
                   help="Permissive cosine floor for the high-IDF lexical safety net (default: 0.50).")

    p.add_argument("--use-llm", action="store_true", help="Add an empathetic tone wrapper with an LLM (ONLY for answered queries).")
    p.add_argument("--llm-model", default="llama3.2", help="Ollama model name (default: llama3.2).")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--audit-log", default="logs/audit.jsonl")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading retriever for LANGUAGE = {args.language}...")
    retriever = HybridFAQRetriever(
        language=args.language,
        data_dir=args.data_dir,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        shared=args.shared,
    )
    retriever.load()

    logger = AuditLogger(args.audit_log)

    if args.language == "fr":
        print("Chatbot prêt (base de connaissances hormonothérapie adjuvante).")
        print("Tapez 'exit' ou 'quit' pour quitter.\n")
    else:
        print("Chatbot ready (adjuvant hormone therapy knowledge base).")
        print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Bye.")
            break

        result = answer_query(
            retriever=retriever,
            user_query=user,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            debug=args.debug,
            sem_accept_threshold=args.sem_threshold,
            rerank_accept_threshold=args.rerank_threshold,
            dense_floor=args.dense_floor,
        )

        if args.debug:
            print_debug(result)

        logger.log_query(
            query=user,
            language=args.language,
            answered=result.answered,
            used_llm=(args.use_llm and result.answered),
            source_index=result.source_index,
            source_question=result.source_title,
            meta=(result.debug if args.debug else None),
        )

        print("\nBot:", result.answer_text, "\n")


if __name__ == "__main__":
    main()
