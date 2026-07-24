#!/usr/bin/env python3
"""
chatbot.py

CLI for hormonAI.

Key rules:
- Retrieval is FAQ-only.
- LLM (if enabled) is used ONLY for a short empathetic wrapper (no facts).
- Answers always quote the FAQ content (no added medical facts).
"""

from __future__ import annotations

import argparse

from rag_core import HybridFAQRetriever, answer_query, print_debug
from audit_logger import AuditLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hormonAI CLI chatbot (FAQ-restricted RAG).")
    p.add_argument("--language", "-l", choices=["en", "fr"], default="en")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Reranking is ON by default: it reorders candidates so the coverage/stats
    # gates operate on better-ordered results. Use --no-rerank to disable.
    # The cross-encoder loads lazily and degrades gracefully if unavailable.
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True,
                   help="CrossEncoder reranking (better ordering, slower). Default: on. Disable with --no-rerank.")
    p.add_argument("--rerank-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # Gate tuning (calibrate on tests/eval_set.jsonl for the deployed models).
    # Semantic-first: rerank score is primary when --rerank is on, else dense cosine.
    p.add_argument("--sem-threshold", type=float, default=0.62,
                   help="Dense cosine accept threshold when reranking is OFF (default: 0.62).")
    p.add_argument("--rerank-threshold", type=float, default=0.0,
                   help="Cross-encoder accept threshold when reranking is ON (default: 0.0).")
    p.add_argument("--dense-floor", type=float, default=0.50,
                   help="Permissive cosine floor for the high-IDF lexical safety net (default: 0.50).")

    p.add_argument("--use-llm", action="store_true", help="Add an empathetic tone wrapper with an LLM (ONLY for answered queries).")
    p.add_argument("--llm-model", default="llama3.2", help="Ollama model name (default: llama3.2).")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--audit-log", default="logs/audit.jsonl")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading FAQ retriever for LANGUAGE = {args.language}...")
    retriever = HybridFAQRetriever(
        language=args.language,
        data_dir=args.data_dir,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
    )
    retriever.load()

    logger = AuditLogger(args.audit_log)

    if args.language == "fr":
        print("Chatbot prêt (FAQ hormonothérapie adjuvante).")
        print("Tapez 'exit' ou 'quit' pour quitter.\n")
    else:
        print("Chatbot ready (adjuvant hormone therapy FAQ).")
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
