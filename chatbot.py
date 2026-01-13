#!/usr/bin/env python3
"""
chatbot.py

CLI for hormonAI.

Key rule:
- LLM is ONLY used to add a tone wrapper to an already-answered FAQ response.
- Abstains NEVER go to the LLM.
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

    p.add_argument("--rerank", action="store_true", help="Enable CrossEncoder reranking (better, slower).")
    p.add_argument("--rerank-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

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
