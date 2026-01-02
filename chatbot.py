#!/usr/bin/env python
import argparse
from rag_core import (
    HybridFAQRetriever,
    choose_best_candidate,
    format_faq_answer,
    answer_with_llm,
    DEFAULT_EMBEDDING_MODEL,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", "-l", choices=["en", "fr"], default="en")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--llm-provider", choices=["openai", "ollama"], default="ollama")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--ollama-model", default="llama3.2")

    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--recall-k", type=int, default=60)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--grep", default=None)
    args = ap.parse_args()

    prefix = args.prefix or f"faq_{args.language}"

    retriever = HybridFAQRetriever(
        prefix=prefix,
        language=args.language,
        embedding_model=args.embedding_model,
        enable_rerank=args.rerank,
    )

    if args.grep:
        hits = retriever.grep(args.grep)
        print(f"[GREP] '{args.grep}' matches: {len(hits)}")
        for h in hits[:30]:
            print(f"\n[{h['index']}] SECTION: {h['section']}\nQ: {h['question']}\nA: {h['answer'][:350]}{'…' if len(h['answer'])>350 else ''}")
        return

    print(f"Loading FAQ retriever for LANGUAGE = {args.language}...")
    print("Chatbot ready. Type 'exit' to leave.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        candidates = retriever.retrieve(q, top_k=args.top_k, recall_k=args.recall_k, debug=args.debug)
        best = choose_best_candidate(q, args.language, candidates, debug=args.debug)

        if best is None:
            print("\nBot: I’m not finding a sufficiently relevant FAQ entry to answer reliably.\n")
            continue

        if args.use_llm:
            ans = answer_with_llm(
                language=args.language,
                provider=args.llm_provider,
                openai_model=args.openai_model,
                ollama_model=args.ollama_model,
                user_query=q,
                top=best,
            )
        else:
            ans = format_faq_answer(best, args.language)

        print("\nBot:", ans, "\n")

if __name__ == "__main__":
    main()
