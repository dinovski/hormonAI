#!/usr/bin/env python3
"""
Gold-standard retrieval / gating evaluation harness for hormonAI.

Runs the labelled cases in tests/eval_set.jsonl through the REAL pipeline
(retriever.retrieve + answer_query) and reports:

  - retrieval recall@k        : is a gold FAQ entry among the retrieved candidates?
  - answer decision accuracy  : does answered == expected_answerable?
  - false-abstention rate     : answerable cases that were (wrongly) abstained
  - false-answer rate         : out-of-scope cases that were (wrongly) answered
  - answer correctness        : answered cases whose lead source is a gold entry
  - per-category and per-language breakdowns

Use this to stress-test ANY change to the corpus, retrieval, or gating BEFORE
shipping it. Sweep thresholds with --sem-threshold / --coverage-fraction to
calibrate the gate for the deployed embedding model.

Run from the repo root inside the project venv:

    python tests/eval_retrieval.py                     # all cases, both languages
    python tests/eval_retrieval.py --lang en           # English only
    python tests/eval_retrieval.py --sem-threshold 0.6 --coverage-fraction 0.5
    python tests/eval_retrieval.py --no-rerank --json results.json
    python tests/eval_retrieval.py --include-hard      # count "hard" cases in totals

"hard" cases document known limitations (e.g. the strict stats gate) and are
reported separately by default so they do not mask regressions in core cases.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List

# Allow running from repo root or from tests/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_core import HybridFAQRetriever, answer_query  # noqa: E402


def load_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):5.1f}%" if d else "  n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="hormonAI gold retrieval/gating eval.")
    ap.add_argument("--eval-set", default=os.path.join(os.path.dirname(__file__), "eval_set.jsonl"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lang", choices=["en", "fr"], default=None, help="Restrict to one language.")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--embedding-model",
                    default=os.getenv("HORMONAI_EMBEDDING_MODEL",
                                      "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
                    help="Must match the model used at ingestion (e.g. BAAI/bge-m3).")
    ap.add_argument("--shared", action="store_true",
                    help="Evaluate the combined faq_all_* index (same-language-first + cross-lingual fallback).")
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rerank-model",
                    default=os.getenv("HORMONAI_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    ap.add_argument("--rerank-top-n", type=int,
                    default=int(os.getenv("HORMONAI_RERANK_TOP_N", "20")),
                    help="Rerank only the top-N fused candidates (match the deployed value).")
    ap.add_argument("--sem-threshold", type=float, default=0.62,
                    help="Dense cosine accept threshold (reranking OFF).")
    ap.add_argument("--rerank-threshold", type=float, default=-1.0,
                    help="Cross-encoder accept threshold (reranking ON).")
    ap.add_argument("--dense-floor", type=float, default=0.50,
                    help="Cosine floor for the high-IDF lexical safety net.")
    ap.add_argument("--include-hard", action="store_true", help="Count 'hard' cases in the headline totals.")
    ap.add_argument("--verbose", action="store_true", help="Print every case result.")
    ap.add_argument("--json", default=None, help="Write full per-case results to this JSON file.")
    args = ap.parse_args()

    cases = load_cases(args.eval_set)
    if args.lang:
        cases = [c for c in cases if c.get("lang") == args.lang]

    langs = sorted({c["lang"] for c in cases})
    retrievers: Dict[str, HybridFAQRetriever] = {}
    if args.shared:
        # One combined index serves every language; query language is set per case.
        print(f"Loading SHARED retriever faq_all_* (rerank={args.rerank})...")
        shared = HybridFAQRetriever(
            language="en", data_dir=args.data_dir, top_k=args.top_k,
            embedding_model=args.embedding_model, rerank=args.rerank,
            rerank_model=args.rerank_model, rerank_top_n=args.rerank_top_n,
            shared=True,
        )
        shared.load()
        for lang in langs:
            retrievers[lang] = shared
    else:
        for lang in langs:
            print(f"Loading retriever [{lang}] (rerank={args.rerank})...")
            r = HybridFAQRetriever(
                language=lang,
                data_dir=args.data_dir,
                top_k=args.top_k,
                embedding_model=args.embedding_model,
                rerank=args.rerank,
                rerank_model=args.rerank_model,
                rerank_top_n=args.rerank_top_n,
            )
            r.load()
            retrievers[lang] = r

    results: List[Dict[str, Any]] = []
    for c in cases:
        r = retrievers[c["lang"]]
        r.language = c["lang"]  # active query language (matters in shared mode)

        # Map positional indices -> stable item ids (works in per-language AND
        # shared mode, since the id lives on the item regardless of position).
        def _id_at(pos):
            try:
                return str(r._items[int(pos)].get("id", ""))
            except Exception:
                return ""

        cands = r.retrieve(c["query"])
        retrieved_ids = {_id_at(x.index) for x in cands}
        gold = set(c.get("gold_ids", []))

        res = answer_query(
            retriever=r,
            user_query=c["query"],
            use_llm=False,
            debug=True,
            sem_accept_threshold=args.sem_threshold,
            rerank_accept_threshold=args.rerank_threshold,
            dense_floor=args.dense_floor,
        )

        lead_id = _id_at(res.source_index) if res.answered else None
        bundle_ids = {_id_at(i) for i in (res.source_indices or [])}

        recall_hit = bool(gold & retrieved_ids) if gold else None
        decision_ok = (res.answered == c["expected_answerable"])
        # Did the ANSWER lead with a gold source? / include any gold source?
        lead_correct = (lead_id in gold) if (res.answered and gold) else None
        source_in_bundle = bool(gold & bundle_ids) if (res.answered and gold) else None

        row = {
            "id": c["id"],
            "lang": c["lang"],
            "category": c["category"],
            "hard": bool(c.get("hard", False)),
            "query": c["query"],
            "expected_answerable": c["expected_answerable"],
            "answered": res.answered,
            "decision_ok": decision_ok,
            "recall_hit": recall_hit,
            "lead_correct": lead_correct,
            "source_in_bundle": source_in_bundle,
            "lead_id": lead_id,
            "gold_ids": sorted(gold),
            "decision_path": (res.debug or {}).get("decision_path"),
            "sem_top_sim": (res.debug or {}).get("sem_top_sim"),
            "present_concepts": (res.debug or {}).get("present_concepts"),
            "absent_concepts": (res.debug or {}).get("absent_concepts"),
        }
        results.append(row)
        if args.verbose:
            flag = "OK " if decision_ok else "XX "
            print(f"{flag}[{c['id']}] answered={res.answered} exp={c['expected_answerable']} "
                  f"path={row['decision_path']} recall={recall_hit} lead_ok={lead_correct} "
                  f"in_bundle={source_in_bundle} lead={lead_id}")

    core = [r for r in results if args.include_hard or not r["hard"]]
    hard = [r for r in results if r["hard"]]

    def summarize(rows: List[Dict[str, Any]], title: str) -> None:
        if not rows:
            return
        n = len(rows)
        dec_ok = sum(1 for r in rows if r["decision_ok"])
        ans_cases = [r for r in rows if r["expected_answerable"]]
        oos_cases = [r for r in rows if not r["expected_answerable"]]
        false_abstain = sum(1 for r in ans_cases if not r["answered"])
        false_answer = sum(1 for r in oos_cases if r["answered"])
        recall_cases = [r for r in rows if r["recall_hit"] is not None]
        recall_hits = sum(1 for r in recall_cases if r["recall_hit"])
        lead_cases = [r for r in rows if r["lead_correct"] is not None]
        lead_hits = sum(1 for r in lead_cases if r["lead_correct"])
        bundle_cases = [r for r in rows if r["source_in_bundle"] is not None]
        bundle_hits = sum(1 for r in bundle_cases if r["source_in_bundle"])

        print(f"\n===== {title} (n={n}) =====")
        print(f"  Decision accuracy (answered==expected) : {dec_ok}/{n}  {pct(dec_ok, n)}")
        print(f"  Retrieval recall@{args.top_k} (gold in candidates): {recall_hits}/{len(recall_cases)}  {pct(recall_hits, len(recall_cases))}")
        print(f"  Lead-source correctness (headline src is gold): {lead_hits}/{len(lead_cases)}  {pct(lead_hits, len(lead_cases))}")
        print(f"  Gold source anywhere in answer bundle         : {bundle_hits}/{len(bundle_cases)}  {pct(bundle_hits, len(bundle_cases))}")
        print(f"  FALSE-ABSTENTION (answerable wrongly refused): {false_abstain}/{len(ans_cases)}  {pct(false_abstain, len(ans_cases))}")
        print(f"  FALSE-ANSWER (out-of-scope wrongly answered) : {false_answer}/{len(oos_cases)}  {pct(false_answer, len(oos_cases))}")

        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        print("  By category (decision accuracy):")
        for cat in sorted(by_cat):
            rs = by_cat[cat]
            ok = sum(1 for r in rs if r["decision_ok"])
            print(f"    - {cat:14s}: {ok}/{len(rs)}  {pct(ok, len(rs))}")

    summarize(core, "CORE CASES")
    if hard and not args.include_hard:
        summarize(hard, "HARD CASES (documented limitations, reported separately)")

    # Failing core cases, for quick triage.
    fails = [r for r in core if not r["decision_ok"]]
    if fails:
        print("\n----- FAILING CORE CASES -----")
        for r in fails:
            reason = "false-abstention" if r["expected_answerable"] else "false-answer"
            print(f"  [{r['id']}] {reason}: answered={r['answered']} path={r['decision_path']} "
                  f"sim={r['sem_top_sim']} absent={r['absent_concepts']}")
            print(f"      Q: {r['query']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote per-case results to {args.json}")

    # Exit non-zero if any CORE case regressed (useful in CI).
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
