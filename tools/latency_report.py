#!/usr/bin/env python3
"""
latency_report.py

Aggregate per-query latency from the audit log into p50/p95/p99 percentiles.

Reads records written by AuditLogger with a `timing_ms` block:
  - CLI:  type="query"   (includes the raw query text)
  - App:  type="latency" (PHI-free: timing + language + answered only)

Both are included by default. Percentiles are reported for each stage
(retrieve / rerank / llm / total) so the reranker's share of latency is
visible at a glance. No third-party dependencies.

Usage:
  python3 tools/latency_report.py [--log logs/audit.jsonl] [--answered-only]
                                  [--lang en|fr] [--type query|latency]
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional


STAGES = ["retrieve", "rerank", "llm", "total"]


def _pct(sorted_vals: List[float], q: float) -> float:
    # Nearest-rank percentile; robust for small samples.
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _load(path: str, want_type: Optional[str], lang: Optional[str],
          answered_only: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or not r.get("timing_ms"):
                continue
            if want_type and r.get("type") != want_type:
                continue
            if lang and r.get("language") != lang:
                continue
            if answered_only and not r.get("answered", False):
                continue
            rows.append(r)
    return rows


def _report(rows: List[Dict[str, Any]], label: str) -> None:
    n = len(rows)
    print(f"\n=== {label} (n={n}) ===")
    if n == 0:
        print("  (no matching records)")
        return
    reranked = [r["timing_ms"].get("n_reranked") for r in rows
                if r["timing_ms"].get("n_reranked") is not None]
    if reranked:
        print(f"  reranked pairs/query: min={min(reranked)} max={max(reranked)}")
    header = f"  {'stage':<10}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage in STAGES:
        vals = sorted(float(r["timing_ms"].get(stage, 0.0)) for r in rows)
        # Skip a stage that is uniformly zero (e.g. llm when rephrasing is off).
        if not any(vals):
            continue
        print(f"  {stage:<10}{_pct(vals, 0.50):>10.0f}{_pct(vals, 0.95):>10.0f}"
              f"{_pct(vals, 0.99):>10.0f}{vals[-1]:>10.0f}   ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="p50/p95/p99 latency report from the audit log.")
    ap.add_argument("--log", default=os.getenv("HORMONAI_AUDIT_LOG", "logs/audit.jsonl"))
    ap.add_argument("--type", choices=["query", "latency"], default=None,
                    help="Restrict to one record type (default: both).")
    ap.add_argument("--lang", choices=["en", "fr"], default=None)
    ap.add_argument("--answered-only", action="store_true",
                    help="Exclude abstentions from the percentiles.")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"Audit log not found: {args.log}")

    rows = _load(args.log, args.type, args.lang, args.answered_only)
    scope = []
    if args.type:
        scope.append(f"type={args.type}")
    if args.lang:
        scope.append(f"lang={args.lang}")
    if args.answered_only:
        scope.append("answered-only")
    _report(rows, "ALL" + (f" [{', '.join(scope)}]" if scope else ""))


if __name__ == "__main__":
    main()
