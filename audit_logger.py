#!/usr/bin/env python3
"""
Very lightweight JSONL audit logs.
No PHI, no raw doc dumps—just enough to debug retrieval + answerability.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional


class AuditLogger:
    def __init__(self, path: str = "logs/audit.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_query(
        self,
        query: str,
        language: str,
        answered: bool,
        used_llm: bool,
        source_index: Optional[int] = None,
        source_question: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log(
            {
                "type": "query",
                "query": query,
                "language": language,
                "answered": answered,
                "used_llm": used_llm,
                "source_index": source_index,
                "source_question": source_question,
                "meta": meta or {},
            }
        )
