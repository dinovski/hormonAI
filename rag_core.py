# rag_core.py
import os
import re
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------
# Keywords / gating
# ---------------------------
EN_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","so","because",
    "to","of","in","on","for","with","as","at","by","from","into","about",
    "is","are","was","were","be","been","being",
    "do","does","did","doing",
    "can","could","should","would","will","may","might","must",
    "i","you","we","they","he","she","it","my","your","our","their",
    "this","that","these","those",
    "what","why","how","when","where","which",
}
EN_KEEP = {"pregnancy","recurrence","tamoxifen","aromatase","inhibitor","clot","thrombosis","fertility","child","pause", "sun", "uv", "mri"}

FR_STOPWORDS = {"le","la","les","un","une","des","et","ou","mais","si","alors",
                "de","du","dans","sur","pour","avec","par","au","aux","en",
                "est","sont","été","être","avoir","a","ont",
                "je","tu","il","elle","nous","vous","ils","elles",
                "ce","cet","cette","ces",
                "quoi","pourquoi","comment","quand","où","quel","quelle","quels","quelles"}

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())

def extract_keywords(query: str, language: str) -> List[str]:
    toks = tokenize(query)
    if language == "en":
        kws = []
        for t in toks:
            if t in EN_KEEP:
                kws.append(t)
            elif t not in EN_STOPWORDS and len(t) >= 3:
                kws.append(t)
        seen = set()
        out = []
        for k in kws:
            if k not in seen:
                out.append(k); seen.add(k)
        return out[:12]
    kws = [t for t in toks if t not in FR_STOPWORDS and len(t) >= 3]
    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            out.append(k); seen.add(k)
    return out[:12]

def overlap_score(query_keywords: List[str], item: Dict[str, Any]) -> int:
    blob = " ".join([item.get("question",""), item.get("answer","")]).lower()
    return sum(1 for kw in query_keywords if kw.lower() in blob)

def choose_best_candidate(
    user_query: str,
    language: str,
    candidates: List[Dict[str, Any]],
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Select the best FAQ entry while preventing off-topic answers.

    Strategy:
    1) Compute keyword overlap from extracted keywords (guardrail).
    2) If overlap is informative, require at least 1 overlap.
    3) If overlap is NOT informative (keywords too few / too generic), do NOT let it override retrieval.
       In that case, fall back to the top retrieval hit (rerank_score if available, else fused_score).
    """
    if not candidates:
        return None

    kws = extract_keywords(user_query, language)

    # If keyword extraction produced nothing, we have no reliable gating signal.
    # Fall back to best retrieval hit (still FAQ-restricted, but avoids random “need/careful” matches).
    if not kws:
        if debug:
            print("[DEBUG] No extracted keywords → returning top candidate by retrieval ranking.")
        return candidates[0]

    # Heuristic: if keywords are too few, overlap gating becomes brittle and may prefer generic matches.
    # Example: "need", "careful" will overlap with many unrelated questions.
    # In this case, we treat overlap as weak evidence, not a strict requirement.
    overlap_is_weak = (len(kws) <= 2)

    scored = [(overlap_score(kws, c), c) for c in candidates]
    max_ov = max(ov for ov, _ in scored) if scored else 0

    if debug:
        print("\n[DEBUG] keywords:", kws)
        for ov, c in sorted(scored, key=lambda x: (-x[0], -x[1].get("fused_score", 0)))[:10]:
            rr = c.get("rerank_score")
            rr_s = f" rerank={rr:.3f}" if rr is not None else ""
            print(f"  overlap={ov} idx={c['index']} fused={c.get('fused_score', 0):.5f}{rr_s} | Q={c['question'][:80]}")
        print(f"[DEBUG] overlap_is_weak={overlap_is_weak} max_overlap={max_ov}")

    # If overlap provides signal (max overlap >=1) and keywords are not weak, enforce overlap>=1.
    if (not overlap_is_weak) and max_ov >= 1:
        passed = [c for ov, c in scored if ov >= 1]
        if not passed:
            return None

        # Prefer rerank_score if present (higher is better), else fused_score
        if passed[0].get("rerank_score") is not None:
            passed = sorted(passed, key=lambda c: (c["rerank_score"], c.get("fused_score", 0.0)), reverse=True)
        else:
            passed = sorted(passed, key=lambda c: c.get("fused_score", 0.0), reverse=True)
        return passed[0]

    # Otherwise, overlap is weak or uninformative:
    # rely on retrieval ranking (reranker if enabled, else fused_score ordering).
    # Since candidates are already ordered by the retriever, choose top-1.
    return candidates[0]


# ---------------------------
# LLM calls
# ---------------------------
def call_ollama_chat(model: str, messages: List[Dict[str, str]]) -> str:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def call_openai_chat(model: str, messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return out.choices[0].message.content

def system_prompt(language: str) -> str:
    if language == "fr":
        return (
            "Vous êtes un assistant bienveillant.\n"
            "RÈGLES:\n"
            "- Utilisez UNIQUEMENT les informations du CONTEXTE.\n"
            "- N'inventez pas d'informations médicales.\n"
            "- Si la réponse n'est pas dans le CONTEXTE, dites-le.\n"
            "- Pas de conseils médicaux personnalisés.\n"
            "Ton: chaleureux, clair.\n"
        )
    return (
        "You are a compassionate assistant.\n"
        "RULES:\n"
        "- Use ONLY the CONTEXT.\n"
        "- Do NOT invent medical facts.\n"
        "- If the answer isn't in the CONTEXT, say so.\n"
        "- No personalized medical advice.\n"
        "Tone: warm, clear.\n"
    )

def format_faq_answer(top: Dict[str, Any], language: str) -> str:
    section = top["section"]
    q = top["question"]
    a = top["answer"].strip()
    if language == "fr":
        return (
            "Voici ce que la FAQ indique à ce sujet :\n\n"
            f"{a}\n\n"
            f"— Source FAQ : « {q} » (section : {section})\n\n"
            "Pour toute décision médicale personnalisée, parlez-en avec votre équipe soignante."
        )
    return (
        "Here is what the FAQ says about this topic (this does not replace advice from your care team):\n\n"
        f"{a}\n\n"
        f"— FAQ source: “{q}” (section: {section})\n\n"
        "If you’re considering a personal medical decision, it’s best to discuss it with your oncology team."
    )

def answer_with_llm(
    language: str,
    provider: str,
    openai_model: str,
    ollama_model: str,
    user_query: str,
    top: Dict[str, Any],
) -> str:
    section = top["section"]
    q = top["question"]
    a = top["answer"].strip()

    if language == "fr":
        source_line = f"— Source FAQ : « {q} » (section : {section})"
        user = (
            f"Question utilisateur:\n{user_query}\n\n"
            f"CONTEXTE (utiliser uniquement ceci):\n{a}\n\n"
            "Consignes:\n"
            "- Répondre avec empathie.\n"
            "- Ne pas ajouter d'informations médicales.\n"
            "- Reprendre fidèlement le contenu du contexte.\n"
            "- Finir par la ligne de source EXACTE ci-dessous.\n\n"
            f"Ligne de source:\n{source_line}\n"
        )
    else:
        source_line = f"— FAQ source: “{q}” (section: {section})"
        user = (
            f"User question:\n{user_query}\n\n"
            f"CONTEXT (use only this):\n{a}\n\n"
            "Instructions:\n"
            "- Answer with empathy.\n"
            "- Do not add medical facts.\n"
            "- Stay faithful to the context.\n"
            "- End with the EXACT source line below.\n\n"
            f"Source line:\n{source_line}\n"
        )

    messages = [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": user},
    ]
    if provider == "openai":
        return call_openai_chat(openai_model, messages)
    return call_ollama_chat(ollama_model, messages)


# ---------------------------
# Hybrid Retriever
# ---------------------------
class HybridFAQRetriever:
    def __init__(
        self,
        prefix: str,
        language: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_rerank: bool = False,
        reranker_model: str = DEFAULT_RERANKER,
    ):
        self.prefix = prefix
        self.language = language

        self.index_q = faiss.read_index(f"data/{prefix}_index_q.faiss")
        self.index_qa = faiss.read_index(f"data/{prefix}_index_qa.faiss")

        with open(f"data/{prefix}_qa.pkl", "rb") as f:
            qa = pickle.load(f)
        self.items = qa["items"]

        with open(f"data/{prefix}_bm25.pkl", "rb") as f:
            bm = pickle.load(f)
        self.bm25 = bm["bm25"]

        self.embedder = SentenceTransformer(embedding_model)

        self.enable_rerank = enable_rerank
        self.reranker = CrossEncoder(reranker_model) if enable_rerank else None

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def _search_faiss(self, index: faiss.Index, query: str, k: int) -> List[int]:
        qv = self._embed(query)
        _, inds = index.search(qv, k)
        return [int(i) for i in inds[0] if int(i) != -1]

    def _search_bm25(self, query: str, k: int) -> List[int]:
        toks = tokenize(query)
        scores = self.bm25.get_scores(toks)
        top = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top]

    @staticmethod
    def rrf_fuse(lists: List[List[int]], k: int = 60) -> Dict[int, float]:
        fused: Dict[int, float] = {}
        for lst in lists:
            for rank, doc_id in enumerate(lst):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def retrieve(self, query: str, top_k: int = 10, recall_k: int = 60, debug: bool = False) -> List[Dict[str, Any]]:
        q_ids = self._search_faiss(self.index_q, query, recall_k)
        qa_ids = self._search_faiss(self.index_qa, query, recall_k)
        bm_ids = self._search_bm25(query, recall_k)

        fused = self.rrf_fuse([q_ids, qa_ids, bm_ids], k=60)
        candidates = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[: max(100, recall_k)]

        rerank_scores = None
        if self.enable_rerank and candidates:
            cand = candidates[: min(40, len(candidates))]
            pairs = []
            for i in cand:
                it = self.items[i]
                text = f"{it['section']}\nQ: {it['question']}\nA: {it['answer']}"
                pairs.append((query, text))
            scores = self.reranker.predict(pairs)
            rerank_scores = {cand[i]: float(scores[i]) for i in range(len(cand))}
            candidates = sorted(cand, key=lambda i: rerank_scores[i], reverse=True)

        out = []
        for idx in candidates[:top_k]:
            it = self.items[idx]
            out.append(
                {
                    "index": idx,
                    "fused_score": float(fused.get(idx, 0.0)),
                    "rerank_score": float(rerank_scores[idx]) if rerank_scores else None,
                    "section": it["section"],
                    "question": it["question"],
                    "answer": it["answer"],
                }
            )

        if debug:
            print("\n[DEBUG] Retrieved candidates:")
            for r in out[:10]:
                print(
                    f"  idx={r['index']} fused={r['fused_score']:.5f}"
                    + (f" rerank={r['rerank_score']:.3f}" if r["rerank_score"] is not None else "")
                    + f" | Q={r['question'][:90]}"
                )
        return out

    def grep(self, keyword: str) -> List[Dict[str, Any]]:
        kw = (keyword or "").lower()
        hits = []
        for i, it in enumerate(self.items):
            blob = (it["question"] + "\n" + it["answer"]).lower()
            if kw in blob:
                hits.append({"index": i, **it})
        return hits
