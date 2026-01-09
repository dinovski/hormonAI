#!/usr/bin/env python3
"""
rag_core.py

Core logic for hormonAI:
- hybrid retrieval (FAISS Q + FAISS QA + BM25 + optional rerank)
- answerability gating (corpus keyword presence + non-empty core keywords)
- response builder (NO LLM) OR optional LLM "rephrase only" (NEVER for abstains)

Change requested:
- DO NOT show "(Detected keywords: ...)" or any detected-word hints in abstain responses.
"""

from __future__ import annotations

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional deps (only needed when used)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore


# ---------------------------
# Language resources
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
    "while","taking","take","taken","during","using",
}

FR_STOPWORDS = {
    "le","la","les","un","une","des","et","ou","mais","si","alors",
    "de","du","dans","sur","pour","avec","par","au","aux","en",
    "est","sont","été","être","avoir","a","ont","ai","as","avons","avez",
    "je","tu","il","elle","nous","vous","ils","elles",
    "ce","cet","cette","ces",
    "quoi","pourquoi","comment","quand","où","quel","quelle","quels","quelles",
    "pendant","durant","prendre","pris","prise","utiliser",
}

GENERIC_EN = {
    "safe","safety","careful","need","should","can","could","would",
    "risk","danger","allowed","ok","okay","possible","recommend","recommended","advice",
    "hormone","hormonal","therapy","treatment","medication","pill","medicine","drug","drugs",
    "side","effects","effect",
}

GENERIC_FR = {
    "sûr","sure","sécurité","prudent","prudence","besoin","dois","devrais","peux",
    "risque","danger","autorisé","possible","recommandé","conseil",
    "hormone","hormonale","traitement","thérapie","médicament","comprimé","pilule",
    "effet","effets",
}

EN_KEEP = {
    "pregnancy","recurrence","tamoxifen","letrozole","anastrozole","exemestane",
    "aromatase","inhibitor","clot","clots","thrombosis","embolism",
    "fertility","child","pause","sun","uv","mri","mammogram",
    "depression","hot","flushes","bone","osteoporosis","cholesterol",
}

FR_KEEP = {
    "grossesse","récidive","recidive","tamoxifène","tamoxifene","létrozole","letrozole",
    "anastrozole","exemestane","aromatase","inhibiteur","thrombose","embolie",
    "fertilité","fertilite","enfant","pause","soleil","uv","irm","mammographie",
    "dépression","depression","bouffées","bouffees","os","ostéoporose","osteoporose",
    "cholestérol","cholesterol",
}


def _norm_lang(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang in ("fr", "fra", "french"):
        return "fr"
    return "en"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w\-']+\b", (text or "").lower())


def extract_core_keywords(user_query: str, language: str) -> List[str]:
    lang = _norm_lang(language)
    toks = _tokenize(user_query)

    if lang == "fr":
        stop = FR_STOPWORDS
        gen = GENERIC_FR
        keep = FR_KEEP
    else:
        stop = EN_STOPWORDS
        gen = GENERIC_EN
        keep = EN_KEEP

    out: List[str] = []
    for t in toks:
        if t in keep:
            out.append(t)
            continue
        if t in stop:
            continue
        if t in gen:
            continue
        if len(t) <= 2:
            continue
        out.append(t)

    seen = set()
    dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


@dataclass
class RetrievalCandidate:
    index: int
    question: str
    section: str
    answer: str
    fused_score: float
    bm25_rank: Optional[int] = None
    faiss_q_rank: Optional[int] = None
    faiss_qa_rank: Optional[int] = None
    rerank_score: Optional[float] = None


# ---------------------------
# LLM wrappers (optional)
# ---------------------------

class LLMRephraser:
    """
    Only rephrases text that already comes from the FAQ.
    It MUST NOT add new facts.
    """

    def __init__(
        self,
        provider: str,
        language: str,
        openai_model: str = "gpt-4o-mini",
        ollama_model: str = "llama3.2",
        temperature: float = 0.2,
        max_tokens: int = 450,
        timeout_s: int = 60,
    ):
        self.provider = (provider or "").lower().strip()
        self.language = _norm_lang(language)
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def rephrase(self, user_query: str, faq_answer: str) -> str:
        if self.provider == "openai":
            return self._rephrase_openai(user_query, faq_answer)
        if self.provider == "ollama":
            return self._rephrase_ollama(user_query, faq_answer)
        return faq_answer

    def _system_prompt(self) -> str:
        if self.language == "fr":
            return (
                "Tu es un assistant empathique, STRICTEMENT limité au texte fourni. "
                "Tu dois REFORMULER fidèlement le texte FAQ, sans ajouter, supprimer ou modifier des faits. "
                "Aucune déduction clinique. Aucun avis personnalisé. Conserve les puces si présentes."
            )
        return (
            "You are an empathetic assistant STRICTLY limited to provided text. "
            "You must REPHRASE the provided FAQ text without adding, removing, or changing facts. "
            "No clinical inferences. No personalized advice. Preserve bullet points if present."
        )

    def _user_prompt(self, user_query: str, faq_answer: str) -> str:
        if self.language == "fr":
            return (
                f"Question de l'utilisateur:\n{user_query}\n\n"
                f"Texte FAQ (à reformuler fidèlement):\n{faq_answer}\n\n"
                "Réponse (reformulation fidèle, chaleureuse, sans nouveaux faits):"
            )
        return (
            f"User question:\n{user_query}\n\n"
            f"FAQ text (rephrase faithfully):\n{faq_answer}\n\n"
            "Answer (faithful rephrase, warm tone, no new facts):"
        )

    def _rephrase_openai(self, user_query: str, faq_answer: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return faq_answer

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return faq_answer

        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(user_query, faq_answer)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_s,
            )
            txt = (resp.choices[0].message.content or "").strip()
            return txt or faq_answer
        except Exception:
            return faq_answer

    def _rephrase_ollama(self, user_query: str, faq_answer: str) -> str:
        import urllib.request

        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.ollama_model,
            "prompt": self._system_prompt() + "\n\n" + self._user_prompt(user_query, faq_answer),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read().decode("utf-8"))
            txt = (out.get("response") or "").strip()
            return txt or faq_answer
        except Exception:
            return faq_answer


# ---------------------------
# Retriever
# ---------------------------

class HybridFAQRetriever:
    def __init__(
        self,
        language: str = "en",
        data_dir: str = "data",
        top_k: int = 12,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        rerank: bool = False,
        rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ):
        self.language = _norm_lang(language)
        self.data_dir = data_dir
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self.rerank_enabled = rerank
        self.rerank_model_name = rerank_model

        self.qa_items: List[Dict[str, Any]] = []
        self.bm25 = None
        self.bm25_docs = None
        self.index_q = None
        self.index_qa = None

        self._embedder = None
        self._reranker = None
        self._corpus_tokens: set[str] = set()

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss not installed. pip install faiss-cpu")
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 not installed. pip install rank-bm25")

        prefix = f"faq_{self.language}"
        qa_path = os.path.join(self.data_dir, f"{prefix}_qa.pkl")
        bm25_path = os.path.join(self.data_dir, f"{prefix}_bm25.pkl")
        index_q_path = os.path.join(self.data_dir, f"{prefix}_index_q.faiss")
        index_qa_path = os.path.join(self.data_dir, f"{prefix}_index_qa.faiss")

        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"Missing {qa_path}. Run ingest_faq.py first.")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"Missing {bm25_path}. Run ingest_faq.py first.")
        if not os.path.exists(index_q_path) or not os.path.exists(index_qa_path):
            raise FileNotFoundError("Missing FAISS index files. Run ingest_faq.py first.")

        with open(qa_path, "rb") as f:
            blob = pickle.load(f)
        self.qa_items = blob["items"]

        with open(bm25_path, "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.bm25_docs = blob["bm25_docs"]

        self.index_q = faiss.read_index(index_q_path)
        self.index_qa = faiss.read_index(index_qa_path)

        self._embedder = SentenceTransformer(self.embedding_model_name)

        if self.rerank_enabled:
            if CrossEncoder is None:
                raise RuntimeError("cross-encoder rerank requested but CrossEncoder unavailable.")
            self._reranker = CrossEncoder(self.rerank_model_name)

        toks = []
        for it in self.qa_items:
            toks.extend(_tokenize(it.get("question", "")))
            toks.extend(_tokenize(it.get("section", "")))
        self._corpus_tokens = set(toks)

    def _embed(self, text: str) -> np.ndarray:
        emb = self._embedder.encode([text], convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb / norm

    def _bm25_topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_toks = _tokenize(query)
        scores = self.bm25.get_scores(q_toks)
        idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idxs]

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        if not user_query.strip():
            return []

        q_emb = self._embed(user_query)

        Dq, Iq = self.index_q.search(q_emb, self.top_k)
        faiss_q = [(int(idx), float(score)) for idx, score in zip(Iq[0], Dq[0]) if idx >= 0]

        Dqa, Iqa = self.index_qa.search(q_emb, self.top_k)
        faiss_qa = [(int(idx), float(score)) for idx, score in zip(Iqa[0], Dqa[0]) if idx >= 0]

        bm25 = self._bm25_topk(user_query, self.top_k)

        rank_q = {idx: r for r, (idx, _) in enumerate(faiss_q, start=1)}
        rank_qa = {idx: r for r, (idx, _) in enumerate(faiss_qa, start=1)}
        rank_b = {idx: r for r, (idx, _) in enumerate(bm25, start=1)}

        all_ids = set(rank_q) | set(rank_qa) | set(rank_b)

        candidates: List[RetrievalCandidate] = []
        for idx in all_ids:
            it = self.qa_items[idx]
            fused = 0.0
            if idx in rank_q:
                fused += self._rrf(rank_q[idx])
            if idx in rank_qa:
                fused += self._rrf(rank_qa[idx])
            if idx in rank_b:
                fused += self._rrf(rank_b[idx])

            candidates.append(
                RetrievalCandidate(
                    index=idx,
                    question=it.get("question", ""),
                    section=it.get("section", ""),
                    answer=it.get("answer", ""),
                    fused_score=fused,
                    faiss_q_rank=rank_q.get(idx),
                    faiss_qa_rank=rank_qa.get(idx),
                    bm25_rank=rank_b.get(idx),
                )
            )

        candidates.sort(key=lambda c: c.fused_score, reverse=True)

        if self._reranker is not None and candidates:
            topN = candidates[: min(18, len(candidates))]
            pairs = [(user_query, c.question) for c in topN]
            scores = self._reranker.predict(pairs)
            for c, s in zip(topN, scores):
                c.rerank_score = float(s)
            candidates.sort(
                key=lambda c: (
                    c.rerank_score if c.rerank_score is not None else -1e9,
                    c.fused_score
                ),
                reverse=True,
            )

        return candidates


# ---------------------------
# Answerability + response
# ---------------------------

@dataclass
class AnswerResult:
    answered: bool
    answer_text: str
    source_title: Optional[str] = None
    source_section: Optional[str] = None
    source_index: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None


def build_abstain(language: str) -> str:
    """
    Abstain message WITHOUT showing detected keywords or any internal signals.
    """
    lang = _norm_lang(language)
    if lang == "fr":
        return (
            "Je comprends votre question. Cependant, la FAQ sur laquelle je suis basé(e) ne semble pas "
            "contenir d’information spécifique sur ce sujet. Pour éviter d’inventer des informations médicales, "
            "je ne peux pas répondre à partir de la FAQ.\n\n"
            "Je vous recommande d’en parler avec votre équipe d’oncologie."
        )
    return (
        "I understand why you’re asking. However, the FAQ I’m based on does not appear to contain specific "
        "information about this topic. To avoid inventing medical information, I can’t answer this from the FAQ.\n\n"
        "Please discuss this with your oncology team."
    )


def format_answer_no_llm(language: str, candidate: RetrievalCandidate) -> AnswerResult:
    lang = _norm_lang(language)
    if lang == "fr":
        pre = "Voici ce que dit la FAQ sur ce sujet (cela ne remplace pas l’avis de votre équipe soignante) :\n\n"
        src = f"— Source FAQ : “{candidate.question}” (section : {candidate.section})"
    else:
        pre = "Here is what the FAQ says about this topic (this does not replace advice from your care team):\n\n"
        src = f"— FAQ source: “{candidate.question}” (section: {candidate.section})"

    text = pre + candidate.answer.strip() + "\n\n" + src
    return AnswerResult(
        answered=True,
        answer_text=text,
        source_title=candidate.question,
        source_section=candidate.section,
        source_index=candidate.index,
    )


def answer_query(
    retriever: HybridFAQRetriever,
    user_query: str,
    use_llm: bool = False,
    llm_provider: str = "ollama",
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "llama3.2",
    debug: bool = False,
) -> AnswerResult:
    """
    1) retrieve candidates
    2) answerability gate:
       - require >=1 core keyword
       - require at least one core keyword appears in corpus (questions/sections)
    3) if answerable, answer from best candidate; optionally rephrase with LLM
       IMPORTANT: LLM only runs when answered=True
    """
    lang = retriever.language
    candidates = retriever.retrieve(user_query)

    core_kws = extract_core_keywords(user_query, lang)
    in_corpus = [k for k in core_kws if k in retriever._corpus_tokens]  # noqa: SLF001

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["core_kws"] = core_kws
        dbg["in_corpus"] = in_corpus
        dbg["top_candidates"] = [
            {
                "idx": c.index,
                "fused": round(c.fused_score, 5),
                "rerank": (round(c.rerank_score, 3) if c.rerank_score is not None else None),
                "q": c.question[:120],
            }
            for c in candidates[:10]
        ]

    if len(core_kws) < 1:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    if len(in_corpus) < 1:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    if not candidates:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    best = candidates[0]
    base = format_answer_no_llm(lang, best)

    if use_llm:
        rephraser = LLMRephraser(
            provider=llm_provider,
            language=lang,
            openai_model=openai_model,
            ollama_model=ollama_model,
        )

        rephrased_answer = rephraser.rephrase(user_query=user_query, faq_answer=best.answer.strip())

        if lang == "fr":
            pre = "Voici ce que dit la FAQ sur ce sujet (cela ne remplace pas l’avis de votre équipe soignante) :\n\n"
            src = f"— Source FAQ : “{best.question}” (section : {best.section})"
        else:
            pre = "Here is what the FAQ says about this topic (this does not replace advice from your care team):\n\n"
            src = f"— FAQ source: “{best.question}” (section: {best.section})"

        base.answer_text = pre + rephrased_answer.strip() + "\n\n" + src

    if debug:
        base.debug = dbg

    return base


def print_debug(result: AnswerResult) -> None:
    if not result.debug:
        return
    print("\n[DEBUG] core_kws:", result.debug.get("core_kws"))
    print("[DEBUG] in_corpus:", result.debug.get("in_corpus"))
    print("[DEBUG] Retrieved candidates:")
    for row in result.debug.get("top_candidates", []):
        print(f"  idx={row['idx']} fused={row['fused']} rerank={row['rerank']} | Q={row['q']}")
