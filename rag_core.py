from __future__ import annotations

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------------------------
# Stopwords / keyword gating
# ---------------------------

EN_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","so","because",
    "to","of","in","on","for","with","as","at","by","from","into","about",
    "is","are","was","were","be","been","being",
    "do","does","did","doing",
    "while","taking","take","taken","during","using",
    "can","could","should","would","will","may","might","must",
    "i","you","we","they","he","she","it","my","your","our","their",
    "this","that","these","those",
    "what","why","how","when","where","which",
}

FR_STOPWORDS = {
    "le","la","les","un","une","des","et","ou","mais","si","alors",
    "de","du","dans","sur","pour","avec","par","au","aux","en",
    "est","sont","été","etre","être","avoir","a","ont",
    "je","tu","il","elle","nous","vous","ils","elles",
    "ce","cet","cette","ces",
    "quoi","pourquoi","comment","quand","où","ou","quel","quelle","quels","quelles",
}

GENERIC_EN = {
    "safe","safety","careful","need","should","can","could","would","risk","danger",
    "allowed","ok","okay","possible","recommend","recommended","advice",
    "hormone","hormonal","therapy","treatment","medication","pill",
    "medicine","drug","drugs",
}

GENERIC_FR = {
    "sûr","sur","sure","sécurité","securite","prudent","prudence","besoin","dois","devrais","peux",
    "risque","danger","autorisé","autorise","possible","recommandé","recommande","conseil",
    "hormone","hormonale","traitement","thérapie","therapie","médicament","medicament","comprimé","comprime",
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

# Drug/treatment words should NOT count as “anchors”
DRUG_TREATMENT_EN = {
    "tamoxifen","letrozole","anastrozole","exemestane",
    "aromatase","inhibitor","inhibitors",
    "hormone","hormonal","therapy","treatment","medication","pill","medicine","drug","drugs",
}

DRUG_TREATMENT_FR = {
    "tamoxifène","tamoxifene","létrozole","letrozole","anastrozole","exemestane",
    "aromatase","inhibiteur","inhibiteurs",
    "hormone","hormonale","hormonothérapie","hormonotherapie",
    "traitement","thérapie","therapie","médicament","medicament","comprimé","comprime","pilule",
}


def _norm_lang(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang in ("fr", "fra", "french", "français", "francais"):
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


def anchor_keywords(core_kws: List[str], language: str) -> List[str]:
    lang = _norm_lang(language)
    drugset = DRUG_TREATMENT_FR if lang == "fr" else DRUG_TREATMENT_EN
    anchors = [k for k in core_kws if k not in drugset]
    seen = set()
    out = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _contains_any_anchor(text: str, anchors: List[str]) -> bool:
    hay = (text or "").lower()
    return any(a.lower() in hay for a in anchors)


# ---------------------------
# Retrieval dataclasses
# ---------------------------

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


@dataclass
class AnswerResult:
    answered: bool
    answer_text: str
    source_title: Optional[str] = None
    source_section: Optional[str] = None
    source_index: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None


# ---------------------------
# Hybrid Retriever
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

        self.rerank = rerank
        self.rerank_model = rerank_model

        self._items: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[List[str]] = []

        self._index_q: Optional[faiss.Index] = None
        self._index_qa: Optional[faiss.Index] = None

        self._embedder: Optional[SentenceTransformer] = None

        # tokens from questions+sections only
        self._corpus_tokens: set[str] = set()

        self._cross_encoder = None

    def load(self) -> None:
        prefix = f"faq_{self.language}"
        qa_path = os.path.join(self.data_dir, f"{prefix}_qa.pkl")
        bm25_path = os.path.join(self.data_dir, f"{prefix}_bm25.pkl")
        idx_q_path = os.path.join(self.data_dir, f"{prefix}_index_q.faiss")
        idx_qa_path = os.path.join(self.data_dir, f"{prefix}_index_qa.faiss")

        with open(qa_path, "rb") as f:
            payload = pickle.load(f)
        self._items = payload["items"]

        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_docs = payload["bm25_docs"]

        self._index_q = faiss.read_index(idx_q_path)
        self._index_qa = faiss.read_index(idx_qa_path)

        self._embedder = SentenceTransformer(self.embedding_model_name)

        corpus_text = " ".join([f"{it.get('section','')} {it.get('question','')}" for it in self._items])
        self._corpus_tokens = set(_tokenize(corpus_text))

        if self.rerank:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.rerank_model)
            except Exception:
                self._cross_encoder = None

    def _encode(self, text: str) -> np.ndarray:
        assert self._embedder is not None
        emb = self._embedder.encode([text], convert_to_numpy=True)
        emb = emb.astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        if not self._items:
            return []

        assert self._bm25 is not None
        assert self._index_q is not None
        assert self._index_qa is not None

        q_tokens = _tokenize(user_query)
        bm25_scores = self._bm25.get_scores(q_tokens)
        bm25_ranked = np.argsort(-bm25_scores)[: self.top_k]

        q_emb = self._encode(f"Question: {user_query}")
        _, Iq = self._index_q.search(q_emb, self.top_k)

        qa_emb = self._encode(f"Question: {user_query}")
        _, Iqa = self._index_qa.search(qa_emb, self.top_k)

        def rrf(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank)

        fused: Dict[int, float] = {}

        for r, idx in enumerate(bm25_ranked.tolist()):
            fused[idx] = fused.get(idx, 0.0) + rrf(r)
        for r, idx in enumerate(Iq[0].tolist()):
            fused[idx] = fused.get(idx, 0.0) + rrf(r)
        for r, idx in enumerate(Iqa[0].tolist()):
            fused[idx] = fused.get(idx, 0.0) + rrf(r)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        candidates: List[RetrievalCandidate] = []
        for idx, score in ranked:
            it = self._items[int(idx)]
            candidates.append(
                RetrievalCandidate(
                    index=int(idx),
                    question=str(it.get("question", "")),
                    section=str(it.get("section", "")),
                    answer=str(it.get("answer", "")),
                    fused_score=float(score),
                )
            )

        if self._cross_encoder is not None and len(candidates) >= 2:
            pairs = [(user_query, f"{c.section}\n{c.question}\n{c.answer}") for c in candidates]
            try:
                ce_scores = self._cross_encoder.predict(pairs)
                for c, s in zip(candidates, ce_scores):
                    c.rerank_score = float(s)
                candidates.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -1e9, reverse=True)
            except Exception:
                pass

        return candidates


# ---------------------------
# LLM (Ollama-only rephrase)
# ---------------------------

class LLMRephraser:
    def __init__(
        self,
        language: str,
        model: str = "llama3.2",
        temperature: float = 0.2,
        max_tokens: int = 450,
        timeout_s: int = 60,
    ):
        self.language = _norm_lang(language)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

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

    def rephrase(self, user_query: str, faq_answer: str) -> str:
        import urllib.request

        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._user_prompt(user_query, faq_answer),
            "system": self._system_prompt(),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            txt = (out.get("response") or "").strip()
            return txt or faq_answer
        except Exception:
            return faq_answer


# ---------------------------
# Answer formatting + gating
# ---------------------------

def build_abstain(language: str) -> str:
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
        pre = "**Voici ce que dit la FAQ sur ce sujet (cela ne remplace pas l’avis de votre équipe soignante) :**\n\n"
        src = f"**— Source FAQ :** “{candidate.question}” (section : {candidate.section})"
    else:
        pre = "**Here is what the FAQ says about this topic (this does not replace advice from your care team):**\n\n"
        src = f"**— FAQ source:** “{candidate.question}” (section: {candidate.section})"

    text = pre + candidate.answer.strip() + "\n\n" + src
    return AnswerResult(
        answered=True,
        answer_text=text,
        source_title=candidate.question,
        source_section=candidate.section,
        source_index=candidate.index,
    )


def _score_candidate(c: RetrievalCandidate) -> float:
    return float(c.rerank_score) if c.rerank_score is not None else float(c.fused_score)


def _find_best_grounded_candidate(candidates: List[RetrievalCandidate], anchors: List[str]) -> Optional[RetrievalCandidate]:
    ranked = sorted(candidates, key=_score_candidate, reverse=True)

    for c in ranked:
        if _contains_any_anchor(c.question, anchors):
            return c

    for c in ranked:
        if _contains_any_anchor(f"{c.question} {c.section}", anchors):
            return c

    for c in ranked:
        if _contains_any_anchor(f"{c.question} {c.section} {c.answer}", anchors):
            return c

    return None


def answer_query(
    retriever: HybridFAQRetriever,
    user_query: str,
    use_llm: bool = False,
    llm_model: str = "llama3.2",
    debug: bool = False,
) -> AnswerResult:
    lang = retriever.language
    candidates = retriever.retrieve(user_query)

    core_kws = extract_core_keywords(user_query, lang)
    anchors = anchor_keywords(core_kws, lang)
    in_corpus_anchors = [k for k in anchors if k in retriever._corpus_tokens]  # noqa

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["core_kws"] = core_kws
        dbg["anchors"] = anchors
        dbg["in_corpus_anchors"] = in_corpus_anchors
        dbg["top_candidates"] = [
            {
                "idx": c.index,
                "fused": round(c.fused_score, 5),
                "rerank": (round(c.rerank_score, 3) if c.rerank_score is not None else None),
                "q": c.question[:120],
            }
            for c in candidates[:10]
        ]

    if len(core_kws) < 1 or len(anchors) < 1 or len(in_corpus_anchors) < 1 or not candidates:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    best = _find_best_grounded_candidate(candidates, anchors)
    if best is None:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    # margin check (kept)
    anchor_in_question = _contains_any_anchor(best.question, anchors)

    ranked = sorted(candidates, key=_score_candidate, reverse=True)
    best_score = _score_candidate(best)

    second_grounded = None
    for c in ranked:
        if c.index == best.index:
            continue
        if _contains_any_anchor(f"{c.question} {c.section}", anchors) or _contains_any_anchor(
            f"{c.question} {c.section} {c.answer}", anchors
        ):
            second_grounded = c
            break

    second_score = _score_candidate(second_grounded) if second_grounded else 0.0
    margin = best_score - second_score
    has_rerank = best.rerank_score is not None
    min_margin = 0.08 if has_rerank else 0.002

    if debug:
        dbg["chosen_best"] = {"idx": best.index, "q": best.question, "score": best_score}
        dbg["anchor_in_question"] = anchor_in_question
        dbg["margin"] = margin
        dbg["min_margin"] = min_margin

    if (not anchor_in_question) and second_grounded is not None and margin < min_margin:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    base = format_answer_no_llm(lang, best)

    if use_llm:
        rephraser = LLMRephraser(language=lang, model=llm_model)
        rephrased_answer = rephraser.rephrase(user_query=user_query, faq_answer=best.answer.strip())

        if lang == "fr":
            pre = "**Voici ce que dit la FAQ sur ce sujet (cela ne remplace pas l’avis de votre équipe soignante) :**\n\n"
            src = f"**— Source FAQ :** “{best.question}” (section : {best.section})"
        else:
            pre = "**Here is what the FAQ says about this topic (this does not replace advice from your care team):**\n\n"
            src = f"**— FAQ source:** “{best.question}” (section: {best.section})"

        base.answer_text = pre + rephrased_answer.strip() + "\n\n" + src

    if debug:
        base.debug = dbg

    return base


def print_debug(result: AnswerResult) -> None:
    if not result.debug:
        return
    print("\n[DEBUG] core_kws:", result.debug.get("core_kws"))
    print("[DEBUG] anchors:", result.debug.get("anchors"))
    print("[DEBUG] in_corpus_anchors:", result.debug.get("in_corpus_anchors"))
    print("[DEBUG] Retrieved candidates:")
    for row in result.debug.get("top_candidates", []):
        print(f"  idx={row['idx']} fused={row['fused']} rerank={row['rerank']} | Q={row['q']}")
    if "chosen_best" in result.debug:
        print("[DEBUG] chosen_best:", result.debug.get("chosen_best"))
        print("[DEBUG] anchor_in_question:", result.debug.get("anchor_in_question"))
        print("[DEBUG] margin:", result.debug.get("margin"), "min_margin:", result.debug.get("min_margin"))
