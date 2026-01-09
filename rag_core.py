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
    "depression","hot","flushes","bone","osteoporosis","cholesterol","cardiovascular","heart",
}

FR_KEEP = {
    "grossesse","récidive","recidive","tamoxifène","tamoxifene","létrozole","letrozole",
    "anastrozole","exemestane","aromatase","inhibiteur","thrombose","embolie",
    "fertilité","fertilite","enfant","pause","soleil","uv","irm","mammographie",
    "dépression","depression","bouffées","bouffees","os","ostéoporose","osteoporose",
    "cholestérol","cholesterol","cardiovasculaire","coeur","cœur",
}

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
                candidates.sort(
                    key=lambda c: c.rerank_score if c.rerank_score is not None else -1e9,
                    reverse=True,
                )
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
        max_tokens: int = 600,
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
                "Tu es un assistant empathique. "
                "Tu dois UNIQUEMENT reformuler le texte fourni, sans ajouter, supprimer ou modifier des faits. "
                "Aucune déduction clinique, aucun conseil personnalisé. "
                "Ne fais pas de méta-commentaires (ex: 'Voici la reformulation'). "
                "Utilise des formulations et des structures de phrases différentes dans la mesure du possible, tout en conservant le sens."
                "Conserve tous les paragraphes et listes à puces. "
                "Ne mentionne pas 'Block' ou 'Section:' dans la réponse."
            )
        return (
            "You are an empathetic assistant. "
            "You must ONLY rephrase the provided text without adding, removing, or changing facts. "
            "No clinical inferences, no personalized advice. "
            "Do not add meta-commentary (e.g., 'Here is the rephrased text'). "
            "Use different wording and sentence structure where possible while preserving meaning."
            "Keep all paragraphs and bullet points. "
            "Do not mention 'Block' or include 'Section:' in the output."
        )

    def _user_prompt(self, user_query: str, text_to_rephrase: str) -> str:
        if self.language == "fr":
            return (
                f"Question de l'utilisateur:\n{user_query}\n\n"
                "Texte à reformuler fidèlement (sans nouveaux faits):\n"
                f"{text_to_rephrase}\n\n"
                "Réponse (reformulation fidèle, chaleureuse):"
            )
        return (
            f"User question:\n{user_query}\n\n"
            "Text to rephrase faithfully (no new facts):\n"
            f"{text_to_rephrase}\n\n"
            "Answer (faithful rephrase, warm tone):"
        )

    def rephrase(self, user_query: str, text_to_rephrase: str) -> str:
        import urllib.request

        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._user_prompt(user_query, text_to_rephrase),
            "system": self._system_prompt(),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        # check if LLM was callsed
        if os.getenv("HORMONAI_LLM_DEBUG", "0") == "1":
            print(f"[LLM] Calling Ollama model={self.model} host={os.getenv('OLLAMA_HOST','http://localhost:11434')}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            txt = (out.get("response") or "").strip()
            return txt or text_to_rephrase
        except Exception as e:
            if os.getenv("HORMONAI_LLM_DEBUG", "0") == "1":
                print(f"[LLM] Ollama call failed: {type(e).__name__}: {e}")
            return text_to_rephrase



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


def _score_candidate(c: RetrievalCandidate) -> float:
    return float(c.rerank_score) if c.rerank_score is not None else float(c.fused_score)


def _is_grounded(c: RetrievalCandidate, anchors: List[str]) -> bool:
    if _contains_any_anchor(c.question, anchors):
        return True
    if _contains_any_anchor(f"{c.question} {c.section}", anchors):
        return True
    if _contains_any_anchor(f"{c.question} {c.section} {c.answer}", anchors):
        return True
    return False


def _find_best_grounded_candidate(candidates: List[RetrievalCandidate], anchors: List[str]) -> Optional[RetrievalCandidate]:
    ranked = sorted(candidates, key=_score_candidate, reverse=True)
    for c in ranked:
        if _is_grounded(c, anchors):
            return c
    return None


def _select_close_bundle(
    candidates: List[RetrievalCandidate],
    anchors: List[str],
    max_n: int = 3,
    close_delta_rerank: float = 0.15,
    close_delta_fused: float = 0.003,
) -> List[RetrievalCandidate]:
    if not candidates:
        return []

    ranked = sorted(candidates, key=_score_candidate, reverse=True)

    best = None
    for c in ranked:
        if _is_grounded(c, anchors):
            best = c
            break
    if best is None:
        return []

    best_score = _score_candidate(best)
    has_rerank = best.rerank_score is not None
    close_delta = close_delta_rerank if has_rerank else close_delta_fused

    bundle: List[RetrievalCandidate] = [best]
    used_idx = {best.index}

    for c in ranked:
        if c.index in used_idx:
            continue
        if not _is_grounded(c, anchors):
            continue
        if (best_score - _score_candidate(c)) <= close_delta:
            bundle.append(c)
            used_idx.add(c.index)
        if len(bundle) >= max_n:
            break

    return bundle


def _format_bundle_body(language: str, bundle: List[RetrievalCandidate]) -> str:
    """
    This is what we feed to the LLM (if enabled): human-readable, no block markers,
    no section labels, no citations.
    """
    lang = _norm_lang(language)
    if not bundle:
        return ""

    parts: List[str] = []
    for c in bundle:
        # Bold question title for readability in the final answer
        parts.append(f"**{c.question.strip()}**\n{c.answer.strip()}")

    return "\n\n".join(parts).strip()


def _format_preface(language: str) -> str:
    lang = _norm_lang(language)
    if lang == "fr":
        return "**Voici ce que dit la FAQ sur ce sujet (cela ne remplace pas l’avis de votre équipe soignante) :**\n\n"
    return "**Here is what the FAQ says about this topic (this does not replace advice from your care team):**\n\n"


def _format_sources(language: str, bundle: List[RetrievalCandidate]) -> str:
    lang = _norm_lang(language)
    lines: List[str] = []
    if lang == "fr":
        for c in bundle:
            lines.append(f"**— Source FAQ :** “{c.question}” (section : {c.section})")
    else:
        for c in bundle:
            lines.append(f"**— FAQ source:** “{c.question}” (section: {c.section})")
    return "\n\n".join(lines).strip()


def _format_full_answer(language: str, body: str, sources: str) -> str:
    pre = _format_preface(language)
    if sources:
        return (pre + (body or "").strip() + "\n\n" + sources).strip()
    return (pre + (body or "").strip()).strip()


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

    # Close-score bundle to avoid omitting nearly-tied relevant answers
    bundle = _select_close_bundle(
        candidates=candidates,
        anchors=anchors,
        max_n=3,
        close_delta_rerank=0.15,
        close_delta_fused=0.003,
    )

    if debug:
        dbg["chosen_best"] = {"idx": best.index, "q": best.question, "score": _score_candidate(best)}
        dbg["bundle"] = [{"idx": c.index, "q": c.question, "score": _score_candidate(c)} for c in bundle]

    if not bundle:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    body = _format_bundle_body(lang, bundle)
    sources = _format_sources(lang, bundle)

    # No-LLM answer
    final_body = body

    # debug LLM usage
    dbg["use_llm"] = use_llm
    dbg["llm_model"] = llm_model
    dbg["llm_changed"] = (final_body.strip() != body.strip())

    # LLM rephrase ONLY when we have a valid FAQ answer body
    if use_llm:
        rephraser = LLMRephraser(language=lang, model=llm_model)
        final_body = rephraser.rephrase(user_query=user_query, text_to_rephrase=body).strip() or body

    answer_text = _format_full_answer(lang, final_body, sources)

    top = bundle[0]
    out = AnswerResult(
        answered=True,
        answer_text=answer_text,
        source_title=top.question,
        source_section=top.section,
        source_index=top.index,
        debug=(dbg if debug else None),
    )
    return out


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
    if "bundle" in result.debug:
        print("[DEBUG] bundle:", result.debug.get("bundle"))
