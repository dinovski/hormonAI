from __future__ import annotations

import os
import re
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import snowballstemmer  # type: ignore
except Exception:
    snowballstemmer = None


# ---------------------------
# Language + tokenization
# ---------------------------

def _norm_lang(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang in ("fr", "fra", "french", "français", "francais"):
        return "fr"
    return "en"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w\-']+\b", (text or "").lower())


# ---------------------------
# Stopwords / keyword gating
# ---------------------------

EN_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","so","because",
    "to","of","in","on","for","with","as","at","by","from","into","about",
    "is","are","was","were","be","been","being",
    "do","does","did","doing",
    "while","during","using",
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
    "hormone","hormonal","therapy","treatment","medication","pill","medicine","drug","drugs",
    "take","taking","taken",
    "get","getting","got",
}

GENERIC_FR = {
    "sûr","sur","sure","sécurité","securite","prudent","prudence","besoin","dois","devrais","peux",
    "risque","danger","autorisé","autorise","possible","recommandé","recommande","conseil",
    "hormone","hormonale","hormonothérapie","hormonotherapie",
    "traitement","thérapie","therapie","médicament","medicament","comprimé","comprime","pilule",
    "prendre","prends","pris",
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


# ---------------------------
# Stemming (Snowball preferred)
# ---------------------------

_STEMMER_CACHE: Dict[str, Any] = {}


def _snowball_stem(token: str, language: str) -> str:
    assert snowballstemmer is not None
    lang = _norm_lang(language)
    key = "english" if lang == "en" else "french"
    if key not in _STEMMER_CACHE:
        _STEMMER_CACHE[key] = snowballstemmer.stemmer(key)
    return _STEMMER_CACHE[key].stemWord((token or "").lower())


def _fallback_stem_en(token: str) -> str:
    t = (token or "").lower()
    if len(t) <= 2:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("es") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]
    for suf in ("ing", "ed", "ly"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t


def _fallback_stem_fr(token: str) -> str:
    t = (token or "").lower()
    if len(t) <= 2:
        return t
    if t.endswith("es") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]
    for suf in ("ements","ement","ations","ation","ateurs","ateur","ées","ée","er","ir","re"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t


def _stem(token: str, language: str) -> str:
    if snowballstemmer is not None:
        try:
            return _snowball_stem(token, language)
        except Exception:
            pass
    return _fallback_stem_fr(token) if _norm_lang(language) == "fr" else _fallback_stem_en(token)


def _stem_set(tokens: List[str], language: str) -> Set[str]:
    return {_stem(t, language) for t in tokens if t}


# ---------------------------
# Keyword extraction / anchors
# ---------------------------

def extract_core_keywords(user_query: str, language: str) -> List[str]:
    lang = _norm_lang(language)
    toks = _tokenize(user_query)

    stop = FR_STOPWORDS if lang == "fr" else EN_STOPWORDS
    gen = GENERIC_FR if lang == "fr" else GENERIC_EN

    out: List[str] = []
    for t in toks:
        if t in stop:
            continue
        if t in gen:
            continue
        if len(t) <= 2:
            continue
        out.append(t)

    seen = set()
    dedup: List[str] = []
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
    out: List[str] = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _contains_any_anchor(text: str, anchors: List[str], language: str) -> bool:
    hay = (text or "").lower()
    if any(a.lower() in hay for a in anchors):
        return True
    hay_stems = _stem_set(_tokenize(hay), language)
    anchor_stems = _stem_set([a.lower() for a in anchors], language)
    return len(hay_stems.intersection(anchor_stems)) > 0


# ---------------------------
# Dataclasses
# ---------------------------

@dataclass
class RetrievalCandidate:
    index: int
    question: str
    section: str
    answer: str
    fused_score: float
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
        self._index_q: Optional[faiss.Index] = None
        self._index_qa: Optional[faiss.Index] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._cross_encoder = None

        self._corpus_tokens: Set[str] = set()
        self._corpus_stems: Set[str] = set()

        self._stored_embedding_model_name: Optional[str] = None

    def load(self) -> None:
        prefix = f"faq_{self.language}"
        qa_path = os.path.join(self.data_dir, f"{prefix}_qa.pkl")
        bm25_path = os.path.join(self.data_dir, f"{prefix}_bm25.pkl")
        idx_q_path = os.path.join(self.data_dir, f"{prefix}_index_q.faiss")
        idx_qa_path = os.path.join(self.data_dir, f"{prefix}_index_qa.faiss")

        with open(qa_path, "rb") as f:
            payload = pickle.load(f)
        self._items = payload["items"]
        self._stored_embedding_model_name = payload.get("embedding_model_name")

        with open(bm25_path, "rb") as f:
            payload_bm25 = pickle.load(f)
        self._bm25 = payload_bm25["bm25"]

        self._index_q = faiss.read_index(idx_q_path)
        self._index_qa = faiss.read_index(idx_qa_path)

        self._embedder = SentenceTransformer(self.embedding_model_name)

        expected_dim = int(getattr(self._index_q, "d", -1))
        model_dim = int(self._embedder.get_sentence_embedding_dimension())
        if expected_dim > 0 and expected_dim != model_dim:
            hint = ""
            if self._stored_embedding_model_name:
                hint = (
                    f" (ingest used: '{self._stored_embedding_model_name}'). "
                    "Fix: re-run ingest_faq.py with the same --embedding-model as the chatbot, "
                    "or run the chatbot with --embedding-model set to the ingest model."
                )
            raise ValueError(
                f"Embedding dimension mismatch: FAISS index expects d={expected_dim} "
                f"but embedding model '{self.embedding_model_name}' outputs d={model_dim}.{hint}"
            )

        corpus_text = " ".join(
            [f"{it.get('section','')} {it.get('question','')} {it.get('answer','')}" for it in self._items]
        )
        toks = _tokenize(corpus_text)
        self._corpus_tokens = set(toks)
        self._corpus_stems = _stem_set(toks, self.language)

        if self.rerank:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.rerank_model)
            except Exception:
                self._cross_encoder = None

    def _encode(self, text: str) -> np.ndarray:
        assert self._embedder is not None
        emb = self._embedder.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        if not self._items:
            return []

        assert self._bm25 is not None
        assert self._index_q is not None
        assert self._index_qa is not None

        bm25_scores = self._bm25.get_scores(_tokenize(user_query))
        bm25_ranked = np.argsort(-bm25_scores)[: self.top_k]

        q_emb = self._encode(f"Question: {user_query}")
        _, Iq = self._index_q.search(q_emb, self.top_k)
        qa_emb = self._encode(f"Question: {user_query}")
        _, Iqa = self._index_qa.search(qa_emb, self.top_k)

        def rrf(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank)

        fused: Dict[int, float] = {}

        for r, idx in enumerate(bm25_ranked.tolist()):
            fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)
        for r, idx in enumerate(Iq[0].tolist()):
            if idx >= 0:
                fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)
        for r, idx in enumerate(Iqa[0].tolist()):
            if idx >= 0:
                fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)

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
# Formatting functions (requested)
# ---------------------------

def _format_bundle_body(language: str, bundle: List[RetrievalCandidate]) -> str:
    """
    This is what we feed to the LLM (if enabled): human-readable, no block markers,
    no section labels, no citations.
    """
    if not bundle:
        return ""
    parts: List[str] = []
    for c in bundle:
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


# ---------------------------
# Empathy bank (generic, question-agnostic)
# ---------------------------

_EMPATHY_BANK_EN = [
    "I’m glad you asked — it’s completely understandable to have questions about this.",
    "Thank you for asking. It makes sense to want clarity and reassurance.",
    "I hear you — these questions are very valid, and you’re not alone in wondering.",
    "It’s understandable to want a clear answer here, especially with everything you’re managing.",
    "That’s a reasonable question to ask.",
    "I’m here with you — it’s okay to seek reassurance and clear information.",
    "It makes sense to want to feel confident about what to do next.",
    "Thank you for sharing your question — it’s important to feel supported and informed.",
]

_EMPATHY_BANK_FR = [
    "Merci de poser la question — c’est tout à fait normal d’avoir des doutes ou des questions.",
    "Je vous comprends. C’est légitime de chercher une réponse claire et rassurante.",
    "Vous n’êtes pas seul(e) à vous poser ce type de question — elle est tout à fait valable.",
    "C’est compréhensible de vouloir une réponse précise, surtout avec tout ce que vous traversez.",
    "C’est une question très raisonnable.",
    "Je suis là avec vous — c’est normal de chercher à être rassuré(e).",
    "C’est légitime de vouloir se sentir en confiance pour la suite.",
    "Merci d’en parler — c’est important de se sentir soutenu(e) et bien informé(e).",
]


def _stable_choice(items: List[str], key: str) -> str:
    if not items:
        return ""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(items)
    return items[idx]


def _fallback_empathy(language: str, user_query: str, top_question: str) -> str:
    lang = _norm_lang(language)
    key = f"{lang}::{user_query.strip().lower()}::{top_question.strip().lower()}"
    bank = _EMPATHY_BANK_FR if lang == "fr" else _EMPATHY_BANK_EN
    s = _stable_choice(bank, key)
    return (s.strip() + "\n\n") if s else ""


# ---------------------------
# LLM: write ONLY tone wrapper (NO medical facts)
# ---------------------------

class LLMWrapperWriter:
    """
    Generates ONLY an empathetic wrapper:
      - 1–2 sentences of empathy/validation
      - optional 1 sentence suggesting discussing with the medical team
    It must NOT include any medical facts or practical details.
    The factual FAQ content is appended verbatim outside the LLM.
    """
    def __init__(
        self,
        language: str,
        model: str = "llama3.2",
        temperature: float = 0.6,
        max_tokens: int = 120,
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
                "Tu es un assistant très bienveillant et rassurant.\n\n"
                "TÂCHE: Écris UNIQUEMENT un court préambule empathique (1–2 phrases) et, si utile, "
                "UNE phrase invitant à en parler avec l'équipe soignante.\n\n"
                "RÈGLES STRICTES:\n"
                "- N'écris AUCUN fait médical, aucun détail pratique.\n"
                "- N'invente rien sur la situation de la personne (pas de durée, pas d'hypothèses).\n"
                "- N'inclus pas de citations, ni de sections.\n"
                "- 2–3 phrases MAX.\n"
            )
        return (
            "You are a calm, deeply caring assistant.\n\n"
            "TASK: Write ONLY a short empathetic preface (1–2 sentences) and, if helpful, "
            "ONE sentence encouraging the person to discuss with their medical/care team.\n\n"
            "STRICT RULES:\n"
            "- Provide NO medical facts and no practical details.\n"
            "- Do not assume anything about the person’s situation (no durations, no 'you have been taking...').\n"
            "- No citations/sections.\n"
            "- 2–3 sentences MAX.\n"
        )

    def _user_prompt(self, user_query: str) -> str:
        if self.language == "fr":
            return f"Question utilisateur: {user_query}\n\nPréambule empathique:"
        return f"User question: {user_query}\n\nEmpathetic preface:"

    def write(self, user_query: str) -> str:
        import urllib.request

        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.model,
            "prompt": self._user_prompt(user_query),
            "system": self._system_prompt(),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if os.getenv("HORMONAI_LLM_DEBUG", "0") == "1":
            print(f"[LLM] Calling Ollama host={base} model={self.model}")

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            txt = (out.get("response") or "").strip()
            if os.getenv("HORMONAI_LLM_DEBUG", "0") == "1":
                print(f"[LLM] Got {len(txt)} chars back")
            return txt
        except Exception as e:
            if os.getenv("HORMONAI_LLM_DEBUG", "0") == "1":
                print(f"[LLM] Ollama call failed: {type(e).__name__}: {e}")
            return ""


# ---------------------------
# Answer logic + grounding
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


def _is_grounded(c: RetrievalCandidate, anchors: List[str], language: str) -> bool:
    if _contains_any_anchor(c.question, anchors, language):
        return True
    if _contains_any_anchor(f"{c.question} {c.section}", anchors, language):
        return True
    if _contains_any_anchor(f"{c.question} {c.section} {c.answer}", anchors, language):
        return True
    return False


def _find_best_grounded_candidate(
    candidates: List[RetrievalCandidate], anchors: List[str], language: str
) -> Optional[RetrievalCandidate]:
    ranked = sorted(candidates, key=_score_candidate, reverse=True)
    for c in ranked:
        if _is_grounded(c, anchors, language):
            return c
    return None


def _select_close_bundle(
    candidates: List[RetrievalCandidate],
    anchors: List[str],
    language: str,
    max_n: int = 3,
    close_delta_rerank: float = 0.15,
    close_delta_fused: float = 0.003,
) -> List[RetrievalCandidate]:
    if not candidates:
        return []

    ranked = sorted(candidates, key=_score_candidate, reverse=True)

    best = None
    for c in ranked:
        if _is_grounded(c, anchors, language):
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
        if not _is_grounded(c, anchors, language):
            continue
        if (best_score - _score_candidate(c)) <= close_delta:
            bundle.append(c)
            used_idx.add(c.index)
        if len(bundle) >= max_n:
            break

    return bundle


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

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["core_kws"] = core_kws
        dbg["anchors"] = anchors
        dbg["anchor_stems"] = sorted(list(_stem_set(anchors, lang)))
        dbg["use_llm"] = use_llm
        dbg["llm_model"] = llm_model
        dbg["top_candidates"] = [
            {"idx": c.index, "fused": round(c.fused_score, 5),
             "rerank": (round(c.rerank_score, 3) if c.rerank_score is not None else None),
             "q": c.question[:120]}
            for c in candidates[:10]
        ]

    if len(core_kws) < 1 or len(anchors) < 1 or not candidates:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    best = _find_best_grounded_candidate(candidates, anchors, lang)
    if best is None:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    bundle = _select_close_bundle(candidates, anchors, lang)
    if not bundle:
        bundle = [best]

    # Factual content is ALWAYS verbatim from FAQ bundle.
    faq_body = _format_bundle_body(lang, bundle)
    sources = _format_sources(lang, bundle)
    factual_block = _format_full_answer(lang, faq_body, sources)

    # Empathy wrapper: try LLM; fallback to generic bank if empty/fails.
    prefix = ""
    if use_llm:
        wrapper = LLMWrapperWriter(language=lang, model=llm_model).write(user_query=user_query).strip()
        if wrapper:
            prefix = wrapper + "\n\n"
        else:
            prefix = _fallback_empathy(lang, user_query, bundle[0].question)
            if debug:
                dbg["llm_wrapper_fallback"] = True

    answer_text = (prefix + factual_block).strip()

    top = bundle[0]
    return AnswerResult(
        answered=True,
        answer_text=answer_text,
        source_title=top.question,
        source_section=top.section,
        source_index=top.index,
        debug=(dbg if debug else None),
    )


def print_debug(result: AnswerResult) -> None:
    if not result.debug:
        return
    print("\n[DEBUG] core_kws:", result.debug.get("core_kws"))
    print("[DEBUG] anchors:", result.debug.get("anchors"))
    print("[DEBUG] anchor_stems:", result.debug.get("anchor_stems"))
    print("[DEBUG] use_llm:", result.debug.get("use_llm"), "llm_model:", result.debug.get("llm_model"))
    print("[DEBUG] Retrieved candidates:")
    for row in result.debug.get("top_candidates", []):
        print(f"  idx={row['idx']} fused={row['fused']} rerank={row['rerank']} | Q={row['q']}")
