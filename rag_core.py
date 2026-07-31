from __future__ import annotations

import os
import re
import json
import math
import time
import pickle
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

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
    return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", (text or "").lower())


def _has_number_or_percent(text: str) -> bool:
    t = text or ""
    return bool(re.search(r"(\d+(\.\d+)?)\s*%|\b\d+(\.\d+)?\b", t))


_INTERROGATIVE_PREFIX = re.compile(
    r"^(can|do|does|is|are|should|shall|will|would|could|may|might|what|when|why|how|where|which|who|"
    r"puis|dois|doit|est-ce|peut|peut-on|peut-il|faut|faut-il|comment|pourquoi|quand|quoi|quel|quelle|où|y-a-t-il)\b",
    flags=re.IGNORECASE,
)


def _looks_like_faq_question(question: str) -> bool:
    """
    Guard against parse artifacts where an ANSWER fragment was stored as a
    question (e.g. "Do not double up to make up for a missed dose.",
    "When osteoporosis exists before or develops during treatment, ...").

    A well-formed FAQ question either ends with '?' or is a short
    interrogative clause. Long declarative sentences that merely start with an
    interrogative word ("When osteoporosis exists...") are rejected. This is a
    runtime safety net for the prebuilt indexes; ingest_faq.py has been fixed
    so freshly ingested corpora will not contain these artifacts.
    """
    q = (question or "").strip()
    if not q:
        return False
    if q.endswith("?"):
        return True
    # Reject negative imperatives / instruction lines ("Do not double up ...",
    # "Ne pas ...") that start with an interrogative-looking word but are answers.
    if re.match(r"^(do not|don't|never|ne\s+pas|n[’']|il ne faut pas)\b", q, flags=re.IGNORECASE):
        return False
    # No question mark: only accept a short interrogative clause.
    if _INTERROGATIVE_PREFIX.match(q) and len(_tokenize(q)) <= 12:
        return True
    return False


# NOTE: the old hand-maintained EN/FR stopword, generic, emotion, and
# drug-term lists (and extract_core_keywords / anchor_keywords / _emotion_stems)
# were removed. Anchor extraction is now IDF-weighted (extract_anchor_concepts +
# a small domain-independent _MINIMAL_FUNCTION_WORDS set); see that function.


# ---------------------------
# Stats intent vs stats grounding (FIX)
# ---------------------------

# Broad: detect that user is asking for stats
STATS_INTENT_HINTS_EN = {
    "percent","percentage","proportion","rate","prevalence","incidence",
    "how","many","often",
    "odds","chance","probability","likelihood","frequency",
}

STATS_INTENT_HINTS_FR = {
    "pourcentage","percent","proportion","taux","prévalence","prevalence","incidence",
    "combien","souvent",
    "probabilité","probabilite","fréquence","frequence",
}

# Strict: require candidate to actually be “stats-like”
# IMPORTANT: NO "how", "many", "often" here — those cause false positives via section titles.
STATS_GROUND_HINTS_EN = {
    "percent","percentage","proportion","rate","prevalence","incidence",
    "odds","chance","probability","likelihood","frequency",
}

STATS_GROUND_HINTS_FR = {
    "pourcentage","percent","proportion","taux","prévalence","prevalence","incidence",
    "probabilité","probabilite","fréquence","frequence",
}

# For stats queries, ignore broad concepts that would match almost anything.
STATS_BROAD_CONCEPTS_EN = {"cancer", "patient", "patients", "people", "person", "persons", "breast"}
STATS_BROAD_CONCEPTS_FR = {"cancer", "patient", "patients", "personne", "personnes", "sein"}


# Words that, on their own, signal the user wants a statistic.
# NOTE: bare "how"/"many"/"often" are deliberately EXCLUDED -- they caused
# duration questions like "How long do I need to stay on therapy?" to be
# misrouted into the strict stats gate and hard-abstained. Statistical INTENT
# is now detected from explicit statistical words or specific bigrams.
STATS_WORD_EN = {
    "percent", "percentage", "proportion", "rate", "rates",
    "prevalence", "incidence", "odds", "probability", "likelihood", "frequency",
}
STATS_WORD_FR = {
    "pourcentage", "percent", "proportion", "taux", "prévalence", "prevalence",
    "incidence", "probabilité", "probabilite", "fréquence", "frequence",
}

_STATS_BIGRAMS_EN = {
    ("how", "many"), ("how", "often"), ("how", "likely"),
    ("what", "percentage"), ("what", "percent"), ("what", "proportion"),
    ("what", "fraction"), ("what", "share"),
}


def _is_stats_intent(user_query: str, language: str) -> bool:
    lang = _norm_lang(language)
    text = user_query or ""
    if "%" in text:
        return True

    toks = _tokenize(text)          # ordered
    tokset = set(toks)
    words = STATS_WORD_FR if lang == "fr" else STATS_WORD_EN
    if tokset & words:
        return True

    bigrams = set(zip(toks, toks[1:]))
    if lang == "fr":
        # "combien de <X>" is statistical, EXCEPT "combien de temps" (= how long).
        if ("combien", "de") in bigrams and "temps" not in tokset:
            return True
        if ("quelle", "fréquence") in bigrams or ("quelle", "frequence") in bigrams:
            return True
        if ("quel", "pourcentage") in bigrams:
            return True
        return False

    return len(bigrams & _STATS_BIGRAMS_EN) > 0


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
# Lay ↔ clinical synonym sets (MATCHING ONLY)
# ---------------------------

# Lay <-> clinical synonym groups. Any member of a group is considered to
# cover any other member during the coverage gate. This is a stopgap that
# encodes the most common lay/clinical mismatches; the durable fix is a
# domain-adapted embedding model or an LLM query-rewrite step (see
# RETRIEVAL_REVIEW.md, section 5.3).
_SYNONYM_GROUPS_EN: List[List[str]] = [
    ["heart", "cardiovascular", "cardio", "cardiac"],
    ["bone", "bones", "osteoporosis", "osteoporotic", "skeletal"],
    # "break" was the concept that broke the example query: the corpus phrases
    # the same idea as "pause", "stop", "discontinue", etc.
    ["break", "pause", "stop", "stopping", "interrupt", "interruption",
     "discontinue", "discontinuation", "hold", "holiday", "suspend"],
    ["pregnant", "pregnancy", "conceive", "conception", "fertility", "baby", "child"],
    ["tired", "tiredness", "fatigue", "exhausted", "exhaustion", "energy"],
    ["period", "periods", "menstrual", "menstruation", "cycle", "cycles", "bleeding"],
    ["bloodwork", "blood", "bloodtest", "labs"],
    ["mood", "depression", "depressed", "sad"],
    ["sleep", "sleeping", "insomnia"],
    ["libido", "sex", "sexual", "desire"],
    ["diet", "food", "foods", "eat", "eating", "nutrition"],
    ["exercise", "activity", "sport", "sports", "physical"],
    ["recurrence", "relapse", "return", "coming-back", "comeback"],
]

_SYNONYM_GROUPS_FR: List[List[str]] = [
    ["coeur", "cardiovasculaire", "cardio", "cardiaque"],
    ["os", "osseux", "ostéoporose", "osteoporose", "squelette"],
    ["pause", "arrêt", "arret", "arrêter", "arreter", "stopper", "interrompre",
     "interruption", "suspendre", "interrompue"],
    ["enceinte", "grossesse", "concevoir", "conception", "fertilité", "fertilite",
     "bébé", "bebe", "enfant"],
    ["fatigue", "fatigué", "fatigue", "épuisé", "epuise", "épuisement", "énergie", "energie"],
    ["règles", "regles", "menstruel", "menstruation", "cycle", "cycles", "saignement"],
    ["sang", "prise-de-sang", "analyses"],
    ["humeur", "dépression", "depression", "déprimé", "deprime", "triste"],
    ["sommeil", "dormir", "insomnie"],
    ["libido", "sexe", "sexuel", "sexuelle", "désir", "desir"],
    ["régime", "regime", "alimentation", "aliment", "aliments", "manger", "nutrition"],
    ["exercice", "activité", "activite", "sport", "physique"],
    ["récidive", "recidive", "rechute", "retour"],
]


def _build_synonym_map(groups: List[List[str]], lang: str) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for group in groups:
        stems = {_stem(w, lang) for w in group}
        for s in stems:
            m.setdefault(s, set()).update(stems)
    return m


_SYNONYM_MAP_CACHE: Dict[str, Dict[str, Set[str]]] = {}


def _synonym_map(lang: str) -> Dict[str, Set[str]]:
    lang = _norm_lang(lang)
    if lang not in _SYNONYM_MAP_CACHE:
        groups = _SYNONYM_GROUPS_FR if lang == "fr" else _SYNONYM_GROUPS_EN
        _SYNONYM_MAP_CACHE[lang] = _build_synonym_map(groups, lang)
    return _SYNONYM_MAP_CACHE[lang]


def _concept_match_stems(concept_stem: str, language: str) -> Set[str]:
    lang = _norm_lang(language)
    m = _synonym_map(lang)
    return m.get(concept_stem, {concept_stem})


def _anchor_overlap_concepts(text: str, anchor_concepts: Set[str], language: str) -> Set[str]:
    lang = _norm_lang(language)
    text_stems = _stem_set(_tokenize(text or ""), lang)

    covered: Set[str] = set()
    for concept in anchor_concepts:
        acceptable = _concept_match_stems(concept, lang)
        if len(text_stems.intersection(acceptable)) > 0:
            covered.add(concept)
    return covered


# ---------------------------
# IDF-weighted anchor extraction (replaces the stopword/generic/emotion lists)
# ---------------------------

# Minimal, DOMAIN-INDEPENDENT grammatical stop set. This is NOT the old
# maintained keyword machinery -- it only removes universal function words that
# have no retrieval value in any corpus. Everything domain-specific (which
# medical terms are "generic", which words are emotional, which are drugs) is
# now handled automatically by corpus IDF, so those lists are no longer used
# for extraction.
_MINIMAL_FUNCTION_WORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "can", "could", "should", "would",
    "will", "may", "might", "must", "i", "you", "we", "they", "he", "she", "it",
    "my", "your", "our", "their", "this", "that", "these", "those", "what",
    "why", "how", "when", "where", "which", "who", "not", "no", "yes",
    "about", "into", "onto", "over", "under", "than", "then", "because",
    "while", "during", "before", "after", "between", "through", "without",
    "within", "again", "also", "too", "very", "just", "really", "ever",
    "never", "always", "still", "instead", "sometimes", "maybe", "perhaps",
    # discourse markers / interjections (closed-class, no retrieval value)
    "okay", "ok", "please", "thanks", "thank", "hello", "hi", "hey", "yeah", "sure",
    # French
    "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "dans", "sur",
    "pour", "avec", "par", "au", "aux", "en", "est", "sont", "ai", "as", "avez",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce", "cet",
    "cette", "ces", "quoi", "pourquoi", "comment", "quand", "quel", "quelle",
    "pas", "ne", "que", "qui", "sans", "avant", "apres", "après", "encore",
    "aussi", "vraiment", "juste", "jamais", "toujours", "parfois", "peut-etre",
    "peut-être", "bonjour", "salut", "merci", "oui", "non",
}


@dataclass
class AnchorExtraction:
    anchors: List[str]                 # discriminative stems used for coverage
    present: List[Tuple[str, float, int]]   # (stem, idf, df) for all in-corpus tokens
    dropped_absent: List[str]          # tokens not in the corpus (df == 0)
    dropped_common: List[str]          # in-corpus but too frequent to be useful


def extract_anchor_concepts(
    user_query: str,
    language: str,
    retriever: "HybridFAQRetriever",
    max_df_fraction: float = 0.5,
) -> AnchorExtraction:
    """
    Select anchor concepts by corpus IDF instead of hand-maintained lists.

    A stem is an anchor iff it is:
      - not a universal grammatical function word,
      - present in the corpus (df >= 1), and
      - discriminative, i.e. df <= max_df_fraction * corpus_size
        (words appearing in more than ~half the corpus carry no signal).

    Absent tokens (df == 0) are NOT anchors -- lexical matching cannot use them,
    and they are handled by the semantic path. This automatically drops filler
    ("ever", "really"), domain-generic terms ("hormone", "treatment"), emotion
    words, and drug names that saturate the corpus, with zero curation.
    """
    lang = _norm_lang(language)
    cap = max(1, int(retriever.corpus_size * max_df_fraction))

    seen: Set[str] = set()
    present: List[Tuple[str, float, int]] = []
    dropped_absent: List[str] = []
    anchors: List[str] = []
    dropped_common: List[str] = []

    for tok in _tokenize(user_query):
        if len(tok) <= 2 or tok in _MINIMAL_FUNCTION_WORDS:
            continue
        s = _stem(tok, lang)
        if s in seen:
            continue
        seen.add(s)

        d = retriever.df(s)
        if d <= 0:
            dropped_absent.append(tok)
            continue

        present.append((s, retriever.idf(s), d))
        if d <= cap:
            anchors.append(s)
        else:
            dropped_common.append(tok)

    # If everything present was too common (e.g. a very short, generic query),
    # keep the single most specific (highest-IDF) present stem so lexical
    # coverage still has something to work with.
    if not anchors and present:
        best = max(present, key=lambda x: x[1])
        anchors.append(best[0])

    return AnchorExtraction(
        anchors=anchors,
        present=present,
        dropped_absent=dropped_absent,
        dropped_common=dropped_common,
    )


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
    # Best cosine similarity to the query across the dense (FAISS) indexes.
    # Used by the semantic-accept path so the answer/abstain decision is not
    # purely lexical. None when no dense index returned this candidate.
    dense_sim: Optional[float] = None
    # Language of the source item ("en"/"fr"). Used for same-language-first
    # selection with cross-lingual fallback in shared mode.
    lang: str = ""
    # Provenance, so display/citation can branch by type. For an article chunk
    # `question` is a heading breadcrumb (not a real question), so the formatters
    # must NOT present it as one.
    source_type: str = "faq"
    source_id: str = ""
    heading_path: str = ""


@dataclass
class AnswerResult:
    answered: bool
    answer_text: str
    source_title: Optional[str] = None
    source_section: Optional[str] = None
    source_index: Optional[int] = None
    source_indices: Optional[List[int]] = None   # positional indices of every bundled source
    sources_text: Optional[str] = None            # formatted citations, kept OUT of answer_text
    timing_ms: Optional[Dict[str, Any]] = None    # per-query latency breakdown (always populated)
    # LLM rephrasing outcome, so callers can surface a silent fallback:
    #   None                  -> LLM rephrasing not requested (use_llm=False)
    #   "used"                -> LLM rephrased the answer
    #   "fallback_unreachable"-> LLM requested but unreachable; showed verbatim
    #   "fallback_empty"      -> LLM reachable but returned nothing; showed verbatim
    llm_status: Optional[str] = None
    llm_error: Optional[str] = None               # connection/HTTP error detail, when unreachable
    debug: Optional[Dict[str, Any]] = None


# ---------------------------
# Hybrid Retriever
# ---------------------------

class HybridFAQRetriever:
    def __init__(
        self,
        language: str = "en",
        data_dir: str = "data",
        top_k: int = 40,   # candidate pool per channel; sized for the enlarged, bilingual corpus
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        rerank: bool = False,
        rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        rerank_top_n: int = 20,   # only rerank the top-N fused candidates (latency vs. recall)
        shared: bool = False,
        corpus_prefix: Optional[str] = None,
    ):
        # `language` is the ACTIVE QUERY language (drives stemming, IDF anchors,
        # query prefix). In shared mode the corpus contains all languages and
        # this attribute may be reassigned per query.
        self.language = _norm_lang(language)
        self.data_dir = data_dir
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        # Shared multilingual mode: load one combined index ('<prefix>_all_*')
        # spanning all languages, prefer same-language candidates, fall back
        # cross-lingual. `corpus_prefix` overrides the loaded basename.
        self.shared = shared
        self.corpus_prefix = corpus_prefix

        self.rerank = rerank
        self.rerank_model = rerank_model
        self.rerank_top_n = max(2, int(rerank_top_n))

        self._items: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._index_q: Optional[faiss.Index] = None
        self._index_qa: Optional[faiss.Index] = None
        self._index_qp: Optional[faiss.Index] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._cross_encoder = None

        self._stored_embedding_model_name: Optional[str] = None
        self._query_prefix: str = ""

        # Per-language corpus statistics for IDF-weighted anchor extraction
        # (replaces the hand-maintained stopword / generic / emotion lists).
        # Keyed by language: {lang: {df, idf, avg, stems, size}}. Accessors use
        # the ACTIVE query language so the same combined corpus serves both.
        self._stats_by_lang: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        if self.corpus_prefix:
            prefix = self.corpus_prefix
        elif self.shared:
            prefix = "kb_all"
        else:
            prefix = f"kb_{self.language}"
        qa_path = os.path.join(self.data_dir, f"{prefix}_qa.pkl")
        bm25_path = os.path.join(self.data_dir, f"{prefix}_bm25.pkl")
        idx_q_path = os.path.join(self.data_dir, f"{prefix}_index_q.faiss")
        idx_qa_path = os.path.join(self.data_dir, f"{prefix}_index_qa.faiss")
        idx_qp_path = os.path.join(self.data_dir, f"{prefix}_index_qp.faiss")

        with open(qa_path, "rb") as f:
            payload = pickle.load(f)
        self._items = payload["items"]
        self._stored_embedding_model_name = payload.get("embedding_model_name")
        # Model-agnostic query prefix (e.g. "query: " for e5). Empty for mpnet /
        # BGE-M3. Ingestion stores the matching passage prefix on the doc side.
        self._query_prefix = payload.get("query_prefix", "") or ""

        # Precompute per-LANGUAGE stem document frequency and smoothed IDF.
        # Each item is stemmed in its OWN language, so a shared multilingual
        # corpus still yields clean, language-correct anchor statistics. Common
        # words (high df, e.g. "hormone") get low weight automatically -- no
        # stopword/generic list needed. Accessors below key on the active query
        # language.
        df_by_lang: Dict[str, "Counter[str]"] = {}
        size_by_lang: Dict[str, int] = {}
        for it in self._items:
            lg = _norm_lang(it.get("lang", self.language))
            blob = f"{it.get('question','')} {it.get('section','')} {it.get('answer','')}"
            for para in (it.get("q_paraphrases") or []):
                blob += " " + str(para)
            d = df_by_lang.setdefault(lg, Counter())
            for s in _stem_set(_tokenize(blob), lg):  # each stem once per doc
                d[s] += 1
            size_by_lang[lg] = size_by_lang.get(lg, 0) + 1

        self._stats_by_lang = {}
        for lg, d in df_by_lang.items():
            size = max(1, size_by_lang.get(lg, 1))
            idf = {s: math.log((size + 1) / (c + 1)) + 1.0 for s, c in d.items()}
            avg = (sum(idf.values()) / len(idf)) if idf else 0.0
            self._stats_by_lang[lg] = {
                "df": dict(d), "idf": idf, "avg": avg,
                "stems": set(d.keys()), "size": size,
            }

        with open(bm25_path, "rb") as f:
            payload_bm25 = pickle.load(f)
        self._bm25 = payload_bm25["bm25"]

        self._index_q = faiss.read_index(idx_q_path)
        self._index_qa = faiss.read_index(idx_qa_path)

        if os.path.exists(idx_qp_path):
            try:
                self._index_qp = faiss.read_index(idx_qp_path)
            except Exception:
                self._index_qp = None
        else:
            self._index_qp = None

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

        if self._index_qp is not None:
            qp_dim = int(getattr(self._index_qp, "d", -1))
            if qp_dim > 0 and qp_dim != model_dim:
                self._index_qp = None

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

    def _active_stats(self) -> Dict[str, Any]:
        """Corpus statistics for the active query language."""
        return self._stats_by_lang.get(
            self.language, {"df": {}, "idf": {}, "avg": 0.0, "stems": set(), "size": 1}
        )

    @property
    def corpus_size(self) -> int:
        return self._active_stats()["size"]

    def stem_in_corpus(self, stem: str) -> bool:
        """True if the stem (or a synonym) appears in the active-language corpus."""
        stems = self._active_stats()["stems"]
        if not stems:
            return True  # stats unavailable: do not prune
        acceptable = _concept_match_stems(stem, self.language)
        return len(stems.intersection(acceptable)) > 0

    def df(self, stem: str) -> int:
        """Document frequency of a stem (max over its synonym group)."""
        d = self._active_stats()["df"]
        acceptable = _concept_match_stems(stem, self.language)
        return max((d.get(a, 0) for a in acceptable), default=0)

    def idf(self, stem: str) -> float:
        """Smoothed IDF of a stem (min over its synonym group; 0.0 if absent)."""
        idf_map = self._active_stats()["idf"]
        acceptable = _concept_match_stems(stem, self.language)
        vals = [idf_map[a] for a in acceptable if a in idf_map]
        return min(vals) if vals else 0.0

    @property
    def average_idf(self) -> float:
        return self._active_stats()["avg"]

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        # Per-query latency breakdown, readable via `retriever.last_timing_ms`
        # (populated on every call; surfaced in --debug output).
        self.last_timing_ms: Dict[str, Any] = {"retrieve": 0.0, "rerank": 0.0, "n_reranked": 0}
        if not self._items:
            return []

        assert self._bm25 is not None
        assert self._index_q is not None
        assert self._index_qa is not None

        _t_retrieve = time.perf_counter()
        bm25_scores = self._bm25.get_scores(_tokenize(user_query))
        bm25_ranked = np.argsort(-bm25_scores)[: self.top_k]

        # The three dense indexes are queried with the same encoded vector; only
        # the document side differs, so encode once (was encoded three times).
        q_emb = self._encode(f"{self._query_prefix}Question: {user_query}")
        Dq, Iq = self._index_q.search(q_emb, self.top_k)
        Dqa, Iqa = self._index_qa.search(q_emb, self.top_k)

        Iqp = None
        Dqp = None
        if self._index_qp is not None:
            try:
                Dqp, Iqp = self._index_qp.search(q_emb, self.top_k)
            except Exception:
                Iqp = None
                Dqp = None

        def rrf(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank)

        fused: Dict[int, float] = {}
        # Track the best cosine similarity seen for each item across dense indexes
        # (IndexFlatIP on L2-normalized vectors -> inner product == cosine).
        dense_sim: Dict[int, float] = {}

        def _note_dense(D, I):
            if D is None or I is None:
                return
            for sim, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx >= 0:
                    dense_sim[int(idx)] = max(dense_sim.get(int(idx), -1.0), float(sim))

        _note_dense(Dq, Iq)
        _note_dense(Dqa, Iqa)
        _note_dense(Dqp, Iqp)

        for r, idx in enumerate(bm25_ranked.tolist()):
            fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)
        for r, idx in enumerate(Iq[0].tolist()):
            if idx >= 0:
                fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)
        for r, idx in enumerate(Iqa[0].tolist()):
            if idx >= 0:
                fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)
        if Iqp is not None:
            for r, idx in enumerate(Iqp[0].tolist()):
                if idx >= 0:
                    fused[int(idx)] = fused.get(int(idx), 0.0) + rrf(r)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        candidates: List[RetrievalCandidate] = []
        for idx, score in ranked:
            it = self._items[int(idx)]
            question = str(it.get("question", ""))
            # Runtime artifact filter: drop answer-fragments mis-parsed as
            # questions in the prebuilt indexes (see _looks_like_faq_question).
            # Only applies to FAQ items -- article "questions" are heading
            # breadcrumbs and are not expected to be well-formed questions.
            source_type = str(it.get("source_type", "faq"))
            if source_type == "faq" and not _looks_like_faq_question(question):
                continue
            candidates.append(
                RetrievalCandidate(
                    index=int(idx),
                    question=question,
                    section=str(it.get("section", "")),
                    answer=str(it.get("answer", "")),
                    fused_score=float(score),
                    dense_sim=(dense_sim.get(int(idx)) if int(idx) in dense_sim else None),
                    lang=_norm_lang(str(it.get("lang", self.language))),
                    source_type=source_type,
                    source_id=str(it.get("source_id", "")),
                    heading_path=str(it.get("heading_path", "")),
                )
            )

        # Embedding + BM25 + FAISS search + fusion are done; the rest is reranking.
        self.last_timing_ms["retrieve"] = (time.perf_counter() - _t_retrieve) * 1000.0

        if self._cross_encoder is not None and len(candidates) >= 2:
            # Rerank only a SHORTLIST of the best fused candidates, not the whole
            # top_k pool. Cross-encoder cost is linear in the number of pairs, and
            # a large reranker (e.g. bge-reranker-v2-m3) over 40 pairs is the main
            # per-query latency. `candidates` is already sorted by fused score, so
            # the top `rerank_top_n` are the strongest hybrid hits; the reranked
            # shortlist is returned and the weaker tail is dropped.
            shortlist = candidates[: self.rerank_top_n]
            pairs = [(user_query, f"{c.section}\n{c.question}\n{c.answer}") for c in shortlist]
            _t_rerank = time.perf_counter()
            try:
                ce_scores = self._cross_encoder.predict(pairs)
                for c, s in zip(shortlist, ce_scores):
                    c.rerank_score = float(s)
                shortlist.sort(
                    key=lambda c: c.rerank_score if c.rerank_score is not None else -1e9,
                    reverse=True,
                )
                candidates = shortlist
                self.last_timing_ms["n_reranked"] = len(pairs)
            except Exception:
                pass
            self.last_timing_ms["rerank"] = (time.perf_counter() - _t_rerank) * 1000.0

        return candidates


# ---------------------------
# Formatting functions
# ---------------------------

def _candidate_title(c: RetrievalCandidate) -> str:
    """A display heading for a bundle entry. FAQ -> the question; article -> its
    source document / section heading (never the fake 'question' breadcrumb)."""
    if c.source_type == "faq":
        return (c.question or "").strip()
    hp = (c.heading_path or "").strip()
    src = (c.source_id or "").strip()
    if hp and src and hp != src:
        return f"{src} — {hp}"
    return hp or src or (c.section or "").strip()


def _format_bundle_body(language: str, bundle: List[RetrievalCandidate]) -> str:
    if not bundle:
        return ""
    parts: List[str] = []
    for c in bundle:
        parts.append(f"**{_candidate_title(c)}**\n{c.answer.strip()}")
    return "\n\n".join(parts).strip()


def _format_preface(language: str) -> str:
    lang = _norm_lang(language)
    if lang == "fr":
        return "**Voici les informations disponibles sur ce sujet :**\n\n"
    return "**Here is the available information on this topic:**\n\n"


def _format_sources(language: str, bundle: List[RetrievalCandidate]) -> str:
    lang = _norm_lang(language)
    lines: List[str] = []
    label = "Source :" if lang == "fr" else "Source:"
    sec_label = "section :" if lang == "fr" else "section:"
    for c in bundle:
        if c.source_type == "faq":
            lines.append(f"**— {label}** “{c.question}” ({sec_label} {c.section})")
        else:
            # Article: cite the source document (+ section heading when it adds
            # information), not the "question" breadcrumb.
            src = (c.source_id or c.section or "").strip()
            hp = (c.heading_path or "").strip()
            cite = f"{src} — {hp}" if (hp and src and hp != src) else src
            lines.append(f"**— {label}** {cite}")
    return "\n\n".join(lines).strip()


def _format_full_answer(language: str, body: str, sources: str) -> str:
    pre = _format_preface(language)
    if sources:
        return (pre + (body or "").strip() + "\n\n" + sources).strip()
    return (pre + (body or "").strip()).strip()


def _bundle_source_text(bundle: List[RetrievalCandidate]) -> str:
    """Verbatim source text handed to the LLM for grounded rephrasing. FAQ items
    keep their question as a mini-heading; article chunks (whose 'question' is a
    filename breadcrumb) contribute their body only."""
    parts: List[str] = []
    for c in bundle:
        q = (c.question or "").strip()
        a = (c.answer or "").strip()
        if q.endswith("?"):
            parts.append(f"{q}\n{a}")
        else:
            parts.append(a)
    return "\n\n".join(p for p in parts if p).strip()


# ---------------------------
# Compassionate survival reframing
# ---------------------------
#
# Patient-facing answers must not describe the person's own outcome with the
# words "death"/"dying" (or FR "mort"/"mourir"/"décès"). It is fine, and often
# more accurate, to speak of SURVIVAL instead. The source corpus itself uses
# mortality phrasing (e.g. NCI: "lower your risk of ... and of dying from breast
# cancer"), so a grounded rephrase or verbatim quote will echo it unless we
# reframe deterministically here -- a prompt alone is not reliable with a small
# local model.
#
# IMPORTANT: the patterns are NARROW on purpose. They match only patient-
# mortality constructions (".. dying/death FROM breast cancer/the cancer/it",
# ".. risk/chance of dying .."). They deliberately do NOT touch statements about
# cancer CELLS dying ("the cells may die", "la mort des cellules cancéreuses"),
# which are accurate, desirable, and not distressing.

_SURVIVAL_REFRAMES_EN: List[Tuple[str, str]] = [
    # "the risk of (you) dying [from ...]" -> threat-to-survival framing
    (r"\bthe risk of (?:you |your )?dying(?: from (?:breast cancer|the cancer|it))?\b",
     "the risk to your long-term survival"),
    # "risk/chance/likelihood of dying [from ...]"
    (r"\b(?:risk|chance|chances|likelihood) of dying(?: from (?:breast cancer|the cancer|it))?\b",
     "risk to long-term survival"),
    # "risk/chance of death from ..."
    (r"\b(?:risk|chance|chances|likelihood) of death(?: from (?:breast cancer|the cancer|it))?\b",
     "risk to long-term survival"),
    # list/conjunction context: "..., as well as / and (of) dying from ..."
    (r"\s*,?\s*(?:as well as|and)\s+(?:of\s+)?dying from (?:breast cancer|the cancer|it)\b",
     ", as well as improving long-term survival"),
    # any remaining "dying/death from <cancer/it>"
    (r"\bdying from (?:breast cancer|the cancer|it)\b", "affecting long-term survival"),
    (r"\bdeath from (?:breast cancer|the cancer|it)\b", "long-term survival"),
]

_SURVIVAL_REFRAMES_FR: List[Tuple[str, str]] = [
    # "réduire/diminuer (le risque de) mortalité du/par cancer du sein" -> survie
    (r"\b(réduire|réduit|diminuer|diminue|abaisser|abaisse|baisser)\s+(?:la\s+|le\s+)?"
     r"(?:risque de\s+)?mortalité\s+(?:du|par)\s+cancer du sein\b",
     r"améliorer la survie au cancer du sein"),
    (r"\bla mortalité\s+(?:du|par)\s+cancer du sein\b", "la survie au cancer du sein"),
    # "risque de décès / de mourir [du cancer du sein / de la maladie]"
    (r"\brisque de (?:décès|mourir)(?:\s+(?:du|par)\s+cancer du sein|\s+de la maladie)?\b",
     "risque pour la survie à long terme"),
    # "mourir du cancer du sein / de la maladie" and "en mourir"
    (r"\bmourir\s+(?:du cancer du sein|de la maladie)\b", "pour la survie à long terme"),
    (r"\ben mourir\b", "pour la survie à long terme"),
]


# Meta-preamble a chat model often prepends before the real answer, e.g.
# "Here's a clear and compassionate response to your question:" or
# "Voici une réponse claire et bienveillante à votre question :". These are
# unnatural in a patient chat, so strip a single leading preamble line/sentence.
_LLM_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|of course|certainly|absolutely|bien sûr|bien sûr|voici|here(?:'|’)?s|here is|"
    r"here(?:'|’)?s?\s+a|below is|i(?:'|’)?d be happy to)\b[^\n:]*?[:\.]\s+",
    re.IGNORECASE,
)


def _strip_llm_preamble(text: str) -> str:
    """Remove a single leading meta-preamble ("Here's a ... response:", "Sure, ...:",
    "Voici ... :") that a chat model may prepend. Only strips when a real answer
    follows, so a legitimate sentence is never removed."""
    if not text:
        return text
    stripped = _LLM_PREAMBLE_RE.sub("", text, count=1).lstrip()
    return stripped if stripped else text


def _reframe_survival(text: str, language: str) -> str:
    """Rewrite patient-mortality phrasing into survival language, in place.

    Deterministic and narrowly scoped so accurate cancer-cell-death statements
    are preserved. Applied to the final patient-facing answer (both the grounded
    LLM rephrase and the verbatim path), never to the cited source snippets in
    the Sources dropdown (those stay verbatim for verifiability)."""
    if not text:
        return text
    is_fr = _norm_lang(language) == "fr"
    rules = _SURVIVAL_REFRAMES_FR if is_fr else _SURVIVAL_REFRAMES_EN
    out = text
    for pattern, repl in rules:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    if is_fr:
        # Restore French elision broken by a substitution starting with a vowel
        # (e.g. "de améliorer" -> "d'améliorer", "que améliorer" -> "qu'améliorer").
        out = re.sub(r"\b([dlncjmts])e améliorer\b", r"\1'améliorer", out, flags=re.IGNORECASE)
        out = re.sub(r"\bque améliorer\b", "qu'améliorer", out, flags=re.IGNORECASE)
    # Tidy artifacts a substitution can leave behind (", ," or doubled spaces).
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


# ---------------------------
# Empathy + LLM wrapper (unchanged behavior)
# ---------------------------

_EMPATHY_BANK_EN = [
    "Thank you for sharing that — it’s completely understandable to want reassurance.",
    "I hear you. It’s very reasonable to be thinking about this.",
    "It makes sense to want clear, reliable guidance about your health.",
    "You’re not alone in asking this — it’s a valid concern.",
    "It’s understandable to want straightforward information, especially with everything you’re managing.",
]

_EMPATHY_BANK_FR = [
    "Merci de le partager — c’est tout à fait normal de vouloir être rassuré(e).",
    "Je vous comprends. C’est une inquiétude très légitime.",
    "C’est normal de vouloir une information claire et fiable sur votre santé.",
    "Vous n’êtes pas seul(e) à vous poser la question — c’est une préoccupation valable.",
    "C’est compréhensible de vouloir une réponse simple et fiable, surtout avec tout ce que vous traversez.",
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


class LLMWrapperWriter:
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
        # Set by _generate on each call: the connection/HTTP error string if the
        # local LLM (Ollama) could not be reached, else None. Lets callers report
        # WHY a rephrase fell back to verbatim instead of failing silently.
        self.last_error: Optional[str] = None

    def _system_prompt(self) -> str:
        if self.language == "fr":
            return (
                "Tu es un assistant très bienveillant et rassurant.\n\n"
                "TÂCHE: Écris UNIQUEMENT un court préambule empathique (1–2 phrases) et, si utile, "
                "UNE phrase invitant à en parler avec l'équipe soignante.\n\n"
                "RÈGLES STRICTES:\n"
                "- N'écris AUCUN fait médical, aucun détail pratique.\n"
                "- N'invente rien sur la situation de la personne.\n"
                "- Ne donne aucune statistique, aucun ordre de grandeur, aucune explication technique.\n"
                "- Emploie un langage doux et non alarmant ; n'évoque pas la mort, le décès ni le pronostic.\n"
                "- N'inclus pas de citations, ni de sections.\n"
                "- 2–3 phrases MAX.\n"
            )
        return (
            "You are a calm, deeply caring assistant.\n\n"
            "TASK: Write ONLY a short empathetic preface (1–2 sentences) and, if helpful, "
            "ONE sentence encouraging the person to discuss with their medical/care team.\n\n"
            "STRICT RULES:\n"
            "- Provide NO medical facts and no practical details.\n"
            "- Do not assume anything about the person’s situation.\n"
            "- Do not provide statistics, magnitudes, or technical explanations.\n"
            "- Use gentle, non-alarming language; do not mention death, dying, or prognosis.\n"
            "- No citations/sections.\n"
            "- 2–3 sentences MAX.\n"
        )

    def _user_prompt(self, user_query: str) -> str:
        if self.language == "fr":
            return f"Question utilisateur: {user_query}\n\nPréambule empathique:"
        return f"User question: {user_query}\n\nEmpathetic preface:"

    # ---- grounded rephrasing (strictly uses the provided source text) ----

    def _rephrase_system_prompt(self) -> str:
        if self.language == "fr":
            return (
                "Tu es un assistant bienveillant pour des patientes sous hormonothérapie "
                "adjuvante du cancer du sein.\n\n"
                "TÂCHE: Réponds à la question en reformulant le TEXTE SOURCE ci-dessous en "
                "un message clair, chaleureux et simple.\n\n"
                "RÈGLES STRICTES:\n"
                "- Utilise UNIQUEMENT les informations présentes dans le TEXTE SOURCE. "
                "N'ajoute AUCUN fait, chiffre, médicament, posologie ou conseil absent du texte.\n"
                "- Réponds DIRECTEMENT et de façon générale à la question posée. Ne centre pas la "
                "réponse sur un scénario particulier (par exemple la grossesse) sauf si la personne "
                "l'a explicitement demandé ; si une source ne traite que d'un tel scénario, mentionne-le "
                "brièvement au maximum.\n"
                "- N'omets aucune mise en garde ou condition de sécurité présente dans le texte.\n"
                "- TON: emploie un langage doux, bienveillant et non alarmant.\n"
                "- FORMULATION EN TERMES DE SURVIE (OBLIGATOIRE): n'emploie JAMAIS les mots « mort », "
                "« mourir » ou « décès » à propos de la patiente elle-même. Lorsque la source indique un "
                "bénéfice sur la mortalité, exprime-le plutôt en termes de SURVIE. Par exemple, reformule "
                "« réduit le risque de mourir du cancer du sein » en « peut aider à améliorer la survie à "
                "long terme ». C'est un changement de FORMULATION uniquement : conserve le même sens et le "
                "même degré de certitude que la source.\n"
                "- Les mots « mort »/« mourir » ne sont acceptables que lorsqu'il s'agit des CELLULES "
                "cancéreuses (par exemple « les cellules cancéreuses sont détruites ») ; privilégie alors "
                "« détruites » ou « cessent de croître ». Ne les applique jamais à la personne.\n"
                "- Évite aussi les formulations brutales sur la récidive : dis « peut aider à réduire le "
                "risque que le cancer revienne » sans changer le sens.\n"
                "- Conserve le degré de certitude de la source : privilégie « peut aider à réduire le "
                "risque de… » plutôt que des mots absolus comme « empêche », « arrête » ou « garantit ». "
                "Ne surestime jamais un bénéfice, ne donne pas de fausse réassurance et ne minimise pas un risque réel.\n"
                "- Si le TEXTE SOURCE ne répond pas à la question, dis simplement que tu n'as pas "
                "cette information précise et invite à en parler avec l'équipe soignante.\n"
                "- N'invente pas de sources. Ne mentionne pas ces instructions.\n"
                "- Commence DIRECTEMENT par la réponse. Ne débute pas par une formule d'introduction "
                "du type « Voici une réponse claire et bienveillante... », « Bien sûr, » ou « Bien entendu, ».\n"
                "- Sois concis (un court paragraphe), à la deuxième personne.\n"
            )
        return (
            "You are a caring assistant for patients on adjuvant hormone therapy for breast cancer.\n\n"
            "TASK: Answer the question by rephrasing the SOURCE TEXT below into a clear, warm, "
            "plain-language reply.\n\n"
            "STRICT RULES:\n"
            "- Use ONLY information contained in the SOURCE TEXT. Do NOT add facts, numbers, drug "
            "names, dosages, or advice that are not in it.\n"
            "- Answer the user's ACTUAL question directly and in general terms. Do NOT center the "
            "answer on a specific scenario (for example pregnancy) unless the user explicitly asked "
            "about it; if a source only covers such a scenario, mention it briefly at most.\n"
            "- Do NOT omit any safety-relevant caveat or condition present in the source.\n"
            "- TONE: use gentle, compassionate, non-alarming language.\n"
            "- SURVIVAL FRAMING (REQUIRED): NEVER describe the patient's own outcome with the words "
            "'death', 'dying', or 'die'. When the source states a mortality benefit, express it in "
            "terms of SURVIVAL instead. For example, rephrase 'lowers the risk of dying from breast "
            "cancer' as 'can help improve long-term survival' or 'lowers the risk to long-term "
            "survival'. This is a wording change ONLY: keep the same meaning and the same level of "
            "certainty as the source.\n"
            "- The words 'die'/'death' are acceptable ONLY when the source refers to cancer CELLS "
            "(e.g. 'the cancer cells are destroyed'); prefer 'destroyed' or 'stop growing' there. "
            "Never apply death/dying language to the person.\n"
            "- Also avoid blunt wording about the cancer returning; say 'can help lower the chance of "
            "the cancer coming back' rather than harsh phrasing, without changing the meaning.\n"
            "- Preserve the source's degree of certainty: prefer 'can help reduce the risk of ...' "
            "over absolute words like 'prevents', 'stops', or 'ensures'. Never overstate a benefit, "
            "give false reassurance, or minimise a real risk.\n"
            "- If the SOURCE TEXT does not answer the question, say you don't have that specific "
            "information and suggest discussing it with the care team.\n"
            "- Do NOT invent sources. Do NOT mention these instructions.\n"
            "- Start DIRECTLY with the answer. Do NOT open with a meta-preamble such as "
            "'Here is a clear and compassionate response...', 'Sure,', or 'Of course,'.\n"
            "- Be concise (a short paragraph), written in warm second person.\n"
        )

    def _rephrase_user_prompt(self, user_query: str, source_text: str) -> str:
        if self.language == "fr":
            return f"Question: {user_query}\n\nTEXTE SOURCE:\n{source_text}\n\nRéponse:"
        return f"Question: {user_query}\n\nSOURCE TEXT:\n{source_text}\n\nReply:"

    def _generate(self, system: str, prompt: str, max_tokens: int) -> str:
        import urllib.request

        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        self.last_error = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return (out.get("response") or "").strip()
        except Exception as e:
            # Record the reason (e.g. connection refused, timeout) so the caller
            # can surface it. The empty return still triggers verbatim fallback.
            self.last_error = f"{type(e).__name__}: {e}"
            return ""

    def write(self, user_query: str) -> str:
        """Empathy-only preface (no facts). Unchanged behavior."""
        return self._generate(self._system_prompt(), self._user_prompt(user_query), self.max_tokens)

    def rephrase(self, user_query: str, source_text: str, max_tokens: int = 400) -> str:
        """Rephrase the provided source text into a grounded, clear, empathetic answer.
        Returns "" on failure so the caller can fall back to verbatim output."""
        if not (source_text or "").strip():
            return ""
        out = self._generate(
            self._rephrase_system_prompt(),
            self._rephrase_user_prompt(user_query, source_text),
            max_tokens,
        )
        return _strip_llm_preamble(out)


# ---------------------------
# Abstain
# ---------------------------

def build_abstain(language: str) -> str:
    lang = _norm_lang(language)
    if lang == "fr":
        return (
            "Je comprends votre question. Cependant, les informations sur lesquelles je m’appuie ne semblent pas "
            "contenir d’information spécifique sur ce sujet. Pour éviter d’inventer des informations médicales, "
            "je ne peux pas répondre à partir de ces sources.\n\n"
            "Je vous recommande d’en parler avec votre équipe d’oncologie."
        )
    return (
        "I understand why you’re asking. However, the information I have does not appear to contain specific "
        "information about this topic. To avoid inventing medical information, I can’t answer this from my sources.\n\n"
        "Please discuss this with your oncology team."
    )


# ---------------------------
# Scoring + selection
# ---------------------------

def _score_candidate(c: RetrievalCandidate) -> float:
    return float(c.rerank_score) if c.rerank_score is not None else float(c.fused_score)


def _select_bundle_with_coverage(
    candidates: List[RetrievalCandidate],
    language: str,
    anchor_concepts: Set[str],
    max_n: int = 3,
    min_member_sim: Optional[float] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """
    Greedily build a bundle that covers as many anchor concepts as possible.

    The FIRST (lead) member is always the highest-scoring candidate that covers
    anything. SECONDARY members must also be independently relevant: if
    `min_member_sim` is set, a secondary candidate is only added when its dense
    similarity clears that bar. This prevents a strong lead answer from being
    padded with low-relevance entries that happen to share a weak keyword
    (e.g. "long flights" / "stay hydrated" for a "how long do I stay on it?"
    question).
    """
    lang = _norm_lang(language)
    ranked = sorted(candidates, key=_score_candidate, reverse=True)

    covered: Set[str] = set()
    bundle: List[RetrievalCandidate] = []
    per_candidate_covered: List[Dict[str, Any]] = []

    for c in ranked:
        if len(bundle) >= max_n:
            break

        text = f"{c.question} {c.section} {c.answer}"
        cov = _anchor_overlap_concepts(text, anchor_concepts, lang)

        if not cov:
            continue
        if cov.issubset(covered):
            continue

        # Quality bar for secondary members (the lead is always allowed).
        if bundle and min_member_sim is not None:
            if c.dense_sim is not None and c.dense_sim < min_member_sim:
                continue

        bundle.append(c)
        covered |= cov
        per_candidate_covered.append({"idx": c.index, "covered_concepts": sorted(list(cov))})

        if covered == anchor_concepts:
            break

    details = {
        "required_concepts": sorted(list(anchor_concepts)),
        "covered_concepts": sorted(list(covered)),
        "coverage_complete": covered == anchor_concepts,
        "bundle_cover_debug": per_candidate_covered,
    }
    return bundle, details


# ---------------------------
# FIXED: Stats gate
# ---------------------------

def _passes_stats_gate(text: str, language: str) -> bool:
    """
    Stats answers must:
      - contain a number/percent
      - contain a strict stats marker (NOT "how")
    """
    lang = _norm_lang(language)
    if not _has_number_or_percent(text):
        return False

    hints = STATS_GROUND_HINTS_FR if lang == "fr" else STATS_GROUND_HINTS_EN
    hint_stems = {_stem(x, lang) for x in hints}
    text_stems = _stem_set(_tokenize(text), lang)
    return len(text_stems.intersection(hint_stems)) > 0


def _stats_concept_stems_to_require(anchor_concepts: Set[str], language: str) -> Set[str]:
    """
    For stats queries, require overlap with at least one non-broad concept.
    E.g. genetics/mutation should be required; cancer/patient should not.
    """
    lang = _norm_lang(language)

    # remove stats-intent stems from concept set (e.g., percent)
    intent_hints = STATS_INTENT_HINTS_FR if lang == "fr" else STATS_INTENT_HINTS_EN
    intent_stems = {_stem(x, lang) for x in intent_hints}
    concepts = set(anchor_concepts) - set(intent_stems)

    broad = STATS_BROAD_CONCEPTS_FR if lang == "fr" else STATS_BROAD_CONCEPTS_EN
    broad_stems = {_stem(x, lang) for x in broad}
    concepts = concepts - broad_stems

    return concepts


# ---------------------------
# Main answer function
# ---------------------------

def _semantic_top(candidates: List[RetrievalCandidate]) -> Optional[RetrievalCandidate]:
    """Candidate with the highest dense cosine similarity to the query."""
    scored = [c for c in candidates if c.dense_sim is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c.dense_sim if c.dense_sim is not None else -1.0)


def _primary_relevance(c: RetrievalCandidate, uses_rerank: bool) -> Optional[float]:
    """The relevance signal used for the accept decision: cross-encoder rerank
    score when reranking is active, otherwise dense cosine similarity."""
    if uses_rerank and c.rerank_score is not None:
        return float(c.rerank_score)
    if c.dense_sim is not None:
        return float(c.dense_sim)
    return None


def _crosslingual_notice(query_lang: str, answer_lang: str) -> str:
    """One-line note shown when the only relevant content is in another language."""
    q, a = _norm_lang(query_lang), _norm_lang(answer_lang)
    if q == a:
        return ""
    if q == "fr":
        names = {"en": "anglais", "fr": "français"}
        return f"(Cette information n’est disponible qu’en {names.get(a, a)}.)"
    names = {"en": "English", "fr": "French"}
    return f"(This information is only available in {names.get(a, a)}.)"


def _decide_bundle(
    retriever: "HybridFAQRetriever",
    cands: List[RetrievalCandidate],
    lang: str,
    anchor_concepts: Set[str],
    strong_anchors: Set[str],
    stats_intent: bool,
    thresholds: Dict[str, float],
    dbg: Dict[str, Any],
    debug: bool,
) -> Optional[List[RetrievalCandidate]]:
    """
    Run the semantic-first accept decision over ONE candidate subset (used once
    for same-language candidates, then again for cross-language as a fallback).
    Returns the answer bundle, or None to abstain. Writes decision debug into
    `dbg`.
    """
    if not cands:
        if debug:
            dbg["decision_path"] = "abstain(no_candidates)"
        return None

    sem_accept_threshold = thresholds["sem_accept_threshold"]
    rerank_accept_threshold = thresholds["rerank_accept_threshold"]
    dense_floor = thresholds["dense_floor"]

    uses_rerank = any(c.rerank_score is not None for c in cands)

    def _prim(c: RetrievalCandidate) -> float:
        v = _primary_relevance(c, uses_rerank)
        return v if v is not None else -1e9

    top_primary = max(cands, key=_prim)
    primary_threshold = rerank_accept_threshold if uses_rerank else sem_accept_threshold
    primary_value = _primary_relevance(top_primary, uses_rerank)

    if debug:
        dbg["primary_signal"] = "rerank" if uses_rerank else "dense"
        dbg["primary_value"] = round(primary_value, 4) if primary_value is not None else None
        dbg["primary_threshold"] = primary_threshold

    # Stats preference: answer with a grounded statistic if the user asked for one.
    if stats_intent:
        required = _stats_concept_stems_to_require(anchor_concepts, lang)
        ok_stats: List[RetrievalCandidate] = []
        for c in cands:
            text = f"{c.question} {c.section} {c.answer}"
            if not _passes_stats_gate(text, lang):
                continue
            if required and not _anchor_overlap_concepts(text, required, lang):
                continue
            ok_stats.append(c)
        if debug:
            dbg["stats_gate_candidates"] = [{"idx": c.index} for c in ok_stats[:10]]
        if ok_stats:
            best = sorted(ok_stats, key=_score_candidate, reverse=True)[0]
            if debug:
                dbg["decision_path"] = "stats"
            return [best]
        elif debug:
            dbg["stats_fell_through"] = True

    primary_ok = (primary_value is not None) and (primary_value >= primary_threshold)

    # Lexical safety net: a candidate covering a high-IDF anchor at >= dense_floor.
    lexical_lead: Optional[RetrievalCandidate] = None
    if strong_anchors:
        for c in sorted(cands, key=_score_candidate, reverse=True):
            text = f"{c.question} {c.section} {c.answer}"
            if not _anchor_overlap_concepts(text, strong_anchors, lang):
                continue
            if c.dense_sim is not None and c.dense_sim < dense_floor:
                continue
            lexical_lead = c
            break

    if not (primary_ok or lexical_lead is not None):
        if debug:
            dbg["decision_path"] = "abstain"
        return None

    lead = top_primary if primary_ok else lexical_lead
    if debug:
        dbg["decision_path"] = "semantic_primary" if primary_ok else "lexical_net"

    def _member_ok(c: RetrievalCandidate) -> bool:
        v = _primary_relevance(c, uses_rerank)
        return v is not None and v >= primary_threshold

    relevant_pool = [c for c in cands if _member_ok(c)]
    if lead is not None and lead not in relevant_pool:
        relevant_pool = [lead] + relevant_pool

    if anchor_concepts:
        bundle, cov_details = _select_bundle_with_coverage(
            relevant_pool, lang, anchor_concepts, max_n=3
        )
        if debug:
            dbg["coverage_gate"] = cov_details
    else:
        bundle = []

    if lead is not None:
        bundle = [lead] + [c for c in bundle if c.index != lead.index]
    if not bundle and lead is not None:
        bundle = [lead]

    # Multi-source: fill the remaining slots with the next most relevant
    # candidates (all above the accept threshold), so the answer presents the
    # top few relevant FAQ entries rather than betting everything on the
    # reranker's single #1. This surfaces complementary answers (e.g. a
    # "therapeutic break" entry alongside a "recurrence risk" entry).
    have = {c.index for c in bundle}
    for c in sorted(relevant_pool, key=_score_candidate, reverse=True):
        if len(bundle) >= 3:
            break
        if c.index not in have:
            bundle.append(c)
            have.add(c.index)

    return bundle if bundle else None


def answer_query(
    retriever: "HybridFAQRetriever",
    user_query: str,
    use_llm: bool = False,
    llm_model: str = "llama3.2",
    debug: bool = False,
    sem_accept_threshold: float = 0.62,
    rerank_accept_threshold: float = -1.0,  # mmarco cross-encoder scores relevant items ~ -1..+2
    dense_floor: float = 0.50,
    coverage_fraction: float = 0.5,  # deprecated: kept for call compatibility
) -> AnswerResult:
    """
    SEMANTIC-FIRST answer/abstain decision.

    The primary relevance signal is the cross-encoder rerank score (when
    reranking is on) or dense cosine similarity (otherwise). Lexical / IDF
    anchors NO LONGER gate the answer -- they only (a) shape the bundle for
    multi-concept questions and (b) provide a recall safety net for rare,
    highly specific terms. This is what lets the IDF anchor extractor stay
    list-free without risking false abstentions.

    Accept if EITHER:
      1. Primary: top candidate's relevance >= its threshold
         (rerank >= `rerank_accept_threshold`, or dense >= `sem_accept_threshold`).
      2. Lexical safety net: a candidate covers a HIGH-IDF anchor (idf >=
         corpus average) AND is at least `dense_floor` cosine-similar. Catches
         specific-term matches the reranker/embedder may under-score.

    Thresholds MUST be calibrated on the gold eval set (tests/eval_set.jsonl)
    for the deployed models. Defaults are conservative starting points, NOT
    tuned values. For a patient-facing tool, watch the false-ANSWER rate when
    lowering thresholds.
    """
    lang = _norm_lang(retriever.language)
    _t_total = time.perf_counter()
    candidates = retriever.retrieve(user_query)
    timing = dict(getattr(retriever, "last_timing_ms", {}) or {})
    timing["llm"] = 0.0

    ax = extract_anchor_concepts(user_query, lang, retriever)
    anchors = ax.anchors
    anchor_concepts = set(anchors)
    # "Strong" (specific) anchors for the lexical safety net: terms that appear
    # in only a small share of the corpus. Defined by document-frequency, NOT by
    # idf >= average_idf -- the latter collapses to the empty set once a large
    # article corpus inflates the average, disabling the net entirely.
    strong_df_cap = max(1, int(0.15 * retriever.corpus_size))
    strong_anchors = {a for a in anchor_concepts if 1 <= retriever.df(a) <= strong_df_cap}
    stats_intent = _is_stats_intent(user_query, lang)

    sem_top = _semantic_top(candidates)
    sem_top_sim = (sem_top.dense_sim if (sem_top and sem_top.dense_sim is not None) else None)

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["query_lang"] = lang
        dbg["timing_ms"] = timing  # same dict object; llm/total filled in below
        dbg["shared_mode"] = bool(getattr(retriever, "shared", False))
        dbg["anchors"] = anchors
        dbg["strong_anchors"] = sorted(list(strong_anchors))
        dbg["present_concepts"] = [f"{s}(idf={idf:.2f},df={d})" for (s, idf, d) in ax.present]
        dbg["absent_concepts"] = ax.dropped_absent
        dbg["dropped_common"] = ax.dropped_common
        dbg["stats_intent"] = stats_intent
        dbg["sem_top_sim"] = (round(sem_top_sim, 4) if sem_top_sim is not None else None)
        dbg["sem_accept_threshold"] = sem_accept_threshold
        dbg["qp_index_loaded"] = (retriever._index_qp is not None)  # type: ignore[attr-defined]
        dbg["top_candidates"] = [
            {"idx": c.index, "lang": c.lang, "fused": round(c.fused_score, 5),
             "rerank": (round(c.rerank_score, 3) if c.rerank_score is not None else None),
             "dense": (round(c.dense_sim, 4) if c.dense_sim is not None else None),
             "q": c.question[:120]}
            for c in candidates[:15]
        ]

    if not candidates:
        timing["total"] = (time.perf_counter() - _t_total) * 1000.0
        return AnswerResult(answered=False, answer_text=build_abstain(lang),
                            timing_ms=timing, debug=(dbg if debug else None))

    thresholds = {
        "sem_accept_threshold": sem_accept_threshold,
        "rerank_accept_threshold": rerank_accept_threshold,
        "dense_floor": dense_floor,
    }

    # Same-language first, cross-lingual fallback. In per-language mode every
    # candidate already shares the query language, so `cross` is empty.
    if getattr(retriever, "shared", False):
        same = [c for c in candidates if c.lang == lang]
        cross = [c for c in candidates if c.lang != lang]
    else:
        same, cross = candidates, []

    notice = ""
    bundle = _decide_bundle(retriever, same, lang, anchor_concepts, strong_anchors,
                            stats_intent, thresholds, dbg, debug)

    if bundle is None and cross:
        cross_dbg: Dict[str, Any] = {}
        cross_bundle = _decide_bundle(retriever, cross, lang, anchor_concepts, strong_anchors,
                                      stats_intent, thresholds, cross_dbg, debug)
        if cross_bundle is not None:
            bundle = cross_bundle
            notice = _crosslingual_notice(lang, bundle[0].lang)
            if debug:
                dbg["crosslingual_fallback"] = True
                dbg["answer_lang"] = bundle[0].lang
                dbg["cross_pass"] = cross_dbg

    if not bundle:
        if debug:
            dbg.setdefault("decision_path", "abstain")
        timing["total"] = (time.perf_counter() - _t_total) * 1000.0
        return AnswerResult(answered=False, answer_text=build_abstain(lang),
                            timing_ms=timing, debug=(dbg if debug else None))

    # Labels/preface are in the QUERY language. Source citations are computed here
    # but kept OUT of the answer body: the UI shows them in the "Sources used for
    # this answer" dropdown (built from source_indices), and CLI/verifiability
    # consumers can read them from AnswerResult.sources_text.
    sources = _format_sources(lang, bundle)
    notice_block = (notice + "\n\n") if notice else ""

    def _verbatim_answer() -> str:
        # FAQ content quoted verbatim + fixed empathy sentence (original behavior).
        faq_body = _format_bundle_body(lang, bundle)
        factual_block = _format_full_answer(lang, faq_body, "")
        prefix = _fallback_empathy(lang, user_query, bundle[0].question)
        return (prefix + notice_block + factual_block).strip()

    llm_status: Optional[str] = None
    llm_error: Optional[str] = None
    if use_llm:
        # Grounded rephrasing: the LLM rewrites ONLY the retrieved source text for
        # clarity + empathy (no new facts), then the verbatim sources are cited.
        # If the LLM is unreachable/empty, fall back to the verbatim answer so we
        # never lose a response -- but record WHY so the CLI/GUI can say so.
        source_text = _bundle_source_text(bundle)
        _t_llm = time.perf_counter()
        writer = LLMWrapperWriter(language=lang, model=llm_model)
        rephrased = writer.rephrase(user_query=user_query, source_text=source_text).strip()
        timing["llm"] = (time.perf_counter() - _t_llm) * 1000.0
        if rephrased:
            answer_text = (notice_block + rephrased).strip()
            llm_status = "used"
            if debug:
                dbg["llm_mode"] = "rephrase"
        else:
            # Distinguish "could not reach the LLM" from "LLM returned nothing".
            answer_text = _verbatim_answer()
            llm_error = writer.last_error
            llm_status = "fallback_unreachable" if writer.last_error else "fallback_empty"
            if debug:
                dbg["llm_mode"] = "verbatim_fallback"
                dbg["llm_error"] = writer.last_error
    else:
        answer_text = _verbatim_answer()
        if debug:
            dbg["llm_mode"] = "verbatim"

    # Compassion guardrail: reframe patient-mortality phrasing ("dying from breast
    # cancer") into survival language. Deterministic, so it holds regardless of
    # what the local LLM emits and covers the verbatim path too. Applied to the
    # answer body only; cited sources in the dropdown stay verbatim.
    answer_text = _reframe_survival(answer_text, lang)

    timing["total"] = (time.perf_counter() - _t_total) * 1000.0

    top = bundle[0]
    return AnswerResult(
        answered=True,
        answer_text=answer_text,
        source_title=_candidate_title(top),   # FAQ question or article source, never a filename breadcrumb
        source_section=top.section,
        source_index=top.index,
        source_indices=[c.index for c in bundle],
        sources_text=sources,
        timing_ms=timing,
        llm_status=llm_status,
        llm_error=llm_error,
        debug=(dbg if debug else None),
    )


# ---------------------------
# Debug printer
# ---------------------------

def print_debug(result: AnswerResult) -> None:
    if not result.debug:
        return

    print("\n[DEBUG] query_lang:", result.debug.get("query_lang"),
          "| shared_mode:", result.debug.get("shared_mode"),
          "| crosslingual_fallback:", result.debug.get("crosslingual_fallback", False))
    print("[DEBUG] anchors:", result.debug.get("anchors"))
    print("[DEBUG] strong_anchors:", result.debug.get("strong_anchors"))
    print("[DEBUG] present_concepts (idf,df):", result.debug.get("present_concepts"))
    print("[DEBUG] dropped_absent:", result.debug.get("absent_concepts"))
    print("[DEBUG] dropped_common:", result.debug.get("dropped_common"))
    print("[DEBUG] stats_intent:", result.debug.get("stats_intent"))
    print("[DEBUG] primary_signal:", result.debug.get("primary_signal"),
          "value:", result.debug.get("primary_value"),
          "threshold:", result.debug.get("primary_threshold"))
    print("[DEBUG] sem_top_sim:", result.debug.get("sem_top_sim"))
    print("[DEBUG] decision_path:", result.debug.get("decision_path"))
    print("[DEBUG] qp_index_loaded:", result.debug.get("qp_index_loaded"))

    t = result.debug.get("timing_ms") or {}
    if t:
        print("[DEBUG] timing_ms: retrieve={:.0f} rerank={:.0f} (n={}) llm={:.0f} total={:.0f}".format(
            t.get("retrieve", 0.0), t.get("rerank", 0.0), t.get("n_reranked", 0),
            t.get("llm", 0.0), t.get("total", 0.0)))

    if "stats_required_concepts" in result.debug:
        print("[DEBUG] stats_required_concepts:", result.debug.get("stats_required_concepts"))

    if "coverage_gate" in result.debug:
        print("[DEBUG] coverage_gate:", result.debug.get("coverage_gate"))

    if "stats_gate_candidates" in result.debug:
        print("[DEBUG] stats_gate_candidates:", result.debug.get("stats_gate_candidates"))

    print("[DEBUG] Retrieved candidates:")
    for row in result.debug.get("top_candidates", []):
        print(f"  idx={row['idx']} fused={row['fused']} rerank={row['rerank']} "
              f"dense={row.get('dense')} | Q={row['q']}")
