from __future__ import annotations

import os
import re
import json
import pickle
import hashlib
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
    "m","im","ive","id","ill","dont","cant","wont","youre","were","theyre","isnt","arent",
    "have","having",
    # function words / discourse fillers that carry no retrievable concept.
    # These previously survived as "anchors" and could never be matched in the
    # corpus, forcing false abstentions (e.g. "ever" in the break/risk example).
    "ever","never","always","really","actually","just","still","instead","anymore",
    "sometimes","often","maybe","perhaps","also","too","much","many","more","most",
    "even","yet","already","soon","later","now","thing","things","stuff","way","ways",
    "okay","ok","please","thanks","thank","hello","hi","hey","yes","no","not",
    "am","being","being","about","around","over","under","between",
    "want","wanting","wanted","wonder","wondering","tell","told","ask","asking","know","knowing",
    "mean","means","meant","like","kind","sort","bit","lot","lots",
}

FR_STOPWORDS = {
    "le","la","les","un","une","des","et","ou","mais","si","alors",
    "de","du","dans","sur","pour","avec","par","au","aux","en",
    "est","sont","été","etre","être","avoir","a","ont",
    "je","tu","il","elle","nous","vous","ils","elles",
    "ce","cet","cette","ces",
    "quoi","pourquoi","comment","quand","où","ou","quel","quelle","quels","quelles",
    "j","t","c","d","l","n","qu",
    "ai","as","avons","avez","ont","avais","avait","aviez","avaient",
    # French function words / fillers (mirror of the English additions).
    "jamais","toujours","vraiment","juste","encore","plutôt","plutot","aussi","trop",
    "beaucoup","plus","parfois","souvent","peut-être","peut","être","déjà","deja",
    "maintenant","bientôt","bientot","chose","choses","truc","trucs","façon","facon",
    "d'accord","daccord","merci","bonjour","salut","oui","non","pas","ne",
    "vouloir","veux","veut","savoir","sais","sait","dire","dis","dit","demander",
    "genre","sorte","peu",
}

GENERIC_EN = {
    "safe","safety","careful","need","should","can","could","would","risk","danger",
    "allowed","ok","okay","possible","recommend","recommended","advice",
    "hormone","hormonal","therapy","treatment","medication","pill","medicine","drug","drugs",
    "take","taking","taken",
    "get","getting","got",
    "health",
    "issue","issues","problem","problems","symptom","symptoms","trouble","troubles",
}

GENERIC_FR = {
    "sûr","sur","sure","sécurité","securite","prudent","prudence","besoin","dois","devrais","peux",
    "risque","danger","autorisé","autorise","possible","recommandé","recommande","conseil",
    "hormone","hormonale","hormonothérapie","hormonotherapie",
    "traitement","thérapie","therapie","médicament","medicament","comprimé","comprime","pilule",
    "prendre","prends","pris",
    "sante","santé",
    "problème","probleme","problèmes","problemes","souci","soucis","symptôme","symptome","symptômes","symptomes",
}

EMOTION_EN = {
    "worried","worry","concerned","concern","anxious","anxiety","scared","afraid",
    "fear","terrified","panic","panicking","stressed","stress","upset",
}

EMOTION_FR = {
    "inquiet","inquiete","inquiète","inquiét","inquietude","inquiétude",
    "angoisse","angoissé","angoissee","peur","effrayé","effrayee",
    "stress","stresse","stressé","stressée","préoccupé","preoccupe","préoccupée","preoccupee",
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


def _emotion_stems(language: str) -> Set[str]:
    lang = _norm_lang(language)
    emo = EMOTION_FR if lang == "fr" else EMOTION_EN
    return {_stem(w, lang) for w in emo}


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
# Keyword extraction / anchors
# ---------------------------

def extract_core_keywords(user_query: str, language: str) -> List[str]:
    lang = _norm_lang(language)
    toks = _tokenize(user_query)

    stop = FR_STOPWORDS if lang == "fr" else EN_STOPWORDS
    gen = GENERIC_FR if lang == "fr" else GENERIC_EN
    emo = EMOTION_FR if lang == "fr" else EMOTION_EN

    out: List[str] = []
    for t in toks:
        if t in stop:
            continue
        if t in gen:
            continue
        if t in emo:
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
        self._index_qp: Optional[faiss.Index] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._cross_encoder = None

        self._stored_embedding_model_name: Optional[str] = None

        # Set of every stem that appears anywhere in the corpus. Used to prune
        # anchor concepts that can never be covered (junk/out-of-vocab tokens).
        self._corpus_stems: Set[str] = set()

    def load(self) -> None:
        prefix = f"faq_{self.language}"
        qa_path = os.path.join(self.data_dir, f"{prefix}_qa.pkl")
        bm25_path = os.path.join(self.data_dir, f"{prefix}_bm25.pkl")
        idx_q_path = os.path.join(self.data_dir, f"{prefix}_index_q.faiss")
        idx_qa_path = os.path.join(self.data_dir, f"{prefix}_index_qa.faiss")
        idx_qp_path = os.path.join(self.data_dir, f"{prefix}_index_qp.faiss")

        with open(qa_path, "rb") as f:
            payload = pickle.load(f)
        self._items = payload["items"]
        self._stored_embedding_model_name = payload.get("embedding_model_name")

        # Precompute the set of corpus stems (question + section + answer, plus
        # any stored paraphrases) for anchor-concept pruning.
        self._corpus_stems = set()
        for it in self._items:
            blob = f"{it.get('question','')} {it.get('section','')} {it.get('answer','')}"
            for para in (it.get("q_paraphrases") or []):
                blob += " " + str(para)
            self._corpus_stems |= _stem_set(_tokenize(blob), self.language)

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

    def stem_in_corpus(self, stem: str) -> bool:
        """True if the given stem (or any of its synonyms) appears in the corpus."""
        if not self._corpus_stems:
            return True  # corpus stems unavailable: do not prune
        acceptable = _concept_match_stems(stem, self.language)
        return len(self._corpus_stems.intersection(acceptable)) > 0

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        if not self._items:
            return []

        assert self._bm25 is not None
        assert self._index_q is not None
        assert self._index_qa is not None

        bm25_scores = self._bm25.get_scores(_tokenize(user_query))
        bm25_ranked = np.argsort(-bm25_scores)[: self.top_k]

        # The three dense indexes are queried with the same encoded vector; only
        # the document side differs, so encode once (was encoded three times).
        q_emb = self._encode(f"Question: {user_query}")
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
            if not _looks_like_faq_question(question):
                continue
            candidates.append(
                RetrievalCandidate(
                    index=int(idx),
                    question=question,
                    section=str(it.get("section", "")),
                    answer=str(it.get("answer", "")),
                    fused_score=float(score),
                    dense_sim=(dense_sim.get(int(idx)) if int(idx) in dense_sim else None),
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
# Formatting functions
# ---------------------------

def _format_bundle_body(language: str, bundle: List[RetrievalCandidate]) -> str:
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
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return (out.get("response") or "").strip()
        except Exception:
            return ""


# ---------------------------
# Abstain
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


def answer_query(
    retriever: "HybridFAQRetriever",
    user_query: str,
    use_llm: bool = False,
    llm_model: str = "llama3.2",
    debug: bool = False,
    sem_accept_threshold: float = 0.62,
    coverage_fraction: float = 0.5,
) -> AnswerResult:
    """
    Answer/abstain decision.

    Two independent accept paths (either one answers):
      1. Coverage floor: the candidate bundle lexically covers at least
         ceil(coverage_fraction * N) of the in-corpus anchor concepts
         (N = number of anchor concepts that actually appear in the corpus).
         This replaces the old all-concepts AND gate.
      2. Semantic accept: the top dense-retrieval candidate is at least
         `sem_accept_threshold` cosine-similar to the query, regardless of
         lexical overlap. This rescues lay/clinical wording mismatches.

    Both thresholds MUST be calibrated on the gold eval set
    (tests/eval_set.jsonl) for the deployed embedding model.
    """
    lang = retriever.language
    candidates = retriever.retrieve(user_query)

    core_kws = extract_core_keywords(user_query, lang)
    anchors = anchor_keywords(core_kws, lang)
    anchor_concepts = _stem_set(anchors, lang)

    # Prune anchor concepts that cannot possibly be covered because neither the
    # concept nor any synonym appears in the corpus (e.g. "ever"). These were
    # the direct cause of false abstentions under the old AND gate.
    present_concepts = {c for c in anchor_concepts if retriever.stem_in_corpus(c)}
    absent_concepts = anchor_concepts - present_concepts

    stats_intent = _is_stats_intent(user_query, lang)
    sem_top = _semantic_top(candidates)
    sem_top_sim = (sem_top.dense_sim if (sem_top and sem_top.dense_sim is not None) else None)

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["core_kws"] = core_kws
        dbg["anchors"] = anchors
        dbg["anchor_stems"] = sorted(list(anchor_concepts))[:40]
        dbg["present_concepts"] = sorted(list(present_concepts))[:40]
        dbg["absent_concepts"] = sorted(list(absent_concepts))[:40]
        dbg["stats_intent"] = stats_intent
        dbg["sem_top_sim"] = (round(sem_top_sim, 4) if sem_top_sim is not None else None)
        dbg["sem_accept_threshold"] = sem_accept_threshold
        dbg["qp_index_loaded"] = (retriever._index_qp is not None)  # type: ignore[attr-defined]
        dbg["top_candidates"] = [
            {"idx": c.index, "fused": round(c.fused_score, 5),
             "rerank": (round(c.rerank_score, 3) if c.rerank_score is not None else None),
             "dense": (round(c.dense_sim, 4) if c.dense_sim is not None else None),
             "q": c.question[:140], "section": c.section[:140]}
            for c in candidates[:10]
        ]

    # Only hard-abstain when there is nothing to work with at all. A query made
    # up solely of drug/treatment terms (empty anchors) is NOT abstained here;
    # it falls through to the semantic path below.
    if len(core_kws) < 1 or not candidates:
        return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    bundle: Optional[List[RetrievalCandidate]] = None

    # -----------------------
    # Stats questions: prefer a grounded statistic when the user asked for one.
    # If no grounded statistic is found we do NOT hard-abstain: we fall through
    # to the general path, which can still answer qualitatively from the FAQ
    # (still quoting the FAQ, never fabricating numbers).
    # -----------------------
    if stats_intent:
        # Only require concepts that actually appear in the corpus.
        required_concepts = _stats_concept_stems_to_require(present_concepts, lang)

        if debug:
            dbg["stats_required_concepts"] = sorted(list(required_concepts))[:20]

        ok_stats_candidates: List[RetrievalCandidate] = []
        for c in candidates:
            text = f"{c.question} {c.section} {c.answer}"

            # must be stats-like
            if not _passes_stats_gate(text, lang):
                continue

            # must match at least one non-broad concept if we have any
            if required_concepts:
                cov = _anchor_overlap_concepts(text, required_concepts, lang)
                if not cov:
                    continue

            ok_stats_candidates.append(c)

        if debug:
            dbg["stats_gate_candidates"] = [{"idx": c.index, "q": c.question[:120]} for c in ok_stats_candidates[:10]]

        if ok_stats_candidates:
            best = sorted(ok_stats_candidates, key=_score_candidate, reverse=True)[0]
            bundle = [best]
            if debug:
                dbg["decision_path"] = "stats"
        elif debug:
            # No grounded statistic; general path takes over below.
            dbg["stats_fell_through"] = True

    if bundle is None:
        # -----------------------
        # General path: coverage floor OR semantic accept
        # (also handles stats queries that found no grounded statistic)
        # -----------------------
        import math

        semantic_ok = (sem_top_sim is not None) and (sem_top_sim >= sem_accept_threshold)

        if not present_concepts:
            # Nothing lexically anchorable (e.g. drug-only query, or all anchors
            # are out-of-vocab). Rely entirely on the semantic signal.
            if debug:
                dbg["decision_path"] = "semantic_only"
            if semantic_ok and sem_top is not None:
                bundle = [sem_top]
            else:
                return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))
        else:
            n_present = len(present_concepts)
            required = max(1, math.ceil(coverage_fraction * n_present))
            bundle, cov_details = _select_bundle_with_coverage(
                candidates, lang, present_concepts, max_n=3,
                min_member_sim=sem_accept_threshold,
            )
            covered_n = len(cov_details["covered_concepts"])
            coverage_ok = covered_n >= required
            cov_details["required_cover_count"] = required
            cov_details["covered_cover_count"] = covered_n
            cov_details["coverage_ok"] = coverage_ok
            if debug:
                dbg["coverage_gate"] = cov_details
                dbg["decision_path"] = "coverage_floor+semantic"

            if not coverage_ok:
                # Coverage floor missed. Rescue with a strong semantic match.
                if semantic_ok and sem_top is not None:
                    if sem_top not in bundle:
                        bundle = [sem_top] + bundle
                    if debug:
                        dbg["decision_path"] = "semantic_rescue"
                else:
                    return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

        if not bundle:
            return AnswerResult(answered=False, answer_text=build_abstain(lang), debug=(dbg if debug else None))

    # Build FAQ answer
    faq_body = _format_bundle_body(lang, bundle)
    sources = _format_sources(lang, bundle)
    factual_block = _format_full_answer(lang, faq_body, sources)

    # Empathy prefix (only when answering)
    if use_llm:
        wrapper = LLMWrapperWriter(language=lang, model=llm_model).write(user_query=user_query).strip()
        prefix = (wrapper + "\n\n") if wrapper else _fallback_empathy(lang, user_query, bundle[0].question)
        if debug and not wrapper:
            dbg["llm_wrapper_fallback"] = True
    else:
        prefix = _fallback_empathy(lang, user_query, bundle[0].question)

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


# ---------------------------
# Debug printer
# ---------------------------

def print_debug(result: AnswerResult) -> None:
    if not result.debug:
        return

    print("\n[DEBUG] core_kws:", result.debug.get("core_kws"))
    print("[DEBUG] anchors:", result.debug.get("anchors"))
    print("[DEBUG] anchor_stems:", result.debug.get("anchor_stems"))
    print("[DEBUG] present_concepts:", result.debug.get("present_concepts"))
    print("[DEBUG] absent_concepts:", result.debug.get("absent_concepts"))
    print("[DEBUG] stats_intent:", result.debug.get("stats_intent"))
    print("[DEBUG] sem_top_sim:", result.debug.get("sem_top_sim"),
          "(threshold:", str(result.debug.get("sem_accept_threshold")) + ")")
    print("[DEBUG] decision_path:", result.debug.get("decision_path"))
    print("[DEBUG] qp_index_loaded:", result.debug.get("qp_index_loaded"))

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
