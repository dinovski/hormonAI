import os
import html
import re
import streamlit as st

from rag_core import (
    HybridFAQRetriever,
    answer_query,
)
from audit_logger import AuditLogger

# ---------- BASIC PAGE CONFIG ----------
st.set_page_config(
    page_title="hormonAI – Breast Cancer Support Chatbot",
    page_icon="💗",
    layout="centered",
)

# ---------- LAUNCH-TIME CONFIG ----------
# The GUI accepts EXACTLY the same launch arguments as the CLI (chatbot.py), via
# Streamlit's passthrough after `--`, e.g.:
#   streamlit run hormonai_app.py -- --rerank-top-n 10 --rerank-model BAAI/bge-reranker-v2-m3
# We reuse the CLI's argparse definition verbatim so the two never drift. With no
# passthrough args, every value falls back to its CLI default (which reads the
# same HORMONAI_* env vars), so existing env-based configuration keeps working.
# Streamlit puts only the post-`--` tokens on sys.argv, so this parses cleanly.
from chatbot import parse_args as _parse_cli_args  # noqa: E402

ARGS = _parse_cli_args()

DEFAULT_LLM_MODEL = ARGS.llm_model
# Must match the model used at ingestion (e.g. BAAI/bge-m3). Overriding the
# embedding model without re-ingesting will fail the FAISS dimension check.
DEFAULT_EMBEDDING_MODEL = ARGS.embedding_model
# Reranker model. Upgrade to a stronger multilingual cross-encoder (e.g.
# BAAI/bge-reranker-v2-m3, which pairs with BGE-M3) via --rerank-model.
# Changing it requires re-calibrating --rerank-threshold (different scale).
DEFAULT_RERANK_MODEL = ARGS.rerank_model
# Only rerank the top-N fused candidates. Lower = faster (esp. with a large
# reranker like bge-reranker-v2-m3 on CPU); higher = a bit more recall.
DEFAULT_RERANK_TOP_N = ARGS.rerank_top_n
DEFAULT_TOP_K = ARGS.top_k
DEFAULT_DATA_DIR = ARGS.data_dir
# Accept/abstain thresholds. Defaults are conservative and model-specific --
# calibrate on the gold eval set, then pass --sem-threshold / --rerank-threshold
# / --dense-floor at launch (or set the matching HORMONAI_* env vars).
RERANK_ACCEPT_THRESHOLD = ARGS.rerank_threshold
SEM_ACCEPT_THRESHOLD = ARGS.sem_threshold
DENSE_FLOOR = ARGS.dense_floor
# --debug (or HORMONAI_SHOW_TIMING=1) prints a per-query latency breakdown under
# each answer (retrieve / rerank / llm / total). Off by default (patient UI).
SHOW_TIMING = bool(getattr(ARGS, "debug", False)) or os.getenv("HORMONAI_SHOW_TIMING", "0") == "1"
# HORMONAI_AUDIT_LATENCY=1 appends a per-query LATENCY record (timing + language
# + answered ONLY -- never the raw patient query) for p50/p95 tracking. Off by
# default. Path from --audit-log (CLI default logs/audit.jsonl).
AUDIT_LATENCY = os.getenv("HORMONAI_AUDIT_LATENCY", "0") == "1"
AUDIT_LOG_PATH = ARGS.audit_log


# ---------- SAMPLE PROMPTS (from patient-forum research) ----------
SAMPLE_PROMPTS = {
    "en": {
        "Side effects": [
            "What are the most common side effects?",
            "Do side effects like hot flashes last the entire duration of treatment?",
            "Who should I contact if I experience troubling side effects?",
        ],
        "Starting or stopping treatment": [
            "Is it more effective to take hormone therapy in the morning, afternoon, or evening?",
            "Is it important to take my hormone therapy pill at the same time every day?",
            "Is it ever okay to take a break from hormone therapy, or does that increase my risk?",
        ],
        "Mood, sleep, cognition": [
            "What can I ask my care team about managing depression or brain fog?",
        ],
        "Nutrition": [
            "Are there foods I must avoid during treatment?",
            "Is it safe to drink herbal teas?",
        ],
        "Managing side effects": [
            "Are there ways to reduce side effects, like changing when I take my dose or switching brands?",
            "What non-drug options are there for joint pain from aromatase inhibitors?",
        ],
        "Sexual and vaginal health": [
            "Is it safe to use vaginal estrogen?",
            "What are my options for painful sex or low libido on this treatment?",
        ],
        "Long-term monitoring": [
            "What tests should I be getting regularly — bone scans, cholesterol, anything else?",
            "Does hormone therapy require cardiovascular monitoring?",
        ],
        "Duration and stopping": [
            "How long will I need to take hormone therapy?",
            "What happens to my body when I eventually stop hormone therapy?",
        ],
    },
    "fr": {
        "Effets secondaires": [
            "Quels sont les effets secondaires les plus courants de l’hormonothérapie ?",
            "Les effets indésirables, comme les bouffées de chaleur, durent-ils toute la durée du traitement ?",
            "Qui contacter en cas d'effet indésirable ?",
        ],
        "Commencer ou arrêter le traitement": [
            "Y a-t-il une meilleure efficacité à prendre le traitement le matin, le midi ou le soir ?",
            "Est-ce important de prendre le cachet à heure fixe ?",
            "Est-ce parfois acceptable de faire une pause dans l'hormonothérapie, ou cela augmente-t-il mon risque ?",
        ],
        "Humeur, sommeil, cognition": [
            "Que puis-je demander à mon équipe soignante pour gérer la dépression ou le brouillard mental pendant ce traitement ?",
        ],
        "Alimentation": [
            "Y a-t-il des aliments interdits pendant le traitement ?",
            "Les tisanes sont-elles compatibles avec l’hormonothérapie ?",
        ],
        "Gérer les effets secondaires": [
            "Existe-t-il des moyens de réduire les effets secondaires, comme changer l'heure de ma prise ou changer de marque ?",
            "Quelles sont les options non médicamenteuses pour les douleurs articulaires liées aux inhibiteurs de l'aromatase ?",
        ],
        "Santé sexuelle et vaginale": [
            "Est-il sûr d'utiliser des œstrogènes vaginaux contre la sécheresse ?",
            "Quelles sont mes options en cas de rapports douloureux ou de baisse de libido pendant ce traitement ?",
        ],
        "Suivi à long terme": [
            "Quels examens devrais-je faire régulièrement — densité osseuse, cholestérol, autre chose ?",
            "L'hormonothérapie nécessite-t-elle une surveillance cardiovasculaire ?",
        ],
        "Durée et arrêt du traitement": [
            "Pendant combien de temps devrai-je suivre une hormonothérapie ?",
            "Que se passe-t-il dans mon corps quand j'arrête finalement l'hormonothérapie ?",
        ],
    },
}


# ---------- RENDER HELPERS ----------
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def render_bubble_text(text: str) -> str:
    """
    Render text inside our HTML chat bubbles while supporting **bold** markup.

    We:
    1) HTML-escape everything (safety)
    2) Convert escaped **...** to <strong>...</strong>
    3) Convert newlines to <br>
    """
    safe = html.escape(text or "")
    safe = BOLD_RE.sub(r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return safe


# ---------- GLOBAL STYLES ----------
st.markdown(
    """
    <style>
    /* ---------------------------------------------------------
       Force light-mode color variables at the source. Streamlit
       auto-detects OS/browser dark mode and swaps internal CSS
       variables (e.g. --text-color) to white; overriding the
       variables themselves here fixes every widget that reads
       from them, instead of patching each widget individually.
       --------------------------------------------------------- */
    :root, html, body {
        color-scheme: light !important;
        --text-color: #b26a7c !important;
        --background-color: #ffffff !important;
        --secondary-background-color: #ffeef6 !important;
    }

    /* Global text color override: warm rose-plum, no black text */
    * {
        color: #b26a7c !important;
    }

    /* Overall page styling */
    .stApp {
        background-color: #ffffff;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffeef6;
        border-right: 1px solid #f8d4e5;
    }

    /* Sidebar headings slightly stronger */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #a34e68 !important;
        font-weight: 650;
    }

    /* Checkbox / radio accent colors */
    input[type="checkbox"], input[type="radio"] {
        accent-color: #ff4da6;
    }

    /* Sidebar labels */
    [data-testid="stSidebar"] label {
        color: #b26a7c !important;
    }

    /* Base input in sidebar */
    [data-testid="stSidebar"] input[type="text"] {
        border-radius: 10px;
        border: 1px solid #ff9ccc !important;
        box-shadow: none !important;
        background-color: #fff9fd !important;
        color: #b26a7c !important;
    }

    /* Streamlit selectbox: closed state */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffdde9 !important;
        border-radius: 10px !important;
        border: 1px solid #ff9ccc !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[role="button"] {
        background-color: transparent !important;
        color: #b26a7c !important;
    }
    [data-testid="stSidebar"] .stSelectbox * {
        color: #b26a7c !important;
    }

    /* ---------------------------------------------------------
       FIX: dropdown menu open-state (language dropdown black)
       --------------------------------------------------------- */

    div[data-baseweb="popover"] {
        background-color: transparent !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #fff9fd !important;
        border: 1px solid #f3c4d9 !important;
        border-radius: 12px !important;
        padding: 6px !important;
        box-shadow: 0 6px 18px rgba(179, 106, 124, 0.18) !important;
    }

    ul[data-baseweb="menu"] li {
        background-color: #fff9fd !important;
        color: #b26a7c !important;
        border-radius: 10px !important;
    }

    ul[data-baseweb="menu"] li * {
        color: #b26a7c !important;
        background-color: transparent !important;
    }

    ul[data-baseweb="menu"] li:hover {
        background-color: #ffeef6 !important;
    }

    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #ffe1ef !important;
    }

    div[role="listbox"] {
        background-color: #fff9fd !important;
        border-radius: 12px !important;
        border: 1px solid #f3c4d9 !important;
    }
    div[role="listbox"] * {
        color: #b26a7c !important;
        background-color: transparent !important;
    }
    div[role="option"][aria-selected="true"] {
        background-color: #ffe1ef !important;
    }
    div[role="option"]:hover {
        background-color: #ffeef6 !important;
    }

    /* Center the header block */
    .hormonai-header {
        text-align: center;
        margin-bottom: 0.75rem;
    }

    .hormonai-title {
        font-size: 2.6rem;
        font-weight: 750;
        color: #b26a7c !important;
        margin-bottom: 0.25rem;
    }

    .hormonai-subtitle {
        font-size: 1.05rem;
        color: #bf7a8b !important;
        margin-bottom: 1.0rem;
    }

    /* Accent color buttons (Send) */
    .stButton > button {
        border-radius: 999px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        cursor: pointer;
        background-color: #d93f88 !important;
        background-image: none !important;
        color: #ffffff !important;
        box-shadow: 0 3px 8px rgba(217, 63, 136, 0.35);
    }
    .stButton > button * {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        opacity: 0.96;
        box-shadow: 0 4px 10px rgba(217, 63, 136, 0.45);
    }

    /* Chat message bubbles */
    .chat-bubble-user {
        background-color: #ffe1ef;
        padding: 0.85rem 1.1rem;
        border-radius: 18px;
        margin: 0.3rem 0 0.6rem 0;
        max-width: 75%;
        margin-left: auto;
        border: 1px solid #ffb3d2;
        color: #b26a7c !important;
        box-shadow: 0 2px 6px rgba(255, 154, 203, 0.25);
    }
    .chat-bubble-bot {
        background-color: #e6faf7;
        padding: 0.85rem 1.1rem;
        border-radius: 18px;
        margin: 0.3rem 0 0.6rem 0;
        max-width: 75%;
        margin-right: auto;
        border: 1px solid #8ad5c8;
        color: #8c6474 !important;
        box-shadow: 0 2px 6px rgba(120, 204, 190, 0.25);
    }

    .chat-role {
        font-size: 0.8rem;
        color: #bf7a8b !important;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .chat-content {
        font-size: 0.95rem;
        line-height: 1.45;
        color: inherit !important;
    }
    .chat-content strong {
        color: inherit !important;
        font-weight: 750;
    }

    .chat-title {
        font-size: 1.6rem;
        font-weight: 680;
        color: #b26a7c !important;
        margin-top: 0.2rem;
        margin-bottom: 0.75rem;
    }

    /* Textarea: dark slate bar, light text */
    textarea {
        border-radius: 14px !important;
        border: 1px solid #4c4f59 !important;
        box-shadow: 0 0 0 1px rgba(60, 63, 75, 0.4);
        background-color: #2f3136 !important;
        color: #f4f4f7 !important;
    }
    textarea::placeholder {
        color: #b9bac4 !important;
        opacity: 1;
    }
    textarea:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255, 77, 166, 0.5) !important;
    }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        background-color: #ffe1ef;
        color: #ff4da6 !important;
        font-size: 0.8rem;
        font-weight: 650;
        margin-left: 0.4rem;
        vertical-align: middle;
    }

    .about-text {
        color: #b26a7c !important;
        line-height: 1.6;
        font-size: 0.96rem;
    }

    .prompt-label {
        font-size: 1.0rem;
        font-weight: 600;
        color: #a3566c !important;
        margin-bottom: 0.25rem;
    }

    [data-testid="stExpander"] > details > summary {
        background-color: #fff7fb !important;
        color: #b26a7c !important;
        border-radius: 10px !important;
        border: 1px solid #f3c4d9 !important;
    }

    [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
    }

    [data-testid="stExpander"] svg path {
        fill: #b26a7c !important;
    }

    [data-testid="stExpander"] button {
        background-color: #fff7fb !important;
        color: #b26a7c !important;
        border-radius: 10px !important;
        border: 1px solid #f3c4d9 !important;
    }

    .score-pill {
        display: inline-block;
        padding: 0.10rem 0.55rem;
        border-radius: 999px;
        background-color: #ffe1ef;
        color: #d93f88 !important;
        font-weight: 700;
        border: 1px solid #ffb3d2;
    }

    /* ---------------------------------------------------------
       FIX: expander header text was rendering white-on-light-pink
       (unreadable). This happens because Streamlit's expander
       label uses a CSS variable that resolves to white under
       OS/browser dark mode, independent of our page background.
       We force a strong dark pink here as a CSS-level backstop
       (the primary fix is locking the Streamlit theme itself via
       .streamlit/config.toml). Button text inside the expander is
       excluded so the white-on-magenta sample-question buttons
       are untouched.
       --------------------------------------------------------- */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] > div > p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #99004d !important;
        font-weight: 650 !important;
    }

    [data-testid="stExpander"] .stButton > button,
    [data-testid="stExpander"] .stButton > button *,
    [data-testid="stExpander"] [data-testid="stButton"] button,
    [data-testid="stExpander"] [data-testid="stButton"] button * {
        color: #99004d !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: MONA PORTRAIT ----------
_m_l, _m_c, _m_r = st.sidebar.columns([1, 3, 1])
with _m_c:
    st.image("mona.png", use_container_width=True)
st.sidebar.markdown(
    "<div style='text-align:center; color:#a34e68; font-weight:650; "
    "font-size:1.05rem; margin-top:-4px; margin-bottom:6px;'></div>",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: LANGUAGE FIRST ----------
_LANG_OPTIONS = ["en", "fr"]
language = st.sidebar.selectbox(
    "Language / Langue",
    options=_LANG_OPTIONS,
    index=(_LANG_OPTIONS.index(ARGS.language) if ARGS.language in _LANG_OPTIONS else 0),
    format_func=lambda x: "English" if x == "en" else "Français",
)

# ---------- LANGUAGE-SPECIFIC TEXT ----------
if language == "fr":
    subtitle_text = "Prototype de chatbot de soutien pour le cancer du sein"
    expander_label = "À propos d’hormonAI & sécurité"
    about_md = """
<div class="about-text">

**Qu’est-ce que hormonAI ?**

- hormonAI est un *prototype* de chatbot construit à partir d’une base de connaissances (articles et FAQ) sur l’hormonothérapie adjuvante du cancer du sein.
- Il utilise une approche de recherche augmentée par génération (RAG) : pour chaque question, il cherche dans la base de connaissances et s’appuie sur les entrées les plus pertinentes.

**Points de sécurité très importants**

- hormonAI ne remplace **en aucun cas** votre oncologue, votre médecin traitant ou votre équipe soignante.
- Ce n’est **pas** un service d’urgence et il ne fournit pas de conseils médicaux personnalisés.
- Il ne doit jamais être utilisé pour décider de commencer, arrêter ou modifier un traitement.
- hormonAI est limité au contenu de sa base de connaissances et peut répondre « Je ne sais pas » lorsque la question dépasse ce cadre.

Discutez toujours de votre situation et de toute décision thérapeutique directement avec votre équipe d’oncologie.

</div>
    """
    sidebar_header = "Paramètres"
    sidebar_reminder = (
        "⚠️ hormonAI ne remplace pas votre équipe d’oncologie et ne peut pas "
        "donner de recommandations personnalisées sur les traitements."
    )
    chat_title_label = "Discuter avec Mona"
    placeholder = "Par exemple : « Les bouffées de chaleur sont-elles fréquentes ? »"
    prompt_label = "Posez votre question sur l’hormonothérapie adjuvante…"
    use_llm_label = "Utiliser un LLM pour reformuler (avancé)"
    use_llm_help = (
        "Si désactivé, hormonAI répond directement avec le texte source.\n"
        "Si activé, le LLM reformule uniquement lorsque la réponse est trouvée dans la base de connaissances."
    )
    show_sources_label = "Afficher les sources pour chaque réponse"
    use_rerank_label = "Activer le re-ranking (plus lent, parfois plus précis)"
    use_rerank_help = (
        "Utilise un modèle de re-ranking (CrossEncoder) pour réordonner les passages récupérés. "
        "Cela peut améliorer la pertinence, mais c’est plus lent et demande des dépendances supplémentaires."
    )
    sample_prompts_label = "Vous ne savez pas comment formuler votre question ? Essayez l’une de celles-ci :"
else:
    subtitle_text = "A breast cancer support chatbot prototype"
    expander_label = "About hormonAI & safety"
    about_md = """
<div class="about-text">

**What is hormonAI?**

- hormonAI is a *proof-of-concept* chatbot built on a knowledge base (articles and FAQs) about adjuvant hormone therapy for breast cancer.
- It uses retrieval-augmented generation (RAG): for each question, it searches the knowledge base and bases its answer on the most relevant entries.

**Very important safety notes**

- hormonAI does **not** replace your oncologist, GP, or healthcare team.
- It is **not** an emergency service and does not provide personalized medical advice.
- It should never be used to decide whether to start, stop, or change a treatment.
- hormonAI is restricted to its knowledge base and may say “I don’t know” when a question goes beyond that scope.

Always discuss your situation and any treatment decisions directly with your oncology team.

</div>
    """
    sidebar_header = "Settings"
    sidebar_reminder = (
        "⚠️ hormonAI does not replace your oncology team and cannot give "
        "individual treatment recommendations."
    )
    chat_title_label = "Chat with Mona"
    placeholder = 'For example: "Is sun exposure contraindicated while taking tamoxifen?"'
    prompt_label = "Ask your question about adjuvant hormone therapy…"
    use_llm_label = "Use LLM for rephrasing (advanced)"
    use_llm_help = (
        "If disabled, hormonAI answers directly with the source text.\n"
        "If enabled, the LLM only rephrases when an answer is found in the knowledge base."
    )
    show_sources_label = "Show sources for each answer"
    use_rerank_label = "Enable reranking (slower, sometimes more accurate)"
    use_rerank_help = (
        "Uses a CrossEncoder reranker to reorder retrieved entries. "
        "This can improve relevance, but it’s slower and requires extra dependencies."
    )
    sample_prompts_label = "Not sure how to phrase your question? Try one of these:"

# ---------- LOGO + TITLE ----------
with st.container():
    st.markdown('<div class="hormonai-header">', unsafe_allow_html=True)
    st.image("hormonAI.png", width=440)
    st.markdown(
        f"""
        <div class="hormonai-title">
            hormonAI <span class="pill">prototype</span>
        </div>
        <div class="hormonai-subtitle">{subtitle_text}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- ABOUT & SAFETY SECTION ----------
with st.expander(expander_label, expanded=True):
    st.markdown(about_md, unsafe_allow_html=True)

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header(sidebar_header)

use_llm = st.sidebar.checkbox(
    use_llm_label,
    value=bool(ARGS.use_llm),
    help=use_llm_help,
)

use_rerank = st.sidebar.checkbox(
    use_rerank_label,
    value=bool(ARGS.rerank),
    help=use_rerank_help,
)

show_sources = st.sidebar.checkbox(
    show_sources_label,
    value=True,
    help="Display the source entries/sections used for the answer.",
)

st.sidebar.markdown("---")
st.sidebar.caption(sidebar_reminder)

# ---------- RETRIEVER CACHING ----------
@st.cache_resource
def load_shared_retriever(rerank: bool) -> HybridFAQRetriever:
    # Combined multilingual index (faq_all_*): one shared embedding space,
    # same-language preferred, cross-lingual fallback. Query language is set
    # per request below.
    r = HybridFAQRetriever(language="en", rerank=rerank, shared=True,
                           data_dir=DEFAULT_DATA_DIR, top_k=DEFAULT_TOP_K,
                           embedding_model=DEFAULT_EMBEDDING_MODEL,
                           rerank_model=DEFAULT_RERANK_MODEL,
                           rerank_top_n=DEFAULT_RERANK_TOP_N)
    r.load()
    return r

@st.cache_resource
def load_perlang_retriever(lang: str, rerank: bool) -> HybridFAQRetriever:
    r = HybridFAQRetriever(language=lang, rerank=rerank,
                           data_dir=DEFAULT_DATA_DIR, top_k=DEFAULT_TOP_K,
                           embedding_model=DEFAULT_EMBEDDING_MODEL,
                           rerank_model=DEFAULT_RERANK_MODEL,
                           rerank_top_n=DEFAULT_RERANK_TOP_N)
    r.load()
    return r

retriever = None
retrieval_mode = "shared"
try:
    retriever = load_shared_retriever(use_rerank)
    retriever.language = language  # active query language on the shared corpus
except Exception:
    # faq_all_* not built (or unreadable): fall back to per-language indexes.
    retrieval_mode = "per-language"
    try:
        retriever = load_perlang_retriever(language, use_rerank)
    except Exception as e:
        st.error(f"Error loading knowledge base for language '{language}': {e}")
        st.stop()

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_language" not in st.session_state:
    st.session_state.last_language = language

if "last_rerank" not in st.session_state:
    st.session_state.last_rerank = use_rerank

if language != st.session_state.last_language or use_rerank != st.session_state.last_rerank:
    st.session_state.history = []
    st.session_state.last_language = language
    st.session_state.last_rerank = use_rerank

# ---------- HANDLE SEND ----------
@st.cache_resource
def _get_audit_logger(path: str) -> AuditLogger:
    return AuditLogger(path)


# ---------- HANDLE SEND ----------
def _coerce_answer_result(res: object) -> tuple[bool, str]:
    if isinstance(res, dict):
        answered = bool(res.get("answered", False))
        answer_text = str(res.get("answer_text", "")).strip()
        return answered, answer_text
    answered = bool(getattr(res, "answered", False))
    answer_text = str(getattr(res, "answer_text", "")).strip()
    return answered, answer_text


def handle_send():
    user_input_val = st.session_state.get("user_input", "").strip()
    if not user_input_val:
        return

    st.session_state.history.append({"role": "user", "content": user_input_val, "sources": None})

    res = None
    try:
        res = answer_query(
            retriever=retriever,
            user_query=user_input_val,
            use_llm=use_llm,
            llm_model=DEFAULT_LLM_MODEL,
            debug=False,
            sem_accept_threshold=SEM_ACCEPT_THRESHOLD,
            rerank_accept_threshold=RERANK_ACCEPT_THRESHOLD,
            dense_floor=DENSE_FLOOR,
        )
        answered, answer_text = _coerce_answer_result(res)
    except Exception as e:
        if language == "fr":
            answered, answer_text = False, (
                "J’ai rencontré une erreur technique en essayant de répondre.\n\n"
                f"Détails : `{e}`"
            )
        else:
            answered, answer_text = False, (
                "I ran into a technical error while trying to answer.\n\n"
                f"Details: `{e}`"
            )

    # Show the sources ACTUALLY cited in the answer (res.source_indices), rendered
    # by type (FAQ vs article) so article chunks are not labelled as questions.
    sources_summary = []
    if answered and res is not None:
        try:
            items = getattr(retriever, "_items", [])
            for idx in (getattr(res, "source_indices", None) or []):
                it = items[int(idx)]
                sources_summary.append(
                    {
                        "index": int(idx),
                        "source_type": str(it.get("source_type", "faq")),
                        "section": str(it.get("section", "")),
                        "question": str(it.get("question", "")),
                        "source_id": str(it.get("source_id", "")),
                        "heading_path": str(it.get("heading_path", "")),
                        "answer": str(it.get("answer", "")),
                    }
                )
        except Exception:
            sources_summary = []

    # PHI-free latency logging: timing + language + answered only, never the query.
    if AUDIT_LATENCY and res is not None and getattr(res, "timing_ms", None):
        try:
            _get_audit_logger(AUDIT_LOG_PATH).log(
                {
                    "type": "latency",
                    "language": language,
                    "answered": bool(getattr(res, "answered", False)),
                    "used_llm": bool(use_llm and getattr(res, "answered", False)),
                    "timing_ms": {
                        k: (v if k == "n_reranked" else round(float(v), 1))
                        for k, v in res.timing_ms.items()
                        if k == "n_reranked" or isinstance(v, (int, float))
                    },
                }
            )
        except Exception:
            pass

    timing_str = None
    if SHOW_TIMING and res is not None and getattr(res, "timing_ms", None):
        t = res.timing_ms or {}
        if t:
            timing_str = (
                "retrieve {:.0f} ms | rerank {:.0f} ms (n={}) | "
                "llm {:.0f} ms | total {:.0f} ms"
            ).format(
                t.get("retrieve", 0.0), t.get("rerank", 0.0), t.get("n_reranked", 0),
                t.get("llm", 0.0), t.get("total", 0.0),
            )

    st.session_state.history.append(
        {
            "role": "bot",
            "content": answer_text,
            "sources": (sources_summary if answered else []),
            "timing": timing_str,
        }
    )

    st.session_state.user_input = ""


def handle_sample_prompt_click(prompt_text: str):
    """Place the sample prompt into the input box for the user to review/edit — does not submit."""
    st.session_state.user_input = prompt_text


# ---------- SAMPLE PROMPTS ----------
prompts_for_lang = SAMPLE_PROMPTS.get(language, SAMPLE_PROMPTS["en"])

with st.expander(sample_prompts_label):
    for theme, prompts in prompts_for_lang.items():
        st.markdown(f"**{theme}**")
        for i, prompt_text in enumerate(prompts):
            st.button(
                prompt_text,
                key=f"sample_prompt_{language}_{theme}_{i}",
                use_container_width=True,
                on_click=handle_sample_prompt_click,
                args=(prompt_text,),
            )

# ---------- CHAT DISPLAY (kept immediately above the input box) ----------
st.markdown(f'<div class="chat-title">{chat_title_label}</div>', unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            safe_text = render_bubble_text(msg["content"])
            user_html = f"""
            <div class="chat-bubble-user">
                <div class="chat-role">{'Vous' if language == 'fr' else 'You'}</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """
            st.markdown(user_html, unsafe_allow_html=True)

        else:
            safe_text = render_bubble_text(msg["content"])
            bot_html = f"""
            <div class="chat-bubble-bot">
                <div class="chat-role">Mona</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """
            st.markdown(bot_html, unsafe_allow_html=True)

            if SHOW_TIMING and msg.get("timing"):
                st.caption(f"⏱ {msg['timing']}")

            # ✅ Only show dropdown when we actually answered AND sources exist
            if show_sources and msg.get("sources"):
                exp_label = (
                    "Sources utilisées pour cette réponse"
                    if language == "fr"
                    else "Sources used for this answer"
                )
                with st.expander(exp_label):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Source {i}**")
                        if src.get("source_type", "faq") == "faq":
                            st.markdown(f"- **Question:** {src['question']}")
                            st.markdown(f"- **Section:** {src['section']}")
                        else:
                            doc = src.get("source_id") or src.get("section") or ""
                            hp = src.get("heading_path") or ""
                            cite = f"{doc} — {hp}" if (hp and hp != doc) else doc
                            st.markdown(f"- **Document:** {cite}")
                        snippet = src["answer"]
                        if len(snippet) > 350:
                            snippet = snippet[:350] + "…"
                        st.markdown(f"- **Excerpt:** {snippet}")

# ---------- USER INPUT ----------
st.markdown(f'<div class="prompt-label">{prompt_label}</div>', unsafe_allow_html=True)

st.text_area(
    "",
    key="user_input",
    height=160,
    placeholder=placeholder,
)

col1, col2 = st.columns([1, 4])
with col1:
    st.button("Send" if language == "en" else "Envoyer", on_click=handle_send)
