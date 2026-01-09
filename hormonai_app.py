import os
import html
import re
import streamlit as st

from rag_core import (
    HybridFAQRetriever,
    answer_query,
)

# ---------- BASIC PAGE CONFIG ----------
st.set_page_config(
    page_title="hormonAI – Breast Cancer Support Chatbot",
    page_icon="💗",
    layout="centered",
)

DEFAULT_LLM_MODEL = os.getenv("HORMONAI_LLM_MODEL", "llama3.2")


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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: LANGUAGE FIRST ----------
language = st.sidebar.selectbox(
    "Language / Langue",
    options=["en", "fr"],
    format_func=lambda x: "English" if x == "en" else "Français",
)

# ---------- LANGUAGE-SPECIFIC TEXT ----------
if language == "fr":
    subtitle_text = "Prototype de chatbot de soutien pour le cancer du sein"
    expander_label = "À propos d’hormonAI & sécurité"
    about_md = """
<div class="about-text">

**Qu’est-ce que hormonAI ?**

- hormonAI est un *prototype* de chatbot construit à partir d’une FAQ écrite sur l’hormonothérapie adjuvante du cancer du sein.
- Il utilise une approche de recherche augmentée par génération (RAG) : pour chaque question, il cherche dans la FAQ et s’appuie sur les entrées les plus pertinentes.

**Points de sécurité très importants**

- hormonAI ne remplace **en aucun cas** votre oncologue, votre médecin traitant ou votre équipe soignante.
- Ce n’est **pas** un service d’urgence et il ne fournit pas de conseils médicaux personnalisés.
- Il ne doit jamais être utilisé pour décider de commencer, arrêter ou modifier un traitement.
- hormonAI est limité au contenu de la FAQ et peut répondre « Je ne sais pas » lorsque la question dépasse ce cadre.

Discutez toujours de votre situation et de toute décision thérapeutique directement avec votre équipe d’oncologie.

</div>
    """
    sidebar_header = "Paramètres"
    sidebar_reminder = (
        "⚠️ hormonAI ne remplace pas votre équipe d’oncologie et ne peut pas "
        "donner de recommandations personnalisées sur les traitements."
    )
    chat_title_label = "Discuter avec hormonAI"
    placeholder = "Par exemple : « Les bouffées de chaleur sont-elles fréquentes ? »"
    prompt_label = "Posez votre question sur l’hormonothérapie adjuvante…"
    use_llm_label = "Utiliser un LLM pour reformuler (avancé)"
    use_llm_help = (
        "Si désactivé, hormonAI répond directement avec le texte de la FAQ.\n"
        "Si activé, le LLM reformule uniquement lorsque la réponse est trouvée dans la FAQ."
    )
    show_sources_label = "Afficher les sources de la FAQ pour chaque réponse"
    use_rerank_label = "Activer le re-ranking (plus lent, parfois plus précis)"
    use_rerank_help = (
        "Utilise un modèle de re-ranking (CrossEncoder) pour réordonner les passages récupérés. "
        "Cela peut améliorer la pertinence, mais c’est plus lent et demande des dépendances supplémentaires."
    )
else:
    subtitle_text = "A breast cancer support chatbot prototype"
    expander_label = "About hormonAI & safety"
    about_md = """
<div class="about-text">

**What is hormonAI?**

- hormonAI is a *proof-of-concept* chatbot built on top of a written FAQ about adjuvant hormone therapy for breast cancer.
- It uses retrieval-augmented generation (RAG): for each question, it searches the FAQ and bases its answer on the most relevant entries.

**Very important safety notes**

- hormonAI does **not** replace your oncologist, GP, or healthcare team.
- It is **not** an emergency service and does not provide personalized medical advice.
- It should never be used to decide whether to start, stop, or change a treatment.
- hormonAI is restricted to the content of the FAQ and may say “I don’t know” when a question goes beyond that scope.

Always discuss your situation and any treatment decisions directly with your oncology team.

</div>
    """
    sidebar_header = "Settings"
    sidebar_reminder = (
        "⚠️ hormonAI does not replace your oncology team and cannot give "
        "individual treatment recommendations."
    )
    chat_title_label = "Chat with hormonAI"
    placeholder = 'For example: "Is sun exposure contraindicated while taking tamoxifen?"'
    prompt_label = "Ask your question about adjuvant hormone therapy…"
    use_llm_label = "Use LLM for rephrasing (advanced)"
    use_llm_help = (
        "If disabled, hormonAI answers directly with the FAQ text.\n"
        "If enabled, the LLM only rephrases when an answer is found in the FAQ."
    )
    show_sources_label = "Show FAQ sources for each answer"
    use_rerank_label = "Enable reranking (slower, sometimes more accurate)"
    use_rerank_help = (
        "Uses a CrossEncoder reranker to reorder retrieved FAQ entries. "
        "This can improve relevance, but it’s slower and requires extra dependencies."
    )

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
    value=False,
    help=use_llm_help,
)

use_rerank = st.sidebar.checkbox(
    use_rerank_label,
    value=False,
    help=use_rerank_help,
)

show_sources = st.sidebar.checkbox(
    show_sources_label,
    value=True,
    help="Display the FAQ questions/sections that were used for the answer.",
)

st.sidebar.markdown("---")
st.sidebar.caption(sidebar_reminder)

# ---------- RETRIEVER CACHING ----------
@st.cache_resource
def load_retriever(lang: str, rerank: bool) -> HybridFAQRetriever:
    r = HybridFAQRetriever(language=lang, rerank=rerank)
    r.load()
    return r

try:
    retriever = load_retriever(language, use_rerank)
except Exception as e:
    st.error(f"Error loading FAQ data for language '{language}': {e}")
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

# ---------- CHAT DISPLAY ----------
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
                <div class="chat-role">hormonAI</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """
            st.markdown(bot_html, unsafe_allow_html=True)

            # ✅ Only show dropdown when we actually answered AND sources exist
            if show_sources and msg.get("sources"):
                exp_label = (
                    "Sources de la FAQ utilisées pour cette réponse"
                    if language == "fr"
                    else "FAQ sources used for this answer"
                )
                with st.expander(exp_label):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**Source {i}** – score: <span class='score-pill'>{src['score']:.3f}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"- **Section:** {src['section']}")
                        st.markdown(f"- **Question:** {src['question']}")
                        snippet = src["answer"]
                        if len(snippet) > 350:
                            snippet = snippet[:350] + "…"
                        st.markdown(f"- **Answer snippet:** {snippet}")

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

    # Compute sources summary (top 3) — only attach if answered=True
    sources_summary = []
    try:
        cands = retriever.retrieve(user_input_val) or []
        for c in cands[:3]:
            score = getattr(c, "rerank_score", None)
            if score is None:
                score = getattr(c, "fused_score", 0.0)
            sources_summary.append(
                {
                    "index": int(getattr(c, "index", -1)),
                    "score": float(score),
                    "section": str(getattr(c, "section", "")),
                    "question": str(getattr(c, "question", "")),
                    "answer": str(getattr(c, "answer", "")),
                }
            )
    except Exception:
        sources_summary = []

    try:
        res = answer_query(
            retriever=retriever,
            user_query=user_input_val,
            use_llm=use_llm,
            llm_model=DEFAULT_LLM_MODEL,
            debug=False,
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

    st.session_state.history.append(
        {
            "role": "bot",
            "content": answer_text,
            "sources": (sources_summary if answered else []),
        }
    )

    st.session_state.user_input = ""


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
