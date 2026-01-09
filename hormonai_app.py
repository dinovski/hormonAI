import os
import html
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
        background-color: #ffffff;  /* white main area */
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffeef6;  /* warm light pink */
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

    /* Checkbox/toggle labels in sidebar explicitly */
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

    /* Streamlit selectbox: BaseWeb select container (control area) */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffdde9 !important;  /* light pink, not black */
        border-radius: 10px !important;
        border: 1px solid #ff9ccc !important;
    }

    /* Streamlit selectbox: value/button text */
    [data-testid="stSidebar"] .stSelectbox div[role="button"] {
        background-color: transparent !important;
        color: #b26a7c !important;
    }

    /* Any text inside selectbox in sidebar */
    [data-testid="stSidebar"] .stSelectbox * {
        color: #b26a7c !important;
    }

    /* Dropdown list options when open (anywhere) */
    div[role="listbox"] * {
        color: #b26a7c !important;
        background-color: #fff9fd !important;
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
        background-color: #d93f88 !important;          /* tasteful dark pink */
        background-image: none !important;
        color: #ffffff !important;                     /* white text */
        box-shadow: 0 3px 8px rgba(217, 63, 136, 0.35);
    }

    /* Ensure inner span text stays white despite global * selector */
    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        opacity: 0.96;
        box-shadow: 0 4px 10px rgba(217, 63, 136, 0.45);
    }

    /* Chat message bubbles – rounded rectangles with text inside */
    .chat-bubble-user {
        background-color: #ffe1ef;                 /* light pink */
        padding: 0.85rem 1.1rem;
        border-radius: 18px;                        /* fully rounded rectangle */
        margin: 0.3rem 0 0.6rem 0;
        max-width: 75%;
        margin-left: auto;
        border: 1px solid #ffb3d2;
        color: #b26a7c !important;
        box-shadow: 0 2px 6px rgba(255, 154, 203, 0.25);
    }
    .chat-bubble-bot {
        background-color: #e6faf7;                 /* light teal */
        padding: 0.85rem 1.1rem;
        border-radius: 18px;                        /* fully rounded rectangle */
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

    /* Chat title */
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
        background-color: #2f3136 !important;  /* dark slate gray */
        color: #f4f4f7 !important;             /* light text on dark bar */
    }

    textarea::placeholder {
        color: #b9bac4 !important;  /* soft light gray */
        opacity: 1;
    }

    textarea:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255, 77, 166, 0.5) !important;
    }

    /* Small pill tag for "prototype" */
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

    /* About text readability */
    .about-text {
        color: #b26a7c !important;
        line-height: 1.6;
        font-size: 0.96rem;
    }

    /* Prompt label */
    .prompt-label {
        font-size: 1.0rem;
        font-weight: 600;
        color: #a3566c !important;
        margin-bottom: 0.25rem;
    }

    /* Expander header */
    [data-testid="stExpander"] > details > summary {
        background-color: #fff7fb !important;  /* soft very light pink/white */
        color: #b26a7c !important;
        border-radius: 10px !important;
        border: 1px solid #f3c4d9 !important;
    }

    /* Chevron/arrow icon in expander */
    [data-testid="stExpander"] svg path {
        fill: #b26a7c !important;
    }

    /* Also cover button-based expander header (for older/newer variants) */
    [data-testid="stExpander"] button {
        background-color: #fff7fb !important;
        color: #b26a7c !important;
        border-radius: 10px !important;
        border: 1px solid #f3c4d9 !important;
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
    llm_provider_label = "Fournisseur LLM"
    show_sources_label = "Afficher les sources de la FAQ pour chaque réponse"
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
    llm_provider_label = "LLM provider"
    show_sources_label = "Show FAQ sources for each answer"

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

llm_provider = st.sidebar.selectbox(
    llm_provider_label,
    options=["openai", "ollama"],
    index=1,  # default to Ollama
)

openai_model = st.sidebar.text_input(
    "OpenAI model",
    value="gpt-4o-mini",
    help="Used only if provider = openai.",
)

ollama_model = st.sidebar.text_input(
    "Ollama model",
    value="llama3.2",
    help="Used only if provider = ollama.",
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
def load_retriever(lang: str) -> HybridFAQRetriever:
    """
    Loads the hybrid retriever artifacts produced by ingest_faq.py:
      faq_<lang>_index_q.faiss
      faq_<lang>_index_qa.faiss
      faq_<lang>_qa.pkl
      faq_<lang>_bm25.pkl
    """
    r = HybridFAQRetriever(language=lang)
    # if your HybridFAQRetriever requires explicit loading:
    if hasattr(r, "load") and callable(getattr(r, "load")):
        r.load()
    return r


try:
    retriever = load_retriever(language)
except Exception as e:
    st.error(f"Error loading FAQ data for language '{language}': {e}")
    st.stop()

# ---------- SESSION STATE ----------
# Each history item: {"role": "user"|"bot", "content": str, "sources": Optional[list]}
if "history" not in st.session_state:
    st.session_state.history = []

if "last_language" not in st.session_state:
    st.session_state.last_language = language

# Clear history when language changes so we don't mix EN/FR content
if language != st.session_state.last_language:
    st.session_state.history = []
    st.session_state.last_language = language

# ---------- CHAT DISPLAY ----------
st.markdown(f'<div class="chat-title">{chat_title_label}</div>', unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        safe_text = html.escape(msg["content"]).replace("\n", "<br>")
        if msg["role"] == "user":
            user_html = f"""
            <div class="chat-bubble-user">
                <div class="chat-role">{'Vous' if language == 'fr' else 'You'}</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """
            st.markdown(user_html, unsafe_allow_html=True)
        else:
            bot_label = "hormonAI"
            bot_html = f"""
            <div class="chat-bubble-bot">
                <div class="chat-role">{bot_label}</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """
            st.markdown(bot_html, unsafe_allow_html=True)

            # Optional sources panel
            if show_sources and msg.get("sources"):
                exp_label = (
                    "Sources de la FAQ utilisées pour cette réponse"
                    if language == "fr"
                    else "FAQ sources used for this answer"
                )
                with st.expander(exp_label):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Source {i}** – relevance: `{src['score']:.3f}`")
                        st.markdown(f"- **Section:** {src['section']}")
                        st.markdown(f"- **Question:** {src['question']}")
                        snippet = src["answer"]
                        if len(snippet) > 350:
                            snippet = snippet[:350] + "…"
                        st.markdown(f"- **Answer snippet:** {snippet}")

# ---------- HANDLE SEND (callback) ----------
def _coerce_answer_result(res: object) -> tuple[bool, str]:
    """
    Supports both dict and dataclass-like returns from rag_core.answer_query.
    """
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

    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input_val, "sources": None})

    # Retrieve candidates for sources display (top 3)
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

    # Core answer (LLM only rephrases if answered=True inside rag_core)
    try:
        res = answer_query(
            retriever=retriever,
            user_query=user_input_val,
            use_llm=use_llm,
            llm_provider=llm_provider,
            openai_model=openai_model,
            ollama_model=ollama_model,
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

    # Only attach sources if we actually answered (prevents misleading “sources” on abstain)
    st.session_state.history.append(
        {
            "role": "bot",
            "content": answer_text,
            "sources": (sources_summary if answered else []),
        }
    )

    # Clear the input box after sending (safe inside callback)
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
