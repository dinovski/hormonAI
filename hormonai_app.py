import os
import html
import streamlit as st

from rag_core import (
    HybridFAQRetriever,
    choose_best_candidate,
    format_faq_answer,
    answer_with_llm,
    DEFAULT_EMBEDDING_MODEL,
)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="hormonAI",
    page_icon="💗",
    layout="centered",
)

# ----------------------------
# Global CSS (warm palette, no black except prompt bar)
# ----------------------------
st.markdown(
    """
<style>
/* Warm global text color (no black) */
* { color: #9b5a6f !important; }

/* App background */
.stApp { background-color: #ffffff; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #ffeaf3 !important;  /* warm pink */
    border-right: 1px solid #f5c8dc;
}

/* Sidebar header text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #8d3f5b !important;
    font-weight: 750;
}

/* Sidebar body text */
section[data-testid="stSidebar"] * { color: #9b5a6f !important; }

/* Checkbox/radio accent colors */
input[type="checkbox"], input[type="radio"] { accent-color: #d93f88; }

/* --- Selectbox / dropdowns (prevent black backgrounds) --- */
/* Container */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffe0ee !important;   /* light pink */
    border: 1px solid #f3a3c7 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
/* The clickable button area */
section[data-testid="stSidebar"] div[role="button"] {
    background-color: transparent !important;
    border-radius: 12px !important;
}
/* Dropdown menu */
div[role="listbox"] {
    background-color: #fff7fb !important;
    border: 1px solid #f3a3c7 !important;
    border-radius: 12px !important;
}
/* Options */
div[role="option"] {
    background-color: #fff7fb !important;
}
div[role="option"]:hover {
    background-color: #ffe0ee !important;
}

/* Text inputs in sidebar */
section[data-testid="stSidebar"] input {
    background-color: #fff7fb !important;
    border: 1px solid #f3a3c7 !important;
    border-radius: 12px !important;
    color: #9b5a6f !important;
}

/* Header block */
.hormonai-header { text-align: center; margin-top: 0.25rem; margin-bottom: 0.75rem; }
.hormonai-title { font-size: 2.7rem; font-weight: 800; color: #9b5a6f !important; margin: 0.25rem 0 0.25rem 0; }
.hormonai-subtitle { font-size: 1.05rem; color: #ab6a7f !important; margin: 0 0 0.75rem 0; }

.pill {
    display: inline-block;
    padding: 0.18rem 0.75rem;
    border-radius: 999px;
    background-color: #ffe0ee;
    color: #d93f88 !important;
    font-size: 0.82rem;
    font-weight: 750;
    margin-left: 0.45rem;
    vertical-align: middle;
}

/* Expander header (avoid black) */
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] button {
    background-color: #fff7fb !important;
    border: 1px solid #f3c4d9 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] svg path { fill: #9b5a6f !important; }

/* About text readability */
.about-text { color: #8d3f5b !important; line-height: 1.65; font-size: 0.98rem; }
.about-text b, .about-text strong { color: #8d3f5b !important; }

/* Chat title */
.chat-title { font-size: 1.65rem; font-weight: 750; color: #8d3f5b !important; margin: 0.25rem 0 0.75rem 0; }

/* Chat bubbles with text inside */
.chat-bubble-user {
    background-color: #ffe1ef;   /* light pink */
    padding: 0.95rem 1.15rem;
    border-radius: 18px;
    margin: 0.25rem 0 0.65rem 0;
    max-width: 78%;
    margin-left: auto;
    border: 1px solid #ffb3d2;
    box-shadow: 0 2px 6px rgba(255, 154, 203, 0.23);
}
.chat-bubble-bot {
    background-color: #e6faf7;   /* light teal */
    padding: 0.95rem 1.15rem;
    border-radius: 18px;
    margin: 0.25rem 0 0.65rem 0;
    max-width: 78%;
    margin-right: auto;
    border: 1px solid #8ad5c8;
    box-shadow: 0 2px 6px rgba(120, 204, 190, 0.23);
}
.chat-role {
    font-size: 0.78rem;
    color: #ab6a7f !important;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
}
.chat-content {
    font-size: 0.98rem;
    line-height: 1.5;
    color: inherit !important;
}

/* Prompt label */
.prompt-label { font-size: 1.02rem; font-weight: 750; color: #8d3f5b !important; margin: 0.75rem 0 0.25rem 0; }

/* Prompt bar (dark slate) – allowed as the only dark element */
textarea {
    border-radius: 14px !important;
    border: 1px solid #565a66 !important;
    background-color: #2f3136 !important;
    color: #f3f4f7 !important;
    box-shadow: 0 0 0 1px rgba(86, 90, 102, 0.35) !important;
}
textarea::placeholder { color: #b9bac4 !important; opacity: 1; }
textarea:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(217, 63, 136, 0.45) !important;
}

/* Send button */
/* ===== FORCE FORM SUBMIT BUTTON (Send/Envoyer) TO LOGO PINK ===== */

/* Target BOTH regular buttons and form submit buttons */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stButton"] button:focus,
[data-testid="stFormSubmitButton"] button:focus,
[data-testid="stButton"] button:active,
[data-testid="stFormSubmitButton"] button:active {
    background: #F84880 !important;            /* hormonAI logo pink */
    background-color: #F84880 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 750 !important;
    padding: 0.62rem 1.55rem !important;
    box-shadow: 0 3px 8px rgba(248, 72, 128, 0.35) !important;
}

/* Kill Streamlit’s inner dark layers */
[data-testid="stButton"] button *,
[data-testid="stFormSubmitButton"] button * {
    color: #ffffff !important;
    background: transparent !important;
}

/* Hover color slightly lighter */
[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #F65A8C !important;
    background-color: #F65A8C !important;
}


/* Ensure markdown links not black */
a { color: #b85c7c !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar: language + settings
# ----------------------------
language = st.sidebar.selectbox(
    "Language / Langue",
    options=["en", "fr"],
    format_func=lambda x: "English" if x == "en" else "Français",
)

TEXT = {
    "en": {
        "subtitle": "A breast cancer support chatbot prototype",
        "about_title": "About hormonAI & safety",
        "about_html": """
<div class="about-text">
<b>What is hormonAI?</b><br>
• hormonAI is a proof-of-concept chatbot built from a written FAQ about adjuvant hormone therapy.<br>
• It uses retrieval-augmented generation (RAG): it searches the FAQ and answers only from matching entries.<br><br>

<b>Very important safety notes</b><br>
• hormonAI does <b>not</b> replace your oncologist, GP, or care team.<br>
• It is <b>not</b> an emergency service and it does not provide personalized medical advice.<br>
• Do not use it to decide whether to start/stop/change treatment.<br>
• If your question goes beyond the FAQ, hormonAI may say it can’t answer reliably.<br><br>

Always discuss your situation and any treatment decisions with your oncology team.
</div>
""",
        "settings": "Settings",
        "sidebar_warning": "⚠️ hormonAI does not replace your oncology team and cannot give individual treatment recommendations.",
        "chat_title": "Chat with hormonAI",
        "prompt_label": "Ask your question about adjuvant hormone therapy…",
        "placeholder": 'For example: "Does pregnancy increase the risk of recurrence?"',
        "use_llm": "Use LLM for rephrasing (advanced)",
        "use_llm_help": "If off: answer uses the FAQ text directly. If on: the LLM rewrites using only the FAQ context.",
        "provider": "LLM provider",
        "openai_model": "OpenAI model",
        "ollama_model": "Ollama model",
        "rerank": "Improve relevance (reranking)",
        "rerank_help": "Reranking can help the best matching FAQ entry rise to the top.",
        "show_sources": "Show FAQ sources for each answer",
        "send": "Send",
        "you": "You",
        "sources_label": "FAQ sources used for this answer",
        "no_match": "I’m not finding a sufficiently relevant FAQ entry to answer reliably. Could you rephrase, or discuss this with your oncology team?",
    },
    "fr": {
        "subtitle": "Prototype de chatbot de soutien pour le cancer du sein",
        "about_title": "À propos d’hormonAI & sécurité",
        "about_html": """
<div class="about-text">
<b>Qu’est-ce que hormonAI ?</b><br>
• hormonAI est un prototype de chatbot basé sur une FAQ écrite sur l’hormonothérapie adjuvante.<br>
• Il utilise une approche RAG : il recherche dans la FAQ et répond uniquement à partir d’entrées correspondantes.<br><br>

<b>Points de sécurité très importants</b><br>
• hormonAI ne remplace <b>en aucun cas</b> votre oncologue, médecin ou équipe soignante.<br>
• Ce n’est <b>pas</b> un service d’urgence et il ne donne pas de conseils médicaux personnalisés.<br>
• Ne l’utilisez pas pour décider de commencer/arrêter/modifier un traitement.<br>
• Si la question dépasse la FAQ, hormonAI peut indiquer qu’il ne peut pas répondre de façon fiable.<br><br>

Parlez toujours de votre situation et de toute décision thérapeutique avec votre équipe d’oncologie.
</div>
""",
        "settings": "Paramètres",
        "sidebar_warning": "⚠️ hormonAI ne remplace pas votre équipe d’oncologie et ne peut pas donner de recommandations personnalisées.",
        "chat_title": "Discuter avec hormonAI",
        "prompt_label": "Posez votre question sur l’hormonothérapie adjuvante…",
        "placeholder": "Par exemple : « La grossesse augmente-t-elle le risque de récidive ? »",
        "use_llm": "Utiliser un LLM pour reformuler (avancé)",
        "use_llm_help": "Si désactivé : réponse = texte de la FAQ. Si activé : le LLM reformule en utilisant uniquement le contexte de la FAQ.",
        "provider": "Fournisseur LLM",
        "openai_model": "Modèle OpenAI",
        "ollama_model": "Modèle Ollama",
        "rerank": "Améliorer la pertinence (reranking)",
        "rerank_help": "Le reranking peut aider la meilleure entrée de la FAQ à remonter.",
        "show_sources": "Afficher les sources de la FAQ pour chaque réponse",
        "send": "Envoyer",
        "you": "Vous",
        "sources_label": "Sources de la FAQ utilisées pour cette réponse",
        "no_match": "Je ne trouve pas d’information suffisamment pertinente dans la FAQ pour répondre de façon fiable. Pouvez-vous reformuler, ou en parler avec votre équipe soignante ?",
    },
}

t = TEXT[language]

st.sidebar.header(t["settings"])

use_llm = st.sidebar.checkbox(t["use_llm"], value=False, help=t["use_llm_help"])
llm_provider = st.sidebar.selectbox(t["provider"], options=["ollama", "openai"], index=0)

openai_model = st.sidebar.text_input(t["openai_model"], value="gpt-4o-mini")
ollama_model = st.sidebar.text_input(t["ollama_model"], value="llama3.2")

enable_rerank = st.sidebar.checkbox(t["rerank"], value=(language == "en"), help=t["rerank_help"])
show_sources = st.sidebar.checkbox(t["show_sources"], value=True)

st.sidebar.markdown("---")
st.sidebar.caption(t["sidebar_warning"])

# ----------------------------
# Header (logo bigger)
# ----------------------------
st.markdown('<div class="hormonai-header">', unsafe_allow_html=True)
st.image("hormonAI.png", width=460)
st.markdown(
    f"""
    <div class="hormonai-title">hormonAI <span class="pill">prototype</span></div>
    <div class="hormonai-subtitle">{t["subtitle"]}</div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# About / safety
# ----------------------------
with st.expander(t["about_title"], expanded=True):
    st.markdown(t["about_html"], unsafe_allow_html=True)

# ----------------------------
# Retriever loader (hybrid artifacts)
# ----------------------------
@st.cache_resource
def load_retriever(lang: str, rerank: bool = False) -> HybridFAQRetriever:
    prefix = f"faq_{lang}"
    required = [
        f"data/{prefix}_index_q.faiss",
        f"data/{prefix}_index_qa.faiss",
        f"data/{prefix}_qa.pkl",
        f"data/{prefix}_bm25.pkl",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing hybrid retrieval files.\n"
            + "\n".join([f"- {p}" for p in missing])
            + "\n\nTip: re-run ingest_faq.py for this language."
        )

    return HybridFAQRetriever(
        prefix=prefix,
        language=lang,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        enable_rerank=rerank,
    )

try:
    retriever = load_retriever(language, enable_rerank)
except Exception as e:
    st.error(f"Error loading FAQ data for language '{language}': {e}")
    st.stop()

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_lang" not in st.session_state:
    st.session_state.last_lang = language

if "last_rerank" not in st.session_state:
    st.session_state.last_rerank = enable_rerank

# Reset chat if language/rerank changes (prevents mixing)
if language != st.session_state.last_lang or enable_rerank != st.session_state.last_rerank:
    st.session_state.history = []
    st.session_state.last_lang = language
    st.session_state.last_rerank = enable_rerank

# ----------------------------
# Chat display
# ----------------------------
st.markdown(f'<div class="chat-title">{t["chat_title"]}</div>', unsafe_allow_html=True)

for msg in st.session_state.history:
    safe_text = html.escape(msg["content"]).replace("\n", "<br>")

    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div class="chat-role">{t["you"]}</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-bubble-bot">
                <div class="chat-role">hormonAI</div>
                <div class="chat-content">{safe_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_sources and msg.get("sources"):
            with st.expander(t["sources_label"], expanded=False):
                for i, src in enumerate(msg["sources"], start=1):
                    score = float(src.get("score", 0.0))
                    st.markdown(f"**Source {i}** – score: `{score:.4f}`")
                    st.markdown(f"- **Section:** {src.get('section','')}")
                    st.markdown(f"- **Question:** {src.get('question','')}")
                    snippet = src.get("answer", "")
                    if len(snippet) > 350:
                        snippet = snippet[:350] + "…"
                    st.markdown(f"- **Answer snippet:** {snippet}")

# ----------------------------
# Prompt + send (use st.form to clear reliably)
# ----------------------------
st.markdown(f'<div class="prompt-label">{t["prompt_label"]}</div>', unsafe_allow_html=True)

def run_chat(user_query: str) -> tuple[str, list]:
    candidates = retriever.retrieve(user_query, top_k=10, recall_k=60, debug=False)
    best = choose_best_candidate(user_query=user_query, language=language, candidates=candidates, debug=False)

    if best is None:
        return t["no_match"], []

    if use_llm:
        answer = answer_with_llm(
            language=language,
            provider=llm_provider,
            openai_model=openai_model,
            ollama_model=ollama_model,
            user_query=user_query,
            top=best,
        )
    else:
        answer = format_faq_answer(best, language)

    sources_summary = []
    for r in candidates[:3]:
        score = r.get("rerank_score") or r.get("fused_score", 0.0)
        sources_summary.append(
            {
                "index": r.get("index"),
                "score": float(score),
                "section": r.get("section", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
            }
        )
    return answer, sources_summary

with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_area("", height=180, placeholder=t["placeholder"])
    submitted = st.form_submit_button(t["send"])

if submitted and user_text.strip():
    q = user_text.strip()

    st.session_state.history.append({"role": "user", "content": q, "sources": None})

    try:
        answer, sources = run_chat(q)
    except Exception as e:
        # Keep errors warm-toned (no black), but show message
        answer = (f"Technical error: {e}" if language == "en" else f"Erreur technique : {e}")
        sources = []

    st.session_state.history.append({"role": "bot", "content": answer, "sources": sources})

    # Rerun to show the new message immediately
    st.rerun()
