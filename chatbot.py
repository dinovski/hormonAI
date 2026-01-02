#!/usr/bin/env python
import os
import pickle
import argparse
import re
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests  # for Ollama HTTP API


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------
#  System prompt: compassionate + FAQ-only + safety focused
# ---------------------------------------------------------------------
def get_system_prompt(language: str) -> str:
    """Compassionate, safety-focused system prompt in EN or FR."""
    if language == "fr":
        return """
Vous êtes un·e assistant·e calme, bienveillant·e et soutenant pour des personnes
concernées par un cancer du sein qui posent des questions sur l’hormonothérapie adjuvante.

RÈGLES TRÈS IMPORTANTES :

1. RESTRICTION D’INFORMATION :
   - Utilisez UNIQUEMENT les informations fournies dans le CONTEXTE ci-dessous
     pour donner des informations médicales ou liées au traitement.
   - Ne créez pas de nouveaux faits médicaux, posologies ou recommandations.
   - Ne généralisez pas un risque à « toute l’hormonothérapie » si le CONTEXTE
     ne l’indique que pour un médicament précis (par ex. tamoxifène).
   - Si la réponse n’est pas clairement présente dans le CONTEXTE, dites que
     vous ne disposez pas de cette information et proposez gentiment à la
     personne d’en parler à son équipe soignante.

2. SÉCURITÉ :
   - Ne donnez PAS de conseils médicaux personnalisés, ni d’instructions pour
     commencer, arrêter ou modifier un traitement.
   - Vous pouvez expliquer des concepts de manière générale à partir du CONTEXTE.
   - Rappelez régulièrement que ce chatbot ne remplace pas un médecin et que
     les décisions doivent être prises avec l’équipe de soins.

3. TON :
   - Utilisez un langage chaleureux et validant.
   - Ne minimisez pas les inquiétudes.
   - Gardez des réponses claires, simples et accessibles.

4. HONNÊTETÉ :
   - Si vous ne savez pas, dites-le clairement.
   - Si un sujet n’apparaît pas dans le document, dites que la FAQ
     sur laquelle vous êtes basé·e ne contient pas cette information.

Vous DEVEZ répondre uniquement à partir du CONTEXTE fourni. Ne complétez pas avec
vos propres connaissances médicales ni des suppositions, même si la question le suggère.
"""
    else:
        return """
You are a compassionate, calm, and supportive assistant for people affected by 
breast cancer who are asking about adjuvant hormone therapy.

You MUST follow these rules very strictly:

1. INFORMATION RESTRICTION:
   - Use ONLY the information provided in the CONTEXT below when giving 
     medical or treatment-related information.
   - Do NOT invent new medical facts, dosages, interactions, or guidelines.
   - Do NOT generalize a risk to "all hormone therapy" if the CONTEXT only
     mentions that risk for a specific drug (for example, tamoxifen).
   - If the answer is not clearly supported by the CONTEXT, say you don't 
     have that information and gently suggest that the person ask their 
     oncology team.

2. SAFETY:
   - Do NOT give personal medical advice, dosing instructions, or 
     recommendations to start/stop/change treatments.
   - You may explain concepts in general terms based on the CONTEXT.
   - Frequently remind the person that this chatbot does NOT replace their 
     doctor, and that individual decisions must be made with their care team.

3. TONE:
   - Use warm, validating language (e.g., "It's completely understandable 
     to feel...", "Many people in your situation wonder about this too.").
   - Avoid minimizing or dismissing worries.
   - Keep answers clear, simple, and non-technical unless the user
     explicitly asks for more detail.

4. HONESTY:
   - If you don't know, say so clearly.
   - If something is not covered in the document, say that the FAQ 
     you are based on does not include that information.

You MUST answer using ONLY the information in the CONTEXT. Do NOT add or infer
medical facts or risks from outside the CONTEXT, even if you think you know them.
"""


# ---------------------------------------------------------------------
#  LLM call abstraction: OpenAI or Ollama
# ---------------------------------------------------------------------
def call_llm(
    messages: List[Dict[str, str]],
    provider: str,
    openai_model: str,
    ollama_model: str,
) -> str:
    """
    Unified LLM call:
      - provider='openai'  -> use OpenAI Chat Completions API
      - provider='ollama'  -> use local Ollama HTTP API
    """
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        return completion.choices[0].message.content

    elif provider == "ollama":
        # Ollama chat API at http://localhost:11434/api/chat
        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
        }
        resp = requests.post(
            "http://localhost:11434/api/chat", json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama: {"message": {"role": "assistant", "content": "..."}}
        return data["message"]["content"]

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------
#  Retrieval layer
# ---------------------------------------------------------------------
class FAQRetriever:
    def __init__(self, index_path: str, qa_path: str, model_name: str):
        self.index = faiss.read_index(index_path)

        with open(qa_path, "rb") as f:
            data = pickle.load(f)
        self.items = data["items"]
        self.language = data.get("language", "en")
        self.docx_path = data.get("docx_path", "")

        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            item = self.items[idx]
            results.append(
                {
                    "score": float(score),
                    "index": int(idx),
                    "section": item["section"],
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
        return results


def build_context_text(results) -> str:
    """Build CONTEXT string from retrieved Q/A items."""
    parts = []
    for r in results:
        parts.append(
            f"[CHUNK {r['index']} | relevance={r['score']:.3f}]\n"
            f"Section: {r['section']}\n"
            f"Q: {r['question']}\n"
            f"A: {r['answer']}\n"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------
#  Lexical coverage guard: prevent truly out-of-FAQ concepts
# ---------------------------------------------------------------------
def check_coverage(user_query: str, context_text: str, min_len: int = 5, max_missing: int = 2):
    """
    Guard: if the user mentions MANY words (>= min_len chars) that do not
    appear anywhere in the FAQ context, treat the question as out-of-scope.

    We allow up to `max_missing` unseen words to avoid blocking legitimate
    paraphrases (e.g. "clotting" vs "blood clot").

    Returns (covered: bool, missing_words: list[str]).
    """
    uq_words = set(re.findall(r"\b\w+\b", user_query.lower()))
    ctx_words = set(re.findall(r"\b\w+\b", context_text.lower()))

    missing = [
        w for w in uq_words
        if len(w) >= min_len and w not in ctx_words
    ]
    covered = len(missing) <= max_missing
    return covered, missing


# ---------------------------------------------------------------------
#  Answer formatting without LLM
# ---------------------------------------------------------------------
def format_direct_answer(result: Dict[str, Any], language: str) -> str:
    """
    Directly answer from the FAQ Q/A, in a gentle wrapper, without using an LLM.
    """
    section = result["section"]
    question = result["question"]
    answer = result["answer"]

    if language == "fr":
        intro = (
            "Voici ce que la FAQ indique à ce sujet. "
            "Gardez en tête que cela ne remplace pas l’avis de votre équipe soignante :\n\n"
        )
        outro = (
            "\n\nCette information est issue d’une FAQ générale sur l’hormonothérapie adjuvante. "
            "Votre situation personnelle peut être différente : n’hésitez pas à en parler avec "
            "votre oncologue ou votre équipe soignante."
        )
    else:
        intro = (
            "Here is what the FAQ says about this topic. "
            "Please remember this does not replace advice from your care team:\n\n"
        )
        outro = (
            "\n\nThis information comes from a general FAQ about adjuvant hormone therapy. "
            "Your own situation may be different, so it’s important to discuss it with your "
            "oncologist or care team."
        )

    return (
        f"{intro}"
        f"Section: {section}\n"
        f"Question: {question}\n\n"
        f"{answer}"
        f"{outro}"
    )


# ---------------------------------------------------------------------
#  Core chat logic
# ---------------------------------------------------------------------
def chat_once(
    retriever: FAQRetriever,
    user_query: str,
    use_llm: bool = False,
    debug: bool = False,
    llm_provider: str = "openai",
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "llama3.2",
) -> str:
    results = retriever.retrieve(user_query, top_k=5)

    if debug:
        print("\n[DEBUG] Retrieved candidates:")
        if not results:
            print("[DEBUG] No results.")
        else:
            for r in results:
                print(
                    f"  idx={r['index']}  score={r['score']:.3f}  "
                    f"section={r['section'][:50]!r}  question={r['question'][:80]!r}"
                )
        print()

    if not results or results[0]["score"] < 0.2:
        if retriever.language == "fr":
            return (
                "Je ne trouve pas d’information claire à ce sujet dans la FAQ sur laquelle "
                "je suis basée. Comme je ne dois pas inventer d’informations médicales, "
                "je ne peux pas répondre en détail.\n\n"
                "Le mieux est d’en parler directement avec votre équipe soignante, "
                "qui connaît votre situation et pourra vous donner des conseils adaptés."
            )
        else:
            return (
                "I’m not finding clear information about that in the FAQ I’m based on. "
                "Because I must not invent medical information, I can’t safely answer this in detail.\n\n"
                "It would be best to discuss this question directly with your oncology team, "
                "who know your situation and can give you personalized guidance."
            )

    # If we are not using an LLM, just return the top FAQ answer.
    if not use_llm:
        top = results[0]
        return format_direct_answer(top, retriever.language)

    # Build context from retrieved chunks
    context_text = build_context_text(results)

    # ---- SOFTER FAQ-ONLY GUARD ----
    # Allow up to 2 "unknown" content words before treating as out-of-scope
    covered, missing = check_coverage(user_query, context_text, min_len=5, max_missing=2)
    if not covered:
        missing_clean = ", ".join(sorted(set(missing)))
        if retriever.language == "fr":
            return (
                "Votre question fait intervenir plusieurs éléments qui ne figurent pas clairement "
                "dans la FAQ sur laquelle je suis basée (par exemple : "
                f"{missing_clean}). Pour rester fiable et ne pas inventer d’informations médicales, "
                "je ne peux pas répondre précisément.\n\n"
                "Le mieux est d’en parler directement avec votre équipe soignante, "
                "qui connaît votre situation et pourra vous conseiller."
            )
        else:
            return (
                "Your question includes several elements that do not clearly appear in the FAQ I’m based on "
                f"(for example: {missing_clean}). To avoid inventing medical information, I can’t "
                "safely answer this in detail.\n\n"
                "Please discuss this directly with your oncology team, who know your situation "
                "and can advise you."
            )
    # ---- END GUARD ----

    system_prompt = get_system_prompt(retriever.language)

    system_message = {
        "role": "system",
        "content": system_prompt
                   + "\n\nCONTEXT (FAQ excerpts):\n"
                   + context_text,
    }
    user_message = {"role": "user", "content": user_query}

    return call_llm(
        [system_message, user_message],
        provider=llm_provider,
        openai_model=openai_model,
        ollama_model=ollama_model,
    )


# ---------------------------------------------------------------------
#  CLI entrypoint
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG-based FAQ chatbot for adjuvant hormone therapy."
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "fr"],
        default="en",
        help="Language of the FAQ / chatbot.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Path to FAISS index. Default: faq_index_<language>.faiss",
    )
    parser.add_argument(
        "--qa",
        default=None,
        help="Path to Q/A pickle. Default: faq_qa_<language>.pkl",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="If set, call an LLM on top of retrieval (OpenAI or Ollama).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use when --use-llm is set.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name (when --llm-provider openai).",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (when --llm-provider ollama).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks (scores, sections, questions) for each query.",
    )

    args = parser.parse_args()

    language = args.language
    index_path = args.index or f"faq_index_{language}.faiss"
    qa_path = args.qa or f"faq_qa_{language}.pkl"

    print(f"Language: {language}")
    print(f"Index path: {index_path}")
    print(f"Q/A pickle: {qa_path}")
    print(f"Use LLM: {args.use_llm}")
    print(f"LLM provider: {args.llm_provider}")
    print(f"OpenAI model: {args.openai_model}")
    print(f"Ollama model: {args.ollama_model}")
    print(f"Debug mode: {args.debug}\n")

    print("Loading FAQ retriever...")
    retriever = FAQRetriever(
        index_path=index_path,
        qa_path=qa_path,
        model_name=EMBEDDING_MODEL_NAME,
    )

    if retriever.language == "fr":
        print("Chatbot prêt (FAQ hormonothérapie adjuvante, FR).")
        print("Tapez 'exit' ou 'quit' pour quitter.\n")
    else:
        print("Chatbot ready (adjuvant hormone therapy FAQ, EN).")
        print("Type 'exit' or 'quit' to leave.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            if retriever.language == "fr":
                print("Bot: Prenez bien soin de vous. Au revoir.")
            else:
                print("Bot: Take good care of yourself. Goodbye.")
            break

        try:
            answer = chat_once(
                retriever,
                user_query=user_query,
                use_llm=args.use_llm,
                debug=args.debug,
                llm_provider=args.llm_provider,
                openai_model=args.openai_model,
                ollama_model=args.ollama_model,
            )
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            continue

        print("\nBot:", answer, "\n")


if __name__ == "__main__":
    main()
