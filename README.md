![hormonAI](hormonAI.png)

*A compassionate, FAQ-restricted RAG chatbot for adjuvant hormone therapy*

hormonAI is a **proof-of-concept Retrieval-Augmented Generation (RAG) chatbot** built on a curated medical FAQ about **adjuvant hormone therapy for breast cancer**.  
It is designed to:
- Answer **only** from the provided FAQ
- Be **language-aware** (English / French)
- Offer both a **CLI chatbot** and a **Streamlit GUI**
- Prioritize **safety, transparency, and source citation**

## Features
- Language switch: fully updates UI + retrieval
- LLM toggle: FAQ text only vs empathetic rephrasing
- Source transparency: every answer cites its FAQ origin
- Safety guardrails: out-of-scope questions are declined

---

## Repository structure

```text
.
├── ingest_faq.py          # Ingest DOCX FAQ; build hybrid retrieval indexes
├── rag_core.py            # Core RAG logic (retriever, gating, LLM wrappers)
├── chatbot.py             # Command-line chatbot
├── hormonai_app.py        # Streamlit GUI
├── faq_en_*.faiss         # Generated FAISS indexes (English)
├── faq_en_*.pkl           # Generated metadata / BM25 (English)
├── faq_fr_*.faiss         # Generated FAISS indexes (French)
├── faq_fr_*.pkl           # Generated metadata / BM25 (French)
├── hormonAI.png           # Logo (used by the GUI)
└── README.md
```

## Create a virtual environment and install dependencies
```bash
python3 -m venv ht_faq_rag
source ht_faq_rag/bin/activate

pip install -r requirements.txt
```

## Ingest the FAQ (DOCX to RAG indexes)
Parses the FAQ document by section and Q/A. 
Builds a hybrid retriever:  
- FAISS on questions
- FAISS on Q+A
- BM25 lexical search
```python
python ingest_faq.py -l en -d 20250613_FAQ_Hormono_EN.docx
python ingest_faq.py -l fr -d 20250613_FAQ_Hormono_FR.docx
```
```text
faq_<lang>_index_q.faiss    # FAISS index on questions
faq_<lang>_index_qa.faiss   # FAISS index on question+answer
faq_<lang>_bm25.pkl         # BM25 index
faq_<lang>_qa.pkl           # Parsed FAQ items + metadata
```

## Run the chatbot (CLI)
```python
python chatbot.py -l en
```

## Run the chatbot with re-ranking (optional)
```python
python chatbot.py -l en --rerank
```

## Run the chatbot with an LLM for rephrasing
```python
ollama run llama3.2
python chatbot.py -l en --use-llm --llm-provider ollama --ollama-model llama3.2
```

## Run the app
```python
streamlit run hormonai_app.py
```

