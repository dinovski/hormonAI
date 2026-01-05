![hormonAI](hormonAI.png)

*A compassionate, FAQ-restricted RAG chatbot for adjuvant hormone therapy*

hormonAI is a **Retrieval-Augmented Generation (RAG) chatbot** built on a curated medical FAQ about **adjuvant hormone therapy for breast cancer**.  

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
├── hormonAI.png           # Logo (used by the GUI)
├── data/                  # Generated data
│   ├── faq_{lang}_*.faiss # FAISS Q and Q/A indexes
│   └── faq_{lang}_*.pkl   # Metadata / BM25
├── tests/                 # Test scripts
│   ├── inspect_qa.py
│   └── test_retreival.py
├── docs/                  # FAQ documents
│   ├── 20250613_FAQ_Hormono_EN.docx
│   └── 20250613_FAQ_Hormono_FR.docx
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
- BM25 keyword-based retrieval
```python
python ingest_faq.py -l en -d docs/20250613_FAQ_Hormono_EN.docx
python ingest_faq.py -l fr -d docs/20250613_FAQ_Hormono_FR.docx
```
```text
faq_<lang>_index_q.faiss    # FAISS index on questions
faq_<lang>_index_qa.faiss   # FAISS index on question+answer
faq_<lang>_qa.pkl           # Parsed FAQ items + metadata
faq_<lang>_bm25.pkl         # BM25 index
```

## Run the chatbot (CLI)
```python
python chatbot.py -l en
```

## Run the chatbot with re-ranking
```python
python chatbot.py -l en --rerank (--debug)
```

## Run the chatbot with an LLM for rephrasing
```python
ollama run llama3.2
python chatbot.py -l en --use-llm --llm-provider ollama --ollama-model llama3.2
```

## Run the app (GUI)
```python
streamlit run hormonai_app.py
```

