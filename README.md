![hormonAI](hormonAI_transparent.png)

*A compassionate, clincally validated chatbot for adjuvant hormone therapy*

hormonAI is a **Retrieval-Augmented Generation (RAG) chatbot** built on a curated medical knowledge base (FAQ **and** reference articles) about **adjuvant hormone therapy for breast cancer**. In the chat interface the assistant is presented as **Mona**.

It is designed to:
- Answer **only** from the provided knowledge base (FAQ + articles)
- Be **multilingual** (English / French) in one shared embedding space
- Offer both a **CLI chatbot** and a **Streamlit GUI**
- Prioritize **safety, transparency, and source citation**

## Features
- Shared multilingual space (EN/FR): same-language answers preferred, with a cross-lingual recall fallback that carries a clear "only available in <language>" notice (content is quoted verbatim, never machine-translated)
- Type-aware ingestion: atomic FAQ Q/A pairs and long-form articles (heading-split, parent-linked chunks) in one unified index
- Hybrid retrieval: dense vector search + BM25 (keyword), fused with Reciprocal Rank Fusion, then optional CrossEncoder reranking
- Semantic-first gating with IDF-weighted anchors (no hand-maintained stopword/generic lists)
- Model-agnostic embeddings (mpnet, BGE-M3, e5, ...) selected with one setting
- Source transparency: every answer cites its knowledge-base origin
- LLM toggle: verbatim quotes vs grounded LLM rephrasing (rewrites ONLY the retrieved text for clarity + empathy, never adding facts; verbatim sources always cited; falls back to verbatim if the LLM is unavailable)
- Safety guardrails: out-of-scope questions are declined
- Queries are logged for auditing and source material improvement

## Knowledge base
[HormonAI corpus](https://www.dropbox.com/scl/fo/wufdoiep8biwrrjygfska/AAnfOC0u-pUUPFAuDTzhncI?rlkey=u2fn41qvhf0vuo7az0vms6tnt&dl=0)

## Scoring
1) Retrieval produces a fused score (default)

For every query, `HybridFAQRetriever.retrieve()` runs **four retrieval channels**:

1. **BM25** over tokenized query vs BM25 docs (built from *heading/section + question + answer*)
2. **FAISS index over “Q-only” embeddings**
3. **FAISS index over “Q rephrasing” embeddings** (optional)
4. **FAISS index over “QA” embeddings** (question + answer text)

Each channel returns a top-k ranked list of candidate indices. These ranks are then combined using **Reciprocal Rank Fusion (RRF)**:

* For each index `idx`, we sum:
  `rrf(rank) = 1 / (60 + rank)`
  across BM25 + FAISS(Q) + FAISS(QA).

That sum is stored as `RetrievalCandidate.fused_score`.

`fused_score` is **not a similarity score**; it’s a **rank-fusion score**. It’s useful for ordering but its absolute value has no standalone meaning.

2) Optional reranking produces rerank_score (only when enabled)

If `HybridFAQRetriever(rerank=True)` and CrossEncoder loads successfully:
* Create pairs:
  `(user_query, section + question + answer)`
* The CrossEncoder predicts a relevance score for each candidate.
* Store it as `RetrievalCandidate.rerank_score`.
* Sort candidates by rerank_score descending, overriding fused ordering.


* `fused_score` = the hybrid retrieval “first-pass” ordering
* `rerank_score` = second-pass reordering using a CrossEncoder

3) Answer / abstain decision (semantic-first)

The decision to answer or abstain is **semantic-first**: the primary signal is the CrossEncoder `rerank_score` (when reranking is on) or dense cosine similarity (otherwise). IDF-weighted lexical anchors no longer gate the answer; they only shape multi-concept bundles and provide a recall safety net for rare, highly specific terms. Thresholds (`--rerank-threshold`, `--sem-threshold`, `--dense-floor`) are model-specific and should be calibrated on the gold eval set. See `RETRIEVAL_REVIEW.md` for the full design.

---

## Repository structure
```text
.
├── ingest.py              # Type-aware, model-agnostic ingestion (FAQ + articles, shared index)
├── ingest_faq.py          # Legacy FAQ-only ingestion (still supported)
├── rag_core.py            # Core RAG logic (retriever, IDF anchors, semantic-first gate, cross-lingual fallback)
├── chatbot.py             # Command-line chatbot
├── hormonai_app.py        # Streamlit GUI (chat assistant shown as "Mona")
├── audit_logger.py        # Query logger
├── manifest.example.json  # Example ingestion manifest (FAQ + articles, EN + FR)
├── RETRIEVAL_REVIEW.md    # Retrieval design notes and change log
├── hormonAI.png           # Logo (used by the GUI)
├── data/                  # Generated indexes
│   ├── kb_all_*          # Combined SHARED multilingual index (FAISS q/qa/qp + BM25 + qa.pkl)
│   ├── kb_{lang}_*.faiss # Per-language FAISS indexes
│   └── kb_{lang}_*.pkl   # Per-language metadata / BM25
├── tests/                 # Evaluation harness + gold set
│   ├── eval_set.jsonl     # Gold-standard eval cases (EN + FR, categorized)
│   ├── eval_retrieval.py  # Runs the gold set: recall@k, false-answer/abstention rates
│   ├── pdf_fidelity.py    # PDF extraction QA (per-source metrics + preview)
│   └── inspect_qa.py
├── corpus/                # Source knowledge base (FAQ + articles), by language
│   ├── en/                # e.g. 20250613_FAQ_Hormono_EN.docx, *.pdf articles
│   └── fr/                # e.g. 20250613_FAQ_Hormono_FR.docx, *.pdf articles
└── README.md
```

## Create a virtual environment and install dependencies
```bash
python3 -m venv ht_faq_rag

source ht_faq_rag/bin/activate

pip install -r requirements.txt
```

## Ingest the knowledge base (FAQ + articles)

`ingest.py` is the current, type-aware ingester. It handles two document shapes and writes one unified, lang-tagged index:

- **FAQ** documents (`.docx`): parsed into atomic Q/A pairs.
- **Article** documents (`.docx`, `.md`, `.txt`, `.pdf`): split on their own headings where available (docx styles / markdown `#`), then into fixed-size overlapping child chunks that keep a heading breadcrumb and a link to their parent section. PDFs carry no reliable heading structure, so each PDF is treated as one flowing document and chunked directly. PDF text is extracted with **PyMuPDF** (falling back to `pypdf`) and cleaned (running headers/footers removed, hyphenated line-wraps rejoined, page-number lines dropped).

Sources are declared in a JSON manifest that tags each file's `type` and `lang`. A local knowledge-base directory (e.g. `corpus/`) is referenced by pointing the manifest paths at it (see `manifest.example.json`):

```json
[
  {"path": "corpus/en/20250613_FAQ_Hormono_EN.docx", "type": "faq",     "lang": "en"},
  {"path": "corpus/en/BCN-Tamoxifen.pdf",             "type": "article", "lang": "en"},
  {"path": "corpus/fr/20250613_FAQ_Hormono_FR.docx",  "type": "faq",     "lang": "fr"},
  {"path": "corpus/fr/SCC-HT.pdf",                    "type": "article", "lang": "fr"}
]
```

### Set the models (embedding + reranker)

The embedding model must be identical at ingestion and query time. Export the models once so ingestion, the CLI, and the GUI all pick them up automatically:

```bash
export HORMONAI_EMBEDDING_MODEL="BAAI/bge-m3"            # dense embeddings (must match ingest)
export HORMONAI_RERANK_MODEL="BAAI/bge-reranker-v2-m3"   # cross-encoder reranker (recommended)
```

The recommended stack is **BGE-M3** embeddings with the matched **bge-reranker-v2-m3** cross-encoder (both multilingual, Apache-2.0). The older `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranker (the default if unset) is lighter and CPU-friendly but ranks lay/clinical questions noticeably worse.

> Both are loaded via `sentence-transformers` and download large models on first use (BGE-M3 ~2.3GB, bge-reranker-v2-m3 ~2.3GB); they are much faster on GPU. Changing the embedding model requires a fresh ingest. Changing **either** model changes the score scale, so re-calibrate the gate thresholds (below) against the gold eval set afterward.

### Check PDF extraction fidelity (recommended before ingesting)

PDF extraction is the weakest link, and for a medical tool a garbled number or a dropped "not" is a safety issue. This script reports per-source extraction health (backend, pages, chars/page, chunk count, and a scanned/thin/ok flag) and, with `--preview`, prints the cleaned head/tail and first chunk so wording and numbers can be spot-checked. It exits non-zero if any PDF looks scanned/empty (no text layer, needs OCR or a cleaner source).

```bash
python tests/pdf_fidelity.py --manifest manifest.json
python tests/pdf_fidelity.py --preview --only NCI-HT.pdf
```

> Note: PyMuPDF is AGPL-licensed. That is fine for internal/research use; review licensing before redistributing a product built on it. `pypdf` (BSD) remains the automatic fallback.

### Build the indexes

By default `ingest.py` builds **both** the combined shared multilingual index (`kb_all_*`, used for cross-lingual fallback) **and** the per-language indexes (`kb_<lang>_*`):

```bash
python ingest.py --manifest manifest.json --embedding-model "BAAI/bge-m3"
```

Chunking and index selection can be tuned:

```bash
# larger article chunks, shared index only
python ingest.py --manifest manifest.json --embedding-model "BAAI/bge-m3" \
  --chunk-size 300 --chunk-overlap 50 --no-per-language

# A/B a second model into a separate prefix (kb_all_bgem3_*, etc.)
python ingest.py --manifest manifest.json --embedding-model "BAAI/bge-m3" --out-prefix kb_bgem3
```

Convenience flags exist for quick runs without a manifest:

```bash
python ingest.py --language en --faq corpus/en/20250613_FAQ_Hormono_EN.docx \
  --article corpus/en/BCN-Tamoxifen.pdf --embedding-model "BAAI/bge-m3"
```

Outputs per index prefix:

```text
kb_all_index_q.faiss     # FAISS on the topic/question side (shared, all languages)
kb_all_index_qa.faiss    # FAISS on question/heading + body text
kb_all_index_qp.faiss    # FAISS on question + paraphrases (if any)
kb_all_qa.pkl            # Unified, lang-tagged items + metadata (embedding model, prefixes)
kb_all_bm25.pkl          # BM25 index
kb_<lang>_*              # The same five files, per language
```

### Legacy FAQ-only ingestion (`ingest_faq.py`)

The original FAQ-only pipeline is still supported and is what produced the current per-language indexes. It does not build the shared `kb_all_*` index and does not handle articles.

```bash
python ingest_faq.py -l en -d corpus/en/20250613_FAQ_Hormono_EN.docx
python ingest_faq.py -l fr -d corpus/fr/20250613_FAQ_Hormono_FR.docx

# optional: question paraphrase augmentation (requires Ollama at http://localhost:11434)
python ingest_faq.py -l en --augment-questions --paraphrase-n 6 -d corpus/en/20250613_FAQ_Hormono_EN.docx
```

## Evaluate retrieval + gating

`tests/eval_set.jsonl` is a gold set (EN + FR, categorized) keyed by **stable item IDs** (e.g. `20250613_FAQ_Hormono_EN:21`), so it survives re-ingestion and corpus growth. `tests/eval_retrieval.py` runs it through the real pipeline and reports decision accuracy, recall@k, **lead-source correctness** (did the answer headline a gold source?), **gold-source-in-bundle**, and the false-abstention / false-answer rates.

```bash
export HORMONAI_EMBEDDING_MODEL="BAAI/bge-m3"
export HORMONAI_RERANK_MODEL="BAAI/bge-reranker-v2-m3"
python tests/eval_retrieval.py --shared --verbose      # evaluate the shared (production) path
```

## Run hormonAI (CLI)
```bash
python chatbot.py -l en [--debug]
```
CrossEncoder reranking is **on by default** (disable with `--no-rerank`).

## Run against the shared multilingual index (cross-lingual fallback)
```bash
python chatbot.py -l fr --shared
```
`-l` selects the active query language; the shared index answers in that
language when possible and falls back cross-lingual with a notice otherwise.

## Run with grounded LLM rephrasing
```bash
python chatbot.py -l en --use-llm   # requires Ollama at http://localhost:11434 (default model: llama3.2)
```
With `--use-llm`, the answer is rephrased by the LLM using **only** the retrieved
source text (for clarity and empathy, never adding facts), and the verbatim
sources are still cited beneath. If the LLM is unreachable, it falls back to the
verbatim answer. Without `--use-llm`, the retrieved text is quoted verbatim
with a fixed empathy line.

## Run the app (GUI)
```bash
# export the same models used at ingestion, then launch
export HORMONAI_EMBEDDING_MODEL="BAAI/bge-m3"
export HORMONAI_RERANK_MODEL="BAAI/bge-reranker-v2-m3"
streamlit run hormonai_app.py
```
The GUI loads the shared `kb_all_*` index automatically (with same-language
preference and cross-lingual fallback) and falls back to the per-language
indexes if `kb_all_*` has not been built. The chat assistant is shown as **Mona**.

## Configuration (environment variables)

The CLI, GUI, and eval harness all read these so a calibrated configuration is applied everywhere:

| Variable | Default | Purpose |
|---|---|---|
| `HORMONAI_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Dense embedding model. **Must match the model used at ingestion.** |
| `HORMONAI_RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder reranker. Recommended: `BAAI/bge-reranker-v2-m3`. |
| `HORMONAI_RERANK_THRESHOLD` | `-1.0` | Accept threshold on the reranker score (when reranking is on). Model-specific — calibrate. |
| `HORMONAI_SEM_THRESHOLD` | `0.62` | Dense-cosine accept threshold (used when reranking is off). |
| `HORMONAI_DENSE_FLOOR` | `0.50` | Cosine floor for the high-IDF lexical safety net. |
| `HORMONAI_LLM_MODEL` | `llama3.2` | Ollama model used for grounded rephrasing (`--use-llm`). |

Thresholds are **model-specific starting points, not tuned values**. After any embedding/reranker change, sweep them on the gold set and set the winning values via these env vars. For a patient-facing tool, keep the out-of-scope **false-answer rate at zero** as the hard constraint.

```bash
# recalibrate after a model change (watch the FALSE-ANSWER and lead-source-correctness rows)
python tests/eval_retrieval.py --shared --verbose
python tests/eval_retrieval.py --shared --rerank-threshold 0     # sweep around the default
```

