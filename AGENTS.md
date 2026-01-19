# AGENTS.md

## AI Agent Guidelines for hormonAI Development

This document provides guidance for AI coding assistants (such as Claude, GitHub Copilot, Cursor, or similar tools) working on the hormonAI codebase. It outlines the project's architecture, design principles, and best practices to ensure consistency and quality.

---

## Project Overview

**hormonAI** is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about adjuvant hormone therapy for breast cancer. The system is built with safety, transparency, and medical accuracy as core priorities.

### Key Characteristics

- **Domain-restricted**: Only answers from a curated FAQ document
- **Bilingual**: Supports English and French with full UI/retrieval language switching
- **Hybrid retrieval**: Combines semantic search (FAISS) with keyword search (BM25)
- **Safety-first**: Declines out-of-scope questions gracefully
- **Source transparency**: Every answer cites its FAQ origin
- **Optional LLM rephrasing**: Can use LLM for empathetic responses or return raw FAQ text

---

## Architecture Components

### Core Modules

1. **ingest_faq.py**: FAQ ingestion and index building
   - Parses DOCX FAQ documents by section and Q/A pairs
   - Builds FAISS indexes (question-only, Q+A, question rephrasings)
   - Creates BM25 keyword index
   - Generates embeddings using sentence-transformers

2. **rag_core.py**: Core RAG logic
   - `HybridFAQRetriever`: Multi-channel retrieval with RRF fusion
   - `FallbackRAG`: Main RAG orchestrator with safety gates
   - LLM wrappers for response generation
   - Optional CrossEncoder reranking

3. **chatbot.py**: Command-line interface
   - Interactive REPL-style chatbot
   - Supports debug mode and various retrieval strategies

4. **hormonai_app.py**: Streamlit GUI
   - User-friendly web interface
   - Language switching, LLM toggle, reranking options
   - Conversation history and source display

5. **audit_logger.py**: Query logging system
   - Logs all queries for auditing and FAQ improvement

---

## Retrieval Architecture

### Four-Channel Hybrid Retrieval

The system uses four retrieval channels that are fused together:

1. **BM25 (keyword-based)**: Built from section + question text
2. **FAISS index over "Q-only" embeddings**: Semantic search on questions
3. **FAISS index over "Q rephrasing" embeddings** (optional): Augmented question variants
4. **FAISS index over "QA" embeddings**: Semantic search on question + answer pairs

### Scoring System

- **Fused Score**: Reciprocal Rank Fusion (RRF) across all channels
  - Formula: `rrf(rank) = 1 / (60 + rank)`
  - Not a similarity score; used for relative ordering
  
- **Rerank Score** (optional): CrossEncoder second-pass reranking
  - Predicts relevance between query and full FAQ text
  - Overrides fused score when enabled

---

## Design Principles

### 1. Safety and Medical Ethics

- **Never hallucinate**: Strictly answer from FAQ only
- **Graceful refusal**: Decline out-of-scope questions politely
- **Transparency**: Always cite sources
- **Language-appropriate**: Match user's language preference

### 2. Code Quality Standards

- **Type hints**: Use Python type annotations for all functions
- **Docstrings**: Document all classes and public methods
- **Error handling**: Graceful degradation, never crash
- **Logging**: Use Python logging module, not print statements
- **Configuration**: Avoid hardcoded values; use constants or config

### 3. Modularity

- Keep RAG logic separate from UI (CLI vs Streamlit)
- Single responsibility principle for each module
- Testable components with clear interfaces

---

## Development Guidelines

### Adding New Features

When adding functionality:

1. **Consider both interfaces**: Ensure CLI and Streamlit parity
2. **Update both languages**: Maintain English/French support
3. **Preserve safety**: Never bypass FAQ-only restriction
4. **Add tests**: Create corresponding test cases in `/tests/`
5. **Update documentation**: Modify README.md and this file

### Modifying Retrieval Logic

When changing `rag_core.py`:

- Maintain backward compatibility with existing indexes
- Consider impact on both scoring methods (fused vs rerank)
- Test with both languages
- Document scoring behavior changes
- Preserve citation mechanism

### Adding New Dependencies

Before adding a new package:

1. Verify it's necessary and well-maintained
2. Check license compatibility (project is MIT)
3. Add to `requirements.txt` with version pinning
4. Test installation in clean environment
5. Document usage in README if user-facing

---

## File Structure Conventions

### Data Directory (`/data/`)

Generated files follow this naming pattern:

```
faq_{lang}_index_q.faiss      # FAISS index on questions
faq_{lang}_index_qp.faiss     # FAISS index on augmented questions (optional)
faq_{lang}_index_qa.faiss     # FAISS index on question+answer
faq_{lang}_qa.pkl             # Parsed FAQ items + metadata
faq_{lang}_bm25.pkl           # BM25 index
```

Do not commit generated data files to git (covered by `.gitignore`).

### Logs Directory (`/logs/`)

- Query logs stored with timestamp
- Format: `queries_YYYYMMDD_HHMMSS.json`
- Includes query, response, sources, and metadata

---

## Testing Strategy

### Current Test Files

- `tests/inspect_qa.py`: Inspect parsed FAQ data
- `tests/test_retrieval.py`: Test retrieval quality

### When Adding Tests

- Test both languages separately
- Test edge cases (empty queries, very long queries)
- Test safety gates (out-of-scope questions)
- Test retrieval accuracy with known Q/A pairs
- Mock LLM calls when appropriate

---

## LLM Integration Notes

### Current LLM Support

- **Ollama** (local): Used for question rephrasing during ingestion
  - Model: `llama3.2` (default)
  - Endpoint: `http://localhost:11434`

- **OpenAI API** (optional): For response generation
  - Models: GPT-4, GPT-3.5-turbo
  - Used only when `--use-llm` flag enabled

### LLM Usage Guidelines

When working with LLM components:

1. **Always make LLM optional**: System must work without LLM
2. **Graceful fallback**: Return FAQ text if LLM fails
3. **Safety prompts**: Include instructions to stay on-topic
4. **Token limits**: Be mindful of context window constraints
5. **Cost awareness**: Log token usage for OpenAI calls

### Prompt Engineering

Current system prompt structure:

```
You are a compassionate assistant for breast cancer patients.
- Answer ONLY from the provided FAQ context
- If question is out of scope, politely decline
- Cite your sources
- Be empathetic but factual
- Match the user's language
```

When modifying prompts:
- Maintain safety constraints
- Test with edge cases
- Preserve citation behavior
- Keep language-specific variations aligned

---

## Language Support

### Adding a New Language

To add support for a new language:

1. **FAQ Document**: Obtain translated FAQ in DOCX format
2. **Ingest**: Run `ingest_faq.py -l {lang_code} -d {path}`
3. **Embedding Model**: Verify multilingual support or use language-specific model
4. **UI Strings**: Add translations to both CLI and Streamlit
5. **Testing**: Create test cases for the new language
6. **Documentation**: Update README with new language option

### Language-Specific Considerations

- **Tokenization**: BM25 may need language-specific tokenizers
- **Stop words**: Configure appropriate stop words for each language
- **Embedding models**: Use multilingual models or language-specific variants
- **UI text**: Keep all user-facing strings translatable

---

## Common Tasks for AI Agents

### Task: Improve Retrieval Accuracy

```python
# Locations to modify:
# - rag_core.py: HybridFAQRetriever class
# - Consider adjusting RRF weights
# - Modify top-k values
# - Experiment with similarity thresholds
# - Test with benchmark queries from tests/
```

### Task: Add Conversation Memory

```python
# Implementation approach:
# 1. Extend chatbot.py and hormonai_app.py with history tracking
# 2. Modify FallbackRAG to accept conversation context
# 3. Update prompts to use context appropriately
# 4. Maintain safety: don't allow context to override FAQ restriction
# 5. Add memory limit to prevent token overflow
```

### Task: Improve LLM Response Quality

```python
# Tuning locations:
# - rag_core.py: _generate_response() method
# - Refine system prompt
# - Adjust temperature and top_p parameters
# - Add few-shot examples in prompt
# - Test tone and empathy with medical stakeholders
```

### Task: Add New Retrieval Channel

```python
# Steps:
# 1. Modify ingest_faq.py to build new index
# 2. Update HybridFAQRetriever.retrieve() to include new channel
# 3. Adjust RRF fusion to incorporate new rankings
# 4. Test impact on retrieval quality
# 5. Update documentation on scoring system
```

---

## Debugging Tips

### Common Issues

**Issue**: Retrieval returns irrelevant results
- Check embedding model compatibility
- Verify BM25 tokenization
- Inspect retrieved candidates with `--debug` flag
- Review RRF weights and top-k values

**Issue**: LLM generates off-topic responses
- Review system prompt constraints
- Check if FAQ context is properly injected
- Verify safety gates are active
- Test with `--no-llm` to isolate problem

**Issue**: Language switching not working
- Verify correct index files exist for target language
- Check language code matching (`en` vs `fr`)
- Ensure FAQ document was ingested for that language

**Issue**: FAISS index errors
- Rebuild indexes: re-run `ingest_faq.py`
- Check FAISS version compatibility
- Verify embedding dimensions match

---

## Performance Considerations

### Optimization Priorities

1. **Retrieval speed**: Keep top-k low, cache embeddings
2. **Memory usage**: Stream large files, manage index sizes
3. **LLM latency**: Use streaming responses where possible
4. **Startup time**: Lazy-load models, cache indexes

### Benchmarking

Key metrics to monitor:

- Retrieval time per query
- LLM response time
- Memory footprint
- Index load time
- End-to-end latency (query → response)

---

## Security and Privacy

### Data Handling

- **No PII collection**: Don't log personally identifiable information
- **Query anonymization**: Remove identifying details from logs
- **Secure API keys**: Use environment variables, never commit keys
- **HTTPS only**: Enforce secure connections for production deployment

### Medical Compliance

- **Not medical advice disclaimer**: Include in UI
- **Source citation**: Always provide FAQ references
- **No diagnosis**: Explicitly avoid diagnostic language
- **Encourage consultation**: Suggest speaking with healthcare providers

---

## Deployment Notes

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] FAQ documents up to date
- [ ] Indexes rebuilt with latest FAQ
- [ ] Environment variables configured
- [ ] API keys secured
- [ ] Logging configured
- [ ] Error monitoring enabled
- [ ] Medical disclaimer visible
- [ ] Language support verified

### Environment Variables

```bash
# Required for LLM mode
OPENAI_API_KEY=sk-...

# Optional: Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Logging level
LOG_LEVEL=INFO
```

---

## Contributing

### Code Review Criteria

When reviewing changes:

1. **Safety**: Does it maintain FAQ-only restriction?
2. **Accuracy**: Does it preserve citation mechanism?
3. **Usability**: Is it user-friendly for both languages?
4. **Performance**: Does it impact response time?
5. **Testing**: Are there adequate tests?
6. **Documentation**: Is it documented?

### AI Agent Self-Review

Before submitting changes, verify:

- [ ] Code follows project conventions
- [ ] Type hints are present
- [ ] Docstrings are clear
- [ ] No hardcoded paths or credentials
- [ ] Both languages work
- [ ] Tests pass
- [ ] README updated if needed
- [ ] No sensitive data in commits

---

## Resources

### Key Dependencies

- **sentence-transformers**: Embedding generation
- **faiss-cpu**: Vector similarity search
- **rank-bm25**: Keyword-based retrieval
- **streamlit**: Web UI framework
- **langchain**: LLM orchestration utilities
- **python-docx**: DOCX parsing

### Useful Links

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

## Changelog

When making significant changes, document them here:

### [Unreleased]

- (Document changes as you work)

### Version 1.0

- Initial release
- Hybrid retrieval with RRF fusion
- Bilingual support (EN/FR)
- CLI and Streamlit interfaces
- Optional LLM rephrasing
- Optional CrossEncoder reranking

---

## Contact

For questions about this codebase or suggestions for improvement, please open an issue on GitHub.

---

*Last updated: 2026-01-19*
