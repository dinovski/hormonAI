# hormonAI retrieval review

Date: 2026-07-23
Scope: `rag_core.py`, `ingest_faq.py`, `chatbot.py`, `data/kb_{en,fr}_*`
Corpus size at review: EN = 73 items, FR = 69 items (parsed from a single adjuvant-hormone-therapy DOCX per language).

## 1. TL;DR

Hybrid fusion (BM25 + three FAISS indexes via RRF) surfaces genuinely relevant candidates. What blocks answers is the **post-retrieval keyword coverage gate** in `answer_query` (`rag_core.py:830-841`), which requires that *every* surviving query keyword be lexically matched in the candidate bundle (`covered == anchor_concepts`, `rag_core.py:704, 710, 834`). One junk word that is absent from the corpus is enough to force an abstention even when the correct answer was retrieved.

Priority order for our project: expand the corpus first (Section 5), then loosen/replace the gate (Section 4). Both matter, but the corpus is the ceiling on everything else.

## 2. Root cause of the example failure (verified)

Query: *"Is it ever okay to take a break from hormone therapy, or does that increase my risk?"*

Traced through the actual code and corpus:

| Step | Result |
|---|---|
| `extract_core_keywords` | `['ever', 'break', 'increase']` |
| `anchor_keywords` | `['ever', 'break', 'increase']` (none are drug terms) |
| `anchor_concepts` (stems) | `{'ever', 'break', 'increas'}` — 3 concepts |
| Branch taken | `len >= 2` → `_select_bundle_with_coverage(max_n=3)` requiring full coverage |
| `stem('ever')` in corpus | **0 of 73 items** contain it |
| `stem('break')` in corpus | items 22, 66 |
| `stem('increas')` in corpus | 11 items incl. 25, 28, 30, 43 |

The debug trace confirms it: `bundle_cover_debug` shows idx 30 covered `increase` and idx 22 covered `break`. Both concepts that *could* be covered *were* covered. Only `ever` was left uncovered, so `coverage_complete = False` and the bot abstained.

Two independent facts make this a clear bug, not correct caution:

1. **`ever` is a function word, not a concept.** It slipped through because it is missing from `EN_STOPWORDS` (`rag_core.py:46-58`). It can never be matched by any corpus, so any query containing it that also has ≥2 content words is structurally unanswerable.

2. **A genuinely on-topic answer was retrieved and then discarded.** Candidate idx=22 ("What are my options if side effects become difficult to tolerate?") has an answer that literally reads *"...your doctor may suggest: A therapeutic break, A change in hormone therapy medication..."* — i.e. it directly addresses taking a break. idx=44 ("waiting period between stopping treatment and conception") covers discontinuation. The content exists; the gate threw it away.

So this specific case is a **gate defect**, not a corpus gap. (Whether the corpus fully answers "does a break increase recurrence risk" is a separate, weaker point — the honest answer is only partially, via idx 22/44, which is exactly why corpus expansion is priority one.)

## 3. How the current gate over-restricts (systemic issues)

These all live in `answer_query` and its helpers:

**3.1 Full-coverage AND gate.** `_select_bundle_with_coverage` demands `covered == anchor_concepts` (`rag_core.py:704`) and both branches abstain unless `coverage_complete` (`rag_core.py:834-841`). Requiring *all* keywords is the single biggest source of over-strictness. Recall drops geometrically as the query gets longer, and one out-of-vocabulary token kills the answer.

**3.2 Hand-maintained stopword/generic lists are incomplete.** Function words like `ever`, `okay`, `really`, `actually`, `just`, `still`, `instead`, `anymore`, `sometimes` are not in `EN_STOPWORDS`/`GENERIC_EN`, so they survive as "concepts" that can never match. This is unbounded maintenance and will keep producing false abstentions.

**3.3 Coverage is exact-stem lexical, not semantic.** `_anchor_overlap_concepts` (`rag_core.py:249-258`) only counts a concept covered if its Snowball stem literally appears in the candidate. The lay↔clinical / synonym map `_concept_match_stems` (`rag_core.py:230-246`) hard-codes only `heart↔cardiovascular` and `bone↔osteoporosis`. So "break" does not match "pause"/"stop"/"interrupt"/"discontinue" even though the corpus uses exactly those words (`stem('pause')` → items 42, 43; `stem('stop')` → items 3, 35, 37, 43, 44). The embedding model already knows these are synonyms; the gate ignores that signal.

**3.4 Empty-anchor abstention on drug-only queries.** `anchor_keywords` strips all drug/treatment terms (`rag_core.py:294-305`), and `answer_query` abstains if `len(anchors) < 1` (`rag_core.py:789`). So "Tell me about tamoxifen" → anchors `[]` → abstain, despite tamoxifen being the corpus's core topic.

**3.5 No use of the similarity signal for the accept/reject decision.** Fused scores and the cross-encoder rerank score are computed but never thresholded. The keep/abstain decision is 100% lexical coverage. This is backwards: dense similarity is what indicates "this candidate is about the same thing," and it is discarded at exactly the moment it matters.

**3.6 Redundant query encoding.** `retrieve` encodes the identical string `f"Question: {user_query}"` three times for the q / qa / qp indexes (`rag_core.py:438-448`). Harmless but wasteful, and it means the three indexes differ only on the document side.

## 4. Short-term fixes (current prototype, ranked by impact/effort)

Do these on the existing 73/69-item corpus; none require re-ingesting except 4.6.

1. **Loosen the coverage gate from AND to a floor.** Replace `covered == anchor_concepts` with "cover the single highest-IDF anchor" or "cover ≥ max(1, ceil(n/2)) concepts." This alone answers the example query. (`_select_bundle_with_coverage`, `answer_query:830-841`) — lowest effort, highest impact.

2. **Add a semantic accept path.** Keep a candidate if its dense cosine (or cross-encoder score) to the query exceeds a tuned threshold, *regardless* of lexical coverage. Combine as: answer if (semantic ≥ τ) OR (coverage floor met). This is the principled version of #1.

3. **Expand stopwords and, better, gate by IDF.** Add the missing function words now, but the durable fix is to weight/drop anchors by corpus IDF so rare-but-meaningless tokens (`ever`) get near-zero weight automatically instead of being maintained by hand.

4. **Broaden the synonym map** in `_concept_match_stems`: add `break↔pause↔stop↔interrupt↔discontinue↔hold`, `pregnant↔pregnancy↔conceive↔conception`, `tired↔fatigue`, `bloodwork↔blood test`, etc. Stopgap until #2 makes it unnecessary.

5. **Turn on reranking by default.** The cross-encoder is implemented (`rag_core.py:485-496`) but `--rerank` is off (`chatbot.py:28`). Enabling it reorders candidates so the gate and the "best" selection operate on better-ordered results. Measure latency on the target hardware first.

6. **Fix ingest parse artifacts.** Answer fragments are being stored as questions: idx 7 ("Do not double up to make up for a missed dose."), idx 27 (the bisphosphonates sentence), and context-less stubs like idx 40 ("Is this common?"). These come from `looks_like_question_sentence`/`is_heading` misfiring in `parse_docx_into_qa` (`ingest_faq.py:81-213`). They pollute retrieval and coverage. Re-parse with tighter rules and spot-check with `--inspect`.

7. **Handle drug-only queries.** If anchors are empty but a drug/treatment term is present, fall back to retrieving on that term rather than abstaining (`answer_query:789`).

## 5. Long-term improvements (priority one: corpus)

**5.1 Expand the corpus — the stated first priority.**
- The whole system is capped by 73 EN / 69 FR items from one document each. Broaden sources (clinical FAQs, patient-education material, oncology society guidance) and, critically, **chunk answers** rather than indexing only one Q per answer, so sub-topics inside long answers (e.g. the "therapeutic break" line buried in idx 22) become independently retrievable.
- Add lay-language question variants per entry. The paraphrase augmentation (`--augment-questions`, already `True` in the saved data) is the right idea; verify the llama3.2 paraphrase quality, since bad paraphrases add noise.

**5.2 Build an evaluation set before tuning anything further.** Collect real patient questions (the audit log at `logs/audit.jsonl` is a start) and label the gold FAQ id(s) for each. Track **recall@k, abstention rate, and false-abstention rate**. There is no way to tell whether a gate change helps without this. This should gate every change in Section 4.

**5.3 Replace hand-crafted linguistics with learned components.** The stopword/generic/synonym/stemmer machinery is brittle and language-forked. Move the lay↔clinical mapping into either (a) an LLM query-rewrite/expansion step at query time, or (b) a domain-adapted bi-encoder fine-tuned on the project's Q–Q and Q–A pairs. Either handles "break↔pause" without a maintained dictionary.

**5.4 Make abstention a calibrated decision, not a lexical one.** Use the cross-encoder rerank score plus a threshold tuned on 5.2 to decide answer vs. abstain. This directly replaces the AND gate with something measurable and safety-tunable (the operating point can be chosen on the precision/recall curve the team is comfortable defending clinically).

**5.5 Improve the abstention UX.** When the gate blocks but near-miss candidates exist, surface them ("The closest topics I have are: pausing therapy for pregnancy; options when side effects are hard to tolerate") instead of a flat "I can't answer." Distinguish "blocked a relevant chunk" from "genuinely no content" using the semantic score.

## 6. Safety note

The strict gate is deliberate — for a patient-facing oncology tool, false answers are worse than abstentions. The recommendations above do not remove that guardrail; they move it from a brittle lexical proxy (which currently produces *false* abstentions on in-corpus content) to a calibrated semantic threshold we can tune and defend with data. Keep the "corpus-only, no invented facts" contract; make the accept/reject boundary measurable.

## 7. Implemented (2026-07-23) — short-term fixes + eval harness

All Section 4 fixes are now in the code:

- **Coverage gate loosened (4.1)** — `answer_query` no longer requires every anchor. It answers when the bundle covers `ceil(coverage_fraction * N)` of the *in-corpus* concepts (`coverage_fraction=0.5` default). `rag_core.py`, non-stats branch.
- **Semantic accept path (4.2)** — `retrieve` now captures the best dense cosine (`RetrievalCandidate.dense_sim`); `answer_query` answers if the top candidate is ≥ `sem_accept_threshold` (default 0.62) even with no lexical overlap, and uses it as a rescue when the coverage floor is missed.
- **Anchor pruning by corpus presence (4.3)** — concepts absent from the corpus (and their synonyms) are dropped before gating via `retriever.stem_in_corpus`. This removes junk tokens like `ever` that caused the example abstention. Stopword lists were also expanded (EN + FR).
- **Synonym map broadened (4.4)** — `_concept_match_stems` is now built from bidirectional groups including `break/pause/stop/interrupt/discontinue`, `pregnant/conceive`, `tired/fatigue`, `period/menstrual`, `mood/depression`, etc.
- **Reranking on by default (4.5)** — `chatbot.py --rerank` defaults on (disable with `--no-rerank`); Streamlit checkbox defaults on. Cross-encoder loads lazily and degrades gracefully.
- **Ingest parse artifacts fixed (4.6)** — `looks_like_question_sentence` in `ingest_faq.py` now rejects long declaratives and negative imperatives, so future ingests won't create answer-fragment "questions." For the *current* prebuilt indexes, `retrieve` applies a runtime filter (`_looks_like_faq_question`) that drops EN idx 7 and 27.
- **Drug-only queries (4.7)** — queries whose only content words are drug/treatment terms (empty anchors) now fall through to the semantic path instead of a hard abstention.

New tunables surfaced on the CLI: `--sem-threshold`, `--coverage-fraction`.

### How this was verified

The macOS venv does not run in a Linux sandbox and the embedding model was not available there, so the **full pipeline (real embeddings + FAISS) must be evaluated in our local environment**. The gate/keyword/pruning/artifact logic was verified against the real corpus by stubbing only the embedding/index layer and replaying an actual candidate set from a debug run. Confirmed:

1. The original example now **answers** (path `coverage_floor+semantic`; `ever` pruned; `break` covered by idx 22, `increase` by idx 30 — idx 22's answer is the "therapeutic break" content).
2. Out-of-scope "capital of France" **abstains** (no in-corpus concepts, dense sim below threshold).
3. Drug-only "aromatase inhibitors" **answers** via the semantic path (idx 3).
4. The artifact filter drops idx 7 and 27 and keeps idx 22.

### Round 2 (2026-07-23) — stats-intent misrouting + bundle precision

Follow-up after testing surfaced a second false-abstention class, e.g. *"How long do I need to stay on hormone therapy?"* was refused despite idx 2 being an almost exact match (dense 0.84, rerank 7.12).

- **Root cause:** `_is_stats_intent` treated the bare word `how` (and `many`/`often`) as a statistics signal, so any "how long / how much" duration question was routed into the strict stats gate, which requires a statistical marker word in the answer, found none, and hard-abstained — never reaching the semantic path. The French `combien` had the same flaw ("combien de temps" = how long).
- **Fix A — narrower stats intent:** `_is_stats_intent` now fires only on explicit statistical words (`percent(age)`, `proportion`, `rate`, `prevalence`, `incidence`, `odds`, `probability`, `likelihood`, `frequency`), a `%` sign, or specific bigrams (`how many`, `how often`, `how likely`, `what percentage/proportion/...`). French `combien de X` fires except `combien de temps`.
- **Fix B — stats gate no longer hard-abstains:** when a query looks statistical but no grounded statistic is found, it now falls through to the general coverage/semantic path (still quoting the FAQ, never fabricating numbers) instead of refusing.
- **Fix C — bundle precision guard:** secondary bundle members must clear the semantic bar (`min_member_sim = sem_accept_threshold`). This stops a strong lead answer from being padded with low-relevance entries that merely share a weak keyword (the "how long" query was pulling in idx 35 "long flights"/"stay hydrated"). Genuine multi-concept answers (e.g. osteoporosis + depression) are unaffected because both members clear the bar.

Re-verified on the real corpus by replaying the exact candidate set from the debug: "how long" now answers with idx 2 alone; the osteoporosis+depression multi-concept case still bundles both; the original break/risk example still answers; stats classification is correct on 8 EN/FR probes.

### Round 3 (2026-07-23) — long-term: IDF anchors + semantic-first gate

This replaces the brittle machinery rather than patching it, per the corpus-expansion priority.

- **IDF-weighted anchor extraction (replaces the word lists).** At load, the retriever computes per-stem document frequency and smoothed IDF over the corpus (`_stem_df`, `_stem_idf`, `average_idf`). `extract_anchor_concepts` selects anchors as stems that are present (df >= 1) and discriminative (df <= 0.5 x corpus_size). Common domain words ("hormone" df 53, "therapy" df 55, "take" df 40) get low weight and drop out automatically; absent fillers ("ever") drop out; no `GENERIC`/`EMOTION`/drug list is consulted. Only a small, domain-independent grammatical/discourse stop set remains (closed-class function words, which don't change with the domain). The old `extract_core_keywords`/`anchor_keywords` and the domain lists are retained in the file but no longer used by the gate.
- **Semantic-first gate.** The accept/abstain decision is now driven by the cross-encoder rerank score (when reranking is on) or dense cosine (otherwise), NOT by lexical coverage. Lexical/IDF anchors only (a) shape the bundle for multi-concept questions and (b) act as a recall safety net: a candidate covering a HIGH-IDF anchor (idf >= corpus average) at >= `dense_floor` cosine can still answer. Every bundle member must independently clear the accept threshold, so a strong lead is never padded with weak entries.
- **New tunables** (defaults are conservative starting points, NOT calibrated): `rerank_accept_threshold=0.0`, `sem_accept_threshold=0.62` (dense, used when rerank off), `dense_floor=0.50`. Surfaced on the CLI as `--rerank-threshold`, `--sem-threshold`, `--dense-floor`. `coverage_fraction` is deprecated (kept only for call compatibility).

Verified on the real corpus by replaying candidate sets with realistic rerank/dense values: IDF extraction drops "hormone/therapy/take" (common) and "ever" (absent) with no lists; "how long" answers via rerank-primary (idx 2); the break/risk example answers via dense-primary; "capital of France" abstains; drug-only "aromatase inhibitors" answers via primary (idx 3); osteoporosis+depression bundles both; an ungrounded stats question abstains.

**Calibration is now the critical step and it must run in our local environment.** The whole gate hinges on the two thresholds, and for a patient-facing tool the false-ANSWER rate is the number to protect. Sweep `--rerank-threshold` (reranking is on by default, so this is the one that matters) against the gold set and pick the point that minimises false-abstention while keeping false-answer at/near zero. `average_idf` as the "strong anchor" bar for the safety net is also a heuristic worth revisiting as the corpus grows.

### Round 4 (2026-07-23) — type-aware, model-agnostic ingestion (`ingest.py`)

Groundwork for corpus expansion. A new `ingest.py` replaces the FAQ-only pipeline with two handlers feeding one unified schema, so FAQs and articles are retrieved uniformly.

- **FAQ handler:** reuses the original `ingest_faq.py` Q/A parser (kept intact for back-compat). One chunk per Q/A, question and answer together.
- **Article handler:** splits on the document's own headings (docx Heading styles or markdown `#`), then splits long sections into fixed-size (~250-word, configurable) child chunks with overlap, never breaking mid-sentence. Each child carries a heading breadcrumb (`Hormone Therapy > Side Effects > Bone health`) and a link to its parent section text (`parent_id`, `parent_text`) for future parent-document retrieval. This follows the current evidence: heading-aligned splitting plus fixed-size children, not costly semantic chunking.
- **Unified schema:** `{id, source_id, source_type, lang, section, heading_path, question, answer, parent_id, parent_text, q_paraphrases}`. A compatible superset of the old schema, so the retriever reads it unchanged. For article chunks the heading breadcrumb is the `question` (topic) side of the index and the chunk text is the `answer` body.
- **Model-agnostic:** `--embedding-model` takes any SentenceTransformer (mpnet, BGE-M3, e5, ...). `--passage-prefix`/`--query-prefix` support instruction-tuned models (e.g. e5's `passage: `/`query: `); the query prefix is stored in the payload and applied by rag_core at query time. Per-language artifacts keep the exact filenames rag_core loads, so swapping models is a re-ingest plus an eval re-calibration, nothing more.
- **rag_core adjustments:** the FAQ artifact filter now applies only to `source_type == "faq"` items (article breadcrumbs are not expected to be well-formed questions); the query encoder prepends the stored `query_prefix`. Legacy pickles without these keys default cleanly (`source_type` -> faq, prefix -> "").

Sources are declared via a JSON manifest or convenience flags. Example manifest:

```json
[
  {"path": "faq_adjuvant_en.docx",   "type": "faq",     "lang": "en"},
  {"path": "faq_adjuvant_fr.docx",   "type": "faq",     "lang": "fr"},
  {"path": "articles/bone_health.md","type": "article", "lang": "en"},
  {"path": "articles/nutrition_fr.docx","type": "article","lang": "fr"}
]
```

```
# rebuild indexes with the default model
python ingest.py --manifest manifest.json

# A/B a different embedding model through the same eval harness
python ingest.py --manifest manifest.json \
  --embedding-model BAAI/bge-m3 --out-prefix kb_bgem3
```

Verified the parsing/chunking/schema logic on sample text (embedding model not runnable in this environment): chunker respects size and overlap and never splits mid-sentence, nested heading breadcrumbs and parent links are built, the unified schema is emitted for both types, and article items correctly bypass the FAQ artifact filter while FAQ fragments are still dropped. Full embedding + retrieval must be run and re-calibrated locally after choosing a model.

### Round 5 (2026-07-23) — shared multilingual space + cross-lingual fallback

Moves retrieval from separate per-language indexes to one shared multilingual embedding space, with same-language answers preferred and cross-lingual used only as a recall fallback.

- **Shared index (`ingest.py`):** in addition to the per-language `kb_<lang>_*` artifacts, ingestion now builds a combined `kb_all_*` index across every language (one multilingual model, one FAISS/BM25, every item carrying its own `lang` tag). Toggle with `--shared/--no-shared` and `--per-language/--no-per-language`.
- **Per-language IDF, one corpus (`rag_core.py`):** the retriever now computes IDF/anchor statistics per language and keys them on the active query language, so a mixed FR/EN corpus still yields clean, language-correct anchors. `RetrievalCandidate` carries `lang`.
- **Same-language first, cross-lingual fallback:** the accept decision runs first over same-language candidates; only if that abstains does it retry over the other language. A cross-lingual answer is prefixed with a localized notice ("This information is only available in French." / "Cette information n'est disponible qu'en anglais.") and the FAQ content is quoted verbatim, never machine-translated, preserving the no-fabrication contract. Per-language mode is unchanged (the cross set is empty).
- **App wiring (`hormonai_app.py`):** loads the shared index and sets the query language per request; falls back to per-language indexes automatically if `kb_all_*` has not been built yet.
- **Model-agnostic still holds:** the shared space uses the one `--embedding-model`; different models per language would break the shared space and are explicitly avoided.

Verified on a synthetic bilingual corpus (embedding model not runnable here): same-language queries answer without a notice; a query whose only relevant content is in the other language answers via the cross-lingual fallback with the correct notice; genuinely out-of-scope queries abstain in both languages; per-language mode is unaffected; and per-language IDF buckets are language-correct (a French stem is "in corpus" for French, an English stem is not).

Operational notes: this needs a re-ingest to produce `kb_all_*` (`python ingest.py --manifest manifest.json`). Thresholds remain model-specific and must be re-calibrated after any model swap. The gold eval set currently uses per-language positional `gold_idx`, so evaluating the cross-lingual path will need gold IDs keyed by item id rather than position (follow-up).

### Chat display name

The chat box now shows "Mona" (chat panel title and bot message label). The tool/app name remains HormonAI everywhere else (page title, header, logo, about text).

### Round 6 (2026-07-25) — full corpus, calibration hooks, and answer generation

After ingesting the full corpus (414 items: 140 FAQ + 274 article chunks, BGE-M3, 1024-dim), the enlarged/bilingual index exposed several things that only appear at scale:

- **Recall:** `top_k` raised 12 → 40 (the fused pool was too small for a 414-item bilingual corpus; cross-lingual neighbours were crowding out same-language candidates).
- **Uncalibrated gate:** the accept thresholds were mismatched to the deployed models. `rerank_accept_threshold` default 0.0 → -1.0 (the mmarco cross-encoder scores relevant-but-indirect passages ~ -1..+1). The lexical safety net was also dead: `strong_anchors` used `idf >= average_idf`, which collapses to empty once article vocabulary inflates the average; it now uses a document-frequency cap (`df <= 0.15 * corpus_size`), which is corpus-size-robust.
- **Stronger reranker:** the reranker model is now an env var (`HORMONAI_RERANK_MODEL`). The recommended upgrade is `BAAI/bge-reranker-v2-m3` (multilingual, distilled from BGE-M3, Apache-2.0), which ranks lay/clinical questions far better than the mmarco MiniLM. Threshold, embedding model, dense floor, and sem threshold are all env vars now (`HORMONAI_*`), read by CLI, GUI, and the eval harness, so a calibrated config applies everywhere.
- **Multi-source answers:** the bundle now returns up to 3 relevant sources (not just the reranker's #1), so complementary entries appear together. `AnswerResult.source_indices` exposes the full bundle.
- **Answer generation:** with `--use-llm`, the LLM now performs GROUNDED REPHRASING — it rewrites only the retrieved source text for clarity/empathy, adds no facts, keeps safety caveats, and is told to answer the question generally rather than centre a scenario (e.g. pregnancy) the user did not ask about. Verbatim sources are always cited; if the LLM is unreachable it falls back to the verbatim answer. Without `--use-llm`, output is verbatim quotes as before. (This restores the pre-427c952 "rephrase" behaviour, but grounded + cited.)
- **Known-hard case:** the "take a break … increase my risk?" query still tends to surface the pregnancy item, because it lexically collides with idx `...:41` "Does pregnancy increase the risk of recurrence?" (same "increase risk" phrase, different subject). This is a lexical false-friend, not a gate bug; the reranker upgrade + general-answer prompt mitigate it. Query expansion is the next lever (retrieval-only, so no fabrication risk) but should be validated on the gold set, not one query.

Naming / wording cleanup:

- **Data-file prefix renamed `faq_*` → `kb_*`** (`kb_all_*`, `kb_<lang>_*`). Reflects that the knowledge base is articles + FAQs, not a single FAQ. `ingest.py --out-prefix` defaults to `kb`; `rag_core` loads `kb_all` / `kb_<lang>`.
- **User-facing "FAQ" wording removed** where it referred to the whole knowledge base: the answer preface, source labels ("— Source:"), the rephrase preface, the abstain message, and the Streamlit About/sidebar text now say "knowledge base" / "available information" / "sources". "FAQ" is retained only where accurate (the document *type* in the manifest, source filenames).

Gold set:

- **Re-keyed to stable item IDs** (`gold_ids`, e.g. `20250613_FAQ_Hormono_EN:21`) instead of positional `gold_idx`, so it survives re-ingestion. All 62 gold IDs verified against the current corpus.
- The harness now also reports **lead-source correctness** (did the answer headline a gold source?) and **gold-source-in-bundle**, so retrieval quality is measurable, not anecdotal.

### Gold eval set — run before shipping any change

`tests/eval_set.jsonl` holds 53 labelled cases (EN + FR) across categories: paraphrase, synonym, multi-concept, filler, drug-only, emotional, stats, duration, and out-of-scope, each with expected answer/abstain behaviour and gold FAQ indices. `tests/eval_retrieval.py` runs them through the real pipeline and reports recall@k, decision accuracy, false-abstention rate, false-answer rate, lead-source correctness, and per-category/per-language breakdowns; it exits non-zero if any core case regresses (CI-friendly).

```
# in the project venv, from the repo root:
python tests/eval_retrieval.py                       # full run, both languages
python tests/eval_retrieval.py --verbose             # per-case detail
python tests/eval_retrieval.py --sem-threshold 0.58 --coverage-fraction 0.5   # sweep the gate
python tests/eval_retrieval.py --json eval_results.json
```

**Calibration required:** `sem_accept_threshold=0.62` is a reasonable starting point for `paraphrase-multilingual-mpnet-base-v2` but was NOT empirically tuned (the model wasn't runnable in-session). Sweep `--sem-threshold` (e.g. 0.50–0.75) against the eval set and pick the operating point that drives false-abstention down while keeping false-answer at/near zero — that trade-off is a clinical decision, so review it before deploying.

**Known limitations surfaced by the eval (tagged `hard`):** the strict stats gate (`_passes_stats_gate`) still won't *route* `en-stat-01/02` as grounded statistics, because it requires a stats *marker word* and those answers (idx 0 = 40% recurrence reduction; idx 30 = 20% depression) only carry a `%`. After Round 2 these queries no longer abstain — they fall through and answer qualitatively from the entry that happens to contain the stat — but the number is not being recognised/surfaced *as* a statistic. Loosening `_passes_stats_gate` to accept "number/percent + matched concept" remains a sensible follow-up, deliberately out of this pass. `tests/test_retreival.py` is stale (references file paths that no longer exist) and is superseded by `tests/eval_retrieval.py`.

## Appendix: verification commands

Root cause was reproduced by tracing `extract_core_keywords`/stemming against `data/kb_en_qa.pkl`. Corpus stem-membership counts (`ever`=0, `break`={22,66}, `pause`={42,43}, `stop`={3,35,37,43,44}) and the idx 22 answer text ("A therapeutic break") were read directly from the pickled items. The macOS venv in `ht_faq_rag/` does not run in a Linux sandbox, so the model-loading path was not executed; findings on `answer_query` control flow are from source reading, and the keyword/coverage findings are from direct execution of the equivalent pure-Python logic.
