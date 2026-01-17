<!--
SYNC IMPACT REPORT
Version change: [New File] -> 1.0.0
List of modified principles: Initial definition of all principles.
Added sections: Safety & Containment, Transparency & Citation, Language Parity, Empathy & Clarity, Auditability, Hybrid Retrieval.
Templates requiring updates:
- .specify/templates/plan-template.md: ✅ (Generic enough)
- .specify/templates/spec-template.md: ✅ (Generic enough)
- .specify/templates/tasks-template.md: ✅ (Generic enough)
Follow-up TODOs: None.
-->

# hormonAI Constitution

## Core Principles

### I. Safety & Strict Containment
Answers MUST be derived solely from the provided FAQ content. Out-of-scope queries MUST be explicitly declined. Hallucinations or external knowledge injection MUST be prevented to ensure medical safety.

### II. Transparency & Citation
Every response MUST cite its source section and question from the FAQ. The user must know exactly where the information comes from to build trust and allow verification.

### III. Language Parity
Functionality, retrieval quality, and user experience MUST be equivalent for both supported languages (English and French). Updates to FAQs or features MUST be applied to both languages simultaneously.

### IV. Empathy & Clarity
Responses should be empathetic, clear, and accessible, strictly adhering to the tone set by the FAQ or the "empathetic rephrasing" requirements. When using LLM rephrasing, the tone should be supportive but factual, avoiding medical jargon where possible unless explained.

### V. Auditability
All interactions (queries, retrieved sources, generated answers) MUST be logged for quality assurance and medical review. This data is critical for improving the FAQ and ensuring the system performs as expected.

### VI. Hybrid Retrieval
Search MUST utilize both semantic (vector) and keyword (BM25) methods to ensure high recall and precision. Reliance on a single retrieval method is insufficient for the accuracy required in this domain.

## Governance

This constitution supersedes all other development practices. Amendments require documentation, approval, and a clear migration plan.

### Amendment Procedure
1.  Proposal: Any changes to these principles must be proposed via a pull request modifying this file.
2.  Review: Changes must be reviewed by the project lead or core maintainers.
3.  Version Bump: Semantic versioning rules apply (Major for breaking governance changes, Minor for new principles, Patch for clarifications).

### Compliance
All Pull Requests and Code Reviews MUST verify compliance with these principles.
- **Safety**: Verify no external knowledge access.
- **Transparency**: Verify citations are present.
- **Language**: Verify En/Fr parity.

**Version**: 1.0.0 | **Ratified**: 2025-06-13 | **Last Amended**: 2026-01-17