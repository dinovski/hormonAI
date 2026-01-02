import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LANGUAGE = "en"  # or "fr"
INDEX_PATH = f"faq_index_{LANGUAGE}.faiss"
QA_PATH = f"faq_qa_{LANGUAGE}.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_data():
    print(f"Loading index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)

    print(f"Loading Q/A items: {QA_PATH}")
    with open(QA_PATH, "rb") as f:
        data = pickle.load(f)

    items = data["items"]
    language = data.get("language", LANGUAGE)

    print(f"Language in file: {language}")
    print(f"Number of Q/A items: {len(items)}")

    return index, items, language


def retrieve(query: str, index, items, model, top_k: int = 5):
    print(f"\nQuery: {query}\n")
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)
    scores = scores[0]
    indices = indices[0]

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx == -1:
            continue
        item = items[idx]
        print(f"Rank {rank} | score={score:.3f} | idx={idx}")
        print(f"  Section : {item['section']}")
        print(f"  Question: {item['question']}")
        print(f"  Answer  : {item['answer'][:300]}...")
        print()


def main():
    index, items, language = load_data()

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Try one or more test queries that you know exist in the FAQ
    if language == "fr":
        test_queries = [
            "Quels sont les effets secondaires fréquents de l’hormonothérapie ?",
            "Combien de temps dure l’hormonothérapie adjuvante ?",
        ]
    else:
        test_queries = [
            "What side effects are common with adjuvant hormone therapy?",
            "How long do I need to stay on hormone therapy?",
        ]

    for q in test_queries:
        retrieve(q, index, items, model, top_k=3)


if __name__ == "__main__":
    main()

