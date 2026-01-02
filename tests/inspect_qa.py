import pickle
from pprint import pprint

LANGUAGE = "en"  # or "fr"
QA_PATH = f"faq_qa_{LANGUAGE}.pkl"

with open(QA_PATH, "rb") as f:
    data = pickle.load(f)

items = data["items"]
print(f"Loaded {len(items)} Q/A items from {QA_PATH}\n")

# Show the first 3 Q/A entries
for i, item in enumerate(items[:3]):
    print(f"--- Q/A #{i} ---")
    print(f"Section : {item['section']}")
    print(f"Question: {item['question']}")
    print(f"Answer  : {item['answer'][:400]}...")
    print()
