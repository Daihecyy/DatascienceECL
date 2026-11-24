import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_texts, load_corpus, load_qrels, load_queries

np.random.seed(42)

print("Loading dataset...")
corpus = load_corpus("corpus.jsonl")
queries = load_queries("queries.jsonl")
qrels_valid = load_qrels("valid.tsv")


print(f"Loaded {len(corpus)} documents in corpus")
print(f"Loaded {len(queries)} queries")
print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")


doc_ids = list(corpus.keys())
corpus_texts = extract_texts(corpus, keys=("title",))

query_ids = list(queries.keys())
query_texts = extract_texts(queries, keys=("text",))

vectorizer = CountVectorizer()
X_corpus = vectorizer.fit_transform(corpus_texts)
X_queries = vectorizer.transform(query_texts)

similarity_matrix = cosine_similarity(X_queries, X_corpus)

top_k = 10
i = 0
for q_idx, query_id in enumerate(query_ids):
    if i > 5:
        break
    scores = similarity_matrix[q_idx]
    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"=== Requête : {query_id}, {query_texts[q_idx]!r} ===")
    for rank, doc_idx in enumerate(top_indices, start=1):
        doc_id = doc_ids[doc_idx]
        title = corpus[doc_id].get("title", "")
        print(f"{rank:2d}. [{doc_id}] {title} — score cosinus : {scores[doc_idx]:.4f}")
    print()
