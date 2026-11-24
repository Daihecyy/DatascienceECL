import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_texts, load_corpus, load_qrels, load_queries

np.random.seed(42)
# TODO : at the end, python CLI file with best model for each

print("Loading dataset...")
corpus = load_corpus("corpus.jsonl")
queries = load_queries("queries.jsonl")
qrels_valid = load_qrels("valid.tsv")


print(f"Loaded {len(corpus)} documents in corpus")
print(f"Loaded {len(queries)} queries")
print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")


doc_ids = list(corpus.keys())
doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
corpus_texts = extract_texts(corpus, keys=("title",))

query_ids = list(queries.keys())
query_texts = extract_texts(queries, keys=("text",))

vectorizers = {
    "basic": CountVectorizer(),
    "stop_words": CountVectorizer(stop_words="english"),
    "tf-idf": TfidfVectorizer(stop_words="english"),
}
for vectorizer_name, vectorizer in vectorizers.items():
    X_corpus = vectorizer.fit_transform(corpus_texts)
    X_queries = vectorizer.transform(query_texts)

    similarity_matrix = cosine_similarity(X_queries, X_corpus)

    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
    top_k = 25
    for q_idx, query_id in enumerate(query_ids):
        truth_map = qrels_valid.get(query_id)
        if not truth_map:
            continue

        scores = similarity_matrix[q_idx]
        ranked_indices = np.argsort(scores)[::-1]
        top_k_doc_ids = {doc_ids[i] for i in ranked_indices[:top_k]}

        # Evaluate on candidate docs present in qrels for this query
        candidate_ids = [d_id for d_id in truth_map if d_id in doc_id_to_idx]
        if not candidate_ids:
            continue

        y_true = np.array([truth_map[d_id] for d_id in candidate_ids])
        y_pred = np.array([1 if d_id in top_k_doc_ids else 0 for d_id in candidate_ids])
        y_scores = np.array([scores[doc_id_to_idx[d_id]] for d_id in candidate_ids])

        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) > 1:
            metrics["auc"].append(roc_auc_score(y_true, y_scores))

    print(f"=== {vectorizer_name} ===")
    for name, vals in metrics.items():
        mean_val = np.mean(vals) if vals else float("nan")
        print(f"{name:10s}: {mean_val:.4f}")
