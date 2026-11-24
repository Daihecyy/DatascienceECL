import json
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.sparse import find
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)


def extract_texts(
    corpus: dict[str, dict], keys: tuple[str, ...] = ("title",)
) -> list[str]:
    texts = [
        " ".join(document[key].strip() for key in keys) for document in corpus.values()
    ]
    return texts


def print_feats(v, features, top_n=30):
    # cette fonction permet d'afficher une "jolie" représentation du vecteur v
    # ARGS :
    #   v : le vecteur à afficher (par ex. une ligne de la matrice X)
    #   features : le vocabulaire
    #   top_n : le nombre de mots maximum à afficher
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame(
        {"word": [t[2] for t in top_feats], "value": [t[1] for t in top_feats]}
    )


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_corpus(file_path: str = "corpus.jsonl") -> dict[str, dict]:
    """
    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    lines = read_jsonl(file_path)
    return {record.pop("_id"): record for record in lines}


def load_queries(file_path: str = "queries.jsonl") -> dict[str, dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    lines = read_jsonl(file_path)
    return {record.pop("_id"): record for record in lines}


def load_qrels(file_path: str = "valid.tsv") -> dict[str, dict[str, int]]:
    """
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    with open(file_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rows = [line.split("\t") for line in lines[1:] if line.strip()]

    qrels = {}
    for query_id, corpus_id, score in rows:
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = int(score)

    return dict(qrels)
