"""app.py - RAG context retrieval: TF-IDF vs term-frequency baseline.

This module implements a deterministic retrieval-quality benchmark that compares
TF-IDF cosine-similarity retrieval against a raw term-frequency (CountVectorizer)
baseline for RAG context selection.

Method:
- A synthetic corpus of 200 documents is generated (via data.make_dataset's
  underlying logic, but here we regenerate the corpus deterministically from the
  same seed used for queries).
- n_samples queries are generated, each with a known ground-truth document.
- Queries are split 75/25 into dev and test sets (deterministic split by index).
- Algorithm: TfidfVectorizer fitted on corpus + cosine_similarity, top-k retrieval.
- Baseline: CountVectorizer (no IDF) + cosine_similarity, same retrieval.
- Dev split selects optimal k in {3, 5, 7} by MRR for the algorithm.
- Test reports final Recall@k and MRR for both algorithm and baseline.

No LLM is invoked; all scoring is statistical.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import make_dataset


def _build_corpus(seed: int) -> list[str]:
    """
    Rebuild the synthetic corpus deterministically from the same seed.

    This mirrors the corpus-generation logic in data.py so that the corpus
    is consistent with the queries produced by make_dataset(seed).
    """
    rng = np.random.default_rng(seed)

    n_topics = 8
    words_per_topic = 12
    n_shared = 20

    topic_words: list[list[str]] = []
    for t in range(n_topics):
        words = [f"topic{t}_w{i}" for i in range(words_per_topic)]
        topic_words.append(words)

    shared_words = [f"shared_{i}" for i in range(n_shared)]

    n_docs = 200
    docs: list[str] = []
    doc_topics: list[int] = []

    for d in range(n_docs):
        primary_topic = rng.integers(0, n_topics)
        doc_topics.append(primary_topic)

        n_words = rng.integers(15, 31)
        doc_words: list[str] = []

        for _ in range(n_words):
            r = rng.random()
            if r < 0.7:
                doc_words.append(rng.choice(topic_words[primary_topic]))
            elif r < 0.9:
                secondary = rng.integers(0, n_topics)
                while secondary == primary_topic:
                    secondary = rng.integers(0, n_topics)
                doc_words.append(rng.choice(topic_words[secondary]))
            else:
                doc_words.append(rng.choice(shared_words))

        docs.append(" ".join(doc_words))

    return docs


def _retrieve_topk(
    query_matrix: np.ndarray,
    corpus_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Retrieve top-k document indices for each query using cosine similarity.

    Parameters
    ----------
    query_matrix : np.ndarray, shape (n_queries, n_features)
    corpus_matrix : np.ndarray, shape (n_docs, n_features)
    k : int

    Returns
    -------
    np.ndarray, shape (n_queries, k)
        Indices of top-k retrieved documents for each query.
    """
    sims = cosine_similarity(query_matrix, corpus_matrix)
    # argsort in descending order, take top-k
    topk_indices = np.argsort(-sims, axis=1)[:, :k]
    return topk_indices


def _recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@k: fraction of queries where the ground-truth doc is in top-k.

    Parameters
    ----------
    retrieved : np.ndarray, shape (n_queries, k)
    ground_truth : np.ndarray, shape (n_queries,)
    k : int

    Returns
    -------
    float in [0, 1]
    """
    n = len(ground_truth)
    hits = 0
    for i in range(n):
        if ground_truth[i] in retrieved[i, :k]:
            hits += 1
    return hits / n


def _mrr(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over top-k results.

    Parameters
    ----------
    retrieved : np.ndarray, shape (n_queries, k)
    ground_truth : np.ndarray, shape (n_queries,)
    k : int

    Returns
    -------
    float in [0, 1]
    """
    n = len(ground_truth)
    reciprocal_ranks = []
    for i in range(n):
        gt = ground_truth[i]
        rank = None
        for pos in range(k):
            if retrieved[i, pos] == gt:
                rank = pos + 1
                break
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """
    Run the TF-IDF vs term-frequency retrieval benchmark.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of queries. Must be >= 32.

    Returns
    -------
    dict
        JSON-serializable dict with:
        - n_samples: int
        - metrics: dict[str, float] (algorithm: recall_at_k, mrr)
        - baseline_metrics: dict[str, float] (baseline: recall_at_k, mrr)
        - explanation: str
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    # Generate queries and ground truth
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    # Build corpus (deterministic, same seed)
    corpus = _build_corpus(seed)
    n_docs = len(corpus)

    # Split queries 75/25 into dev and test (deterministic by index)
    n_dev = int(0.75 * n_samples)
    dev_indices = np.arange(n_dev)
    test_indices = np.arange(n_dev, n_samples)

    X_dev = X[dev_indices]
    y_dev = y[dev_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Fit vectorizers on corpus only (no leakage from queries)
    tfidf_vec = TfidfVectorizer()
    tfidf_corpus = tfidf_vec.fit_transform(corpus)

    count_vec = CountVectorizer()
    count_corpus = count_vec.fit_transform(corpus)

    # Transform queries
    tfidf_queries = tfidf_vec.transform(X_dev)
    count_queries_dev = count_vec.transform(X_dev)

    tfidf_queries_test = tfidf_vec.transform(X_test)
    count_queries_test = count_vec.transform(X_test)

    # Candidate k values
    k_candidates = [3, 5, 7]

    # --- Dev split: select optimal k by MRR for the algorithm (TF-IDF) ---
    best_k = k_candidates[0]
    best_dev_mrr = -1.0

    for k in k_candidates:
        retrieved_dev = _retrieve_topk(tfidf_queries, tfidf_corpus, k)
        mrr_val = _mrr(retrieved_dev, y_dev, k)
        if mrr_val > best_dev_mrr:
            best_dev_mrr = mrr_val
            best_k = k

    # --- Test split: report final metrics ---
    # Algorithm (TF-IDF)
    retrieved_test_algo = _retrieve_topk(tfidf_queries_test, tfidf_corpus, best_k)
    algo_recall = _recall_at_k(retrieved_test_algo, y_test, best_k)
    algo_mrr = _mrr(retrieved_test_algo, y_test, best_k)

    # Baseline (CountVectorizer)
    retrieved_test_base = _retrieve_topk(count_queries_test, count_corpus, best_k)
    base_recall = _recall_at_k(retrieved_test_base, y_test, best_k)
    base_mrr = _mrr(retrieved_test_base, y_test, best_k)

    explanation = (
        f"TF-IDF retrieval with k={best_k} selected on dev set (MRR={best_dev_mrr:.4f}). "
        f"Test set: algorithm Recall@{best_k}={algo_recall:.4f}, MRR={algo_mrr:.4f}; "
        f"baseline Recall@{best_k}={base_recall:.4f}, MRR={base_mrr:.4f}. "
        f"Observed MRR difference (TF-IDF minus baseline): {algo_mrr - base_mrr:+.6f}. "
        f"A negative difference favors the baseline. IDF does not guarantee improvement. "
        f"Labels identify one sampled document, although other same-topic documents may also "
        f"be relevant. Scores measure recovery of that sampled label, not complete relevance."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": {
            "recall_at_k": float(algo_recall),
            "mrr": float(algo_mrr),
        },
        "baseline_metrics": {
            "recall_at_k": float(base_recall),
            "mrr": float(base_mrr),
        },
        "explanation": explanation,
    }
