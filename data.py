"""data.py - Synthetic corpus and query generation for RAG retrieval evaluation.

This module creates a deterministic, seeded synthetic document corpus and a set of
queries, each with a known relevant document (ground truth). It supports the
retrieval-quality benchmark described in the project plan.

Assumptions and limitations:
- The corpus is synthetic: 200 documents drawn from a small set of latent topics
  with controlled overlap. No real-world text is used.
- Each query is generated to be relevant to exactly one document (its ground truth),
  by sampling terms from that document's topic distribution.
- The function is pure: no disk I/O, no network, no external data.
- Different seeds produce different corpora and queries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[NDArray, NDArray]:
    """
    Generate a synthetic corpus and query set for retrieval evaluation.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of queries to generate. Must be >= 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples,)
        Array of query strings.
    y : np.ndarray, shape (n_samples,)
        Array of ground-truth document indices (integers in [0, n_docs)).

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    # --- Build a small vocabulary of topic words ---
    # 8 latent topics, each with 12 topic-specific words + 20 shared words
    n_topics = 8
    words_per_topic = 12
    n_shared = 20

    # Generate topic-specific words
    topic_words: list[list[str]] = []
    for t in range(n_topics):
        words = [f"topic{t}_w{i}" for i in range(words_per_topic)]
        topic_words.append(words)

    # Shared words (appear across topics)
    shared_words = [f"shared_{i}" for i in range(n_shared)]

    # --- Generate 200 documents ---
    n_docs = 200
    docs: list[str] = []
    doc_topics: list[int] = []  # primary topic for each doc

    for d in range(n_docs):
        # Assign a primary topic
        primary_topic = rng.integers(0, n_topics)
        doc_topics.append(primary_topic)

        # Sample 15-30 words for the document
        n_words = rng.integers(15, 31)
        doc_words: list[str] = []

        # 70% from primary topic, 20% from a secondary topic, 10% shared
        for _ in range(n_words):
            r = rng.random()
            if r < 0.7:
                doc_words.append(rng.choice(topic_words[primary_topic]))
            elif r < 0.9:
                # Secondary topic (different from primary)
                secondary = rng.integers(0, n_topics)
                while secondary == primary_topic:
                    secondary = rng.integers(0, n_topics)
                doc_words.append(rng.choice(topic_words[secondary]))
            else:
                doc_words.append(rng.choice(shared_words))

        docs.append(" ".join(doc_words))

    # --- Generate queries ---
    # Each query is relevant to one document. We sample terms from that doc's
    # topic to make the query topically aligned.
    queries: list[str] = []
    ground_truth: list[int] = []

    for q in range(n_samples):
        # Pick a random document as ground truth
        gt_doc = rng.integers(0, n_docs)
        ground_truth.append(int(gt_doc))

        # Generate a query of 5-10 words, drawn from the ground-truth doc's topic
        gt_topic = doc_topics[gt_doc]
        n_q_words = rng.integers(5, 11)
        q_words: list[str] = []

        for _ in range(n_q_words):
            r = rng.random()
            if r < 0.8:
                q_words.append(rng.choice(topic_words[gt_topic]))
            else:
                q_words.append(rng.choice(shared_words))

        queries.append(" ".join(q_words))

    X = np.array(queries, dtype=object)
    y = np.array(ground_truth, dtype=np.int64)

    return X, y
