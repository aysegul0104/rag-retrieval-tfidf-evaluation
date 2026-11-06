"""test_project.py - Tests for RAG context retrieval: TF-IDF vs term-frequency baseline.

These tests verify the core behaviors of the project:
1. Different seeds produce different query arrays.
2. Invalid n_samples raises ValueError.
3. Ranking and metric calculations match independently computed expected results.
"""

from __future__ import annotations

import numpy as np
import pytest

from data import make_dataset
from app import run_experiment


class TestMakeDataset:
    """Tests for the synthetic data generation function."""

    def test_different_seeds_produce_different_queries(self):
        """Different seeds must yield different query arrays."""
        X1, _ = make_dataset(seed=42, n_samples=64)
        X2, _ = make_dataset(seed=123, n_samples=64)
        # Arrays should differ in at least one element
        assert not np.array_equal(X1, X2)

    def test_invalid_n_samples_raises_value_error(self):
        """n_samples < 32 must raise ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 32"):
            make_dataset(seed=42, n_samples=16)

    def test_n_samples_respected(self):
        """Returned arrays must have length equal to n_samples."""
        X, y = make_dataset(seed=7, n_samples=128)
        assert len(X) == 128
        assert len(y) == 128


class TestRunExperiment:
    """Tests for the retrieval benchmark experiment."""

    def test_output_structure(self):
        """run_experiment must return a dict with required keys and finite metrics."""
        result = run_experiment(seed=42, n_samples=64)
        assert isinstance(result, dict)
        assert result["n_samples"] == 64
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["baseline_metrics"], dict)
        assert len(result["metrics"]) > 0
        assert len(result["baseline_metrics"]) > 0
        for key, val in result["metrics"].items():
            assert np.isfinite(val), f"Metric {key} is not finite"
        for key, val in result["baseline_metrics"].items():
            assert np.isfinite(val), f"Baseline metric {key} is not finite"

    def test_ranking_metrics_match_hand_computed_values(self):
        from app import _mrr, _recall_at_k
        retrieved = np.array([[4, 8, 2], [7, 3, 1], [5, 6, 9], [2, 1, 0]])
        truth = np.array([4, 3, 9, 8])
        # Ranks: 1, 2, 3, absent. Truncation must exclude later hits.
        assert _mrr(retrieved, truth, 3) == pytest.approx((1 + 1/2 + 1/3) / 4)
        assert _recall_at_k(retrieved, truth, 3) == pytest.approx(3/4)
        assert _mrr(retrieved, truth, 1) == pytest.approx(1/4)
        assert _recall_at_k(retrieved, truth, 1) == pytest.approx(1/4)

    def test_cosine_retrieval_on_known_vectors(self):
        from app import _retrieve_topk
        corpus = np.array([[1., 0.], [0., 1.], [1., 1.]])
        queries = np.array([[1., 0.], [0., 1.]])
        expected = np.array([[0, 2], [1, 2]])
        np.testing.assert_array_equal(_retrieve_topk(queries, corpus, 2), expected)

    def test_reported_metrics_match_independent_evaluation(self):
        from app import _build_corpus
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        queries, truth = make_dataset(seed=42, n_samples=256)
        corpus = _build_corpus(42)
        rankings = []
        for vectorizer in (TfidfVectorizer(), CountVectorizer()):
            documents = vectorizer.fit_transform(corpus)
            rankings.append(np.argsort(-cosine_similarity(vectorizer.transform(queries), documents), axis=1))
        def score(rows, labels, k):
            ranks = [next((i + 1 for i, doc in enumerate(row[:k]) if doc == label), 0)
                     for row, label in zip(rows, labels)]
            return sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), sum(rank > 0 for rank in ranks) / len(ranks)
        k = max([3, 5, 7], key=lambda k: score(rankings[0][:192], truth[:192], k)[0])
        result = run_experiment(seed=42, n_samples=256)
        for field, ranking in zip(("metrics", "baseline_metrics"), rankings):
            mrr, recall = score(ranking[192:], truth[192:], k)
            assert result[field]["mrr"] == pytest.approx(mrr)
            assert result[field]["recall_at_k"] == pytest.approx(recall)

    def test_metrics_in_valid_range(self):
        """Recall@k and MRR must be in [0, 1]."""
        result = run_experiment(seed=99, n_samples=128)
        for metrics_dict in (result["metrics"], result["baseline_metrics"]):
            for key, val in metrics_dict.items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] range"

    def test_deterministic_output(self):
        """Same inputs must produce exactly the same output."""
        r1 = run_experiment(seed=42, n_samples=64)
        r2 = run_experiment(seed=42, n_samples=64)
        assert r1["n_samples"] == r2["n_samples"]
        for key in r1["metrics"]:
            assert r1["metrics"][key] == r2["metrics"][key]
        for key in r1["baseline_metrics"]:
            assert r1["baseline_metrics"][key] == r2["baseline_metrics"][key]
