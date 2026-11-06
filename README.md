# RAG Context Retrieval: TF-IDF vs Term-Frequency Baseline

This project evaluates the contribution of Inverse Document Frequency (IDF) weighting in a Retrieval-Augmented Generation (RAG) pre-retrieval stage. It compares a standard TF-IDF cosine-similarity retriever against a raw term-frequency (CountVectorizer) baseline on a seeded, synthetic document corpus.

The goal is to isolate the statistical properties of IDF weighting without invoking any Large Language Model (LLM) or embedding model. All scoring is deterministic and based on standard information retrieval metrics.

## Problem Statement

In RAG pipelines, the quality of context retrieval directly impacts downstream generation. While TF-IDF is a common baseline for lexical retrieval, its advantage over simple term-frequency matching is not guaranteed, especially when documents share significant vocabulary or when the corpus is small. This project provides a reproducible benchmark to measure Recall@k and Mean Reciprocal Rank (MRR) for both approaches under controlled conditions.

## Implementation

The project consists of four core Python modules:

1.  **`data.py`**: Generates a deterministic synthetic corpus of 200 documents and a set of queries. Each query is associated with a single ground-truth document index. The corpus is built from 8 latent topics with controlled overlap.
2.  **`app.py`**: Implements the retrieval experiment.
    *   **Algorithm**: `TfidfVectorizer` fitted on the corpus, followed by cosine similarity.
    *   **Baseline**: `CountVectorizer` (raw term frequency) fitted on the corpus, followed by cosine similarity.
    *   **Splitting**: Queries are split 75/25 into development and test sets.
    *   **Hyperparameter Selection**: The optimal `k` (top-k retrieval depth) is selected from `{3, 5, 7}` using MRR on the development set for the TF-IDF algorithm.
    *   **Evaluation**: Final Recall@k and MRR are reported on the test set for both the algorithm and the baseline.
3.  **`test_project.py`**: Pytest suite verifying data generation, error handling, and metric consistency.
4.  **`main.py`**: Entry point for running the experiment and saving results to JSON.

## Architecture

```
Project Root
├── data.py          # Synthetic data generation
├── app.py           # Retrieval logic and experiment runner
├── test_project.py  # Automated tests
├── main.py          # CLI entry point
├── requirements.txt # Dependencies
└── README.md        # This file
```

### Data Flow

1.  **Generation**: `make_dataset(seed, n_samples)` creates `X` (query strings) and `y` (ground-truth doc indices).
2.  **Corpus Construction**: `_build_corpus(seed)` in `app.py` reconstructs the document corpus deterministically to match the seed used for queries.
3.  **Vectorization**: Both TF-IDF and Count vectorizers are fitted **only** on the corpus to prevent data leakage from queries.
4.  **Retrieval**: Cosine similarity is computed between query vectors and corpus vectors. Top-k indices are extracted.
5.  **Metric Calculation**: Recall@k and MRR are computed by comparing retrieved indices against ground truth.

## Synthetic Dataset Assumptions

*   **Corpus Size**: 200 documents.
*   **Vocabulary**: 8 latent topics, each with 12 unique words, plus 20 shared words.
*   **Document Composition**: Each document contains 15–30 words, drawn 70% from its primary topic, 20% from a secondary topic, and 10% from shared vocabulary.
*   **Queries**: Each query contains 5–10 words, drawn 80% from the ground-truth document's primary topic and 20% from shared vocabulary.
*   **Ground Truth**: Each query is labeled with exactly one document index. Note that other documents from the same topic may also be semantically relevant, but the metric only measures recovery of the specific sampled label.

## Algorithm vs. Baseline

| Component | Algorithm (TF-IDF) | Baseline (Term-Frequency) |
| :--- | :--- | :--- |
| **Vectorizer** | `TfidfVectorizer` | `CountVectorizer` |
| **Weighting** | Term Frequency × Inverse Document Frequency | Raw Term Count |
| **Similarity** | Cosine Similarity | Cosine Similarity |
| **Role** | Proposed method to evaluate | Reference point |

**Note on Performance**: The baseline may outperform the algorithm in certain scenarios, particularly when IDF weighting suppresses important shared terms or when the corpus is small. The project does not assume TF-IDF superiority; it measures the actual difference.

## Metrics

*   **Recall@k**: Fraction of queries where the ground-truth document appears in the top-k retrieved results. Range: `[0, 1]`. Higher is better.
*   **Mean Reciprocal Rank (MRR)**: Average of the reciprocal of the rank of the first correct result for each query. Range: `[0, 1]`. Higher is better.

Both metrics are reported for the test set using the `k` selected on the development set.

## Limitations

1.  **Synthetic Data**: The corpus is artificially generated. Real-world text exhibits noise, typos, and complex semantic structures not captured here.
2.  **Single Ground Truth**: Each query has only one labeled relevant document. In practice, multiple documents may be relevant.
3.  **Lexical Matching Only**: The methods rely on exact term overlap. They do not capture semantic similarity (e.g., "car" vs. "automobile").
4.  **No LLM Evaluation**: This project evaluates retrieval quality only. It does not measure the impact of retrieved context on downstream LLM generation quality.
5.  **Small Corpus**: With 200 documents, IDF values may be unstable compared to larger real-world corpora.

## Reproducibility

The project is fully deterministic. Given the same `seed` and `n_samples`, the output will be identical across runs and platforms.

### Setup

Requires Python 3.11, 3.12, or 3.13.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

This command runs the experiment and saves the metrics to `results.json`.

### Running Tests

```bash
python -m pytest -q
```

The test suite verifies:
*   Different seeds produce different query arrays.
*   Invalid `n_samples` (< 32) raises a `ValueError`.
*   Metric calculations match independently recomputed values.

## References

*   See `validation_report.json` for detailed test results.
*   See `example_results.json` for a sample output of the experiment.

## License

This project is provided for educational purposes. No license is specified.

## Recorded automated validation

Host contract tests and project tests passed (13 tests, 0 skipped). Demo completed on Python 3.13.16. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
