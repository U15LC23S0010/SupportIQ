from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CORPUS_PATH = Path(
    "results/tables/tesco_historical_customer_response_corpus.csv"
)

MODEL_DIR = Path("results/models")

VECTORIZER_PATH = (
    MODEL_DIR / "tesco_retrieval_tfidf_vectorizer.joblib"
)

MATRIX_PATH = (
    MODEL_DIR / "tesco_retrieval_tfidf_matrix.joblib"
)

RECORDS_PATH = (
    MODEL_DIR / "tesco_retrieval_records.joblib"
)

METRICS_PATH = (
    Path("results/tables")
    / "tesco_retrieval_index_metrics.json"
)


def main() -> None:
    print("=" * 80)
    print("SUPPORTIQ — BUILD TESCO RETRIEVAL INDEX")
    print("=" * 80)

    corpus = pd.read_csv(CORPUS_PATH)

    required_columns = [
        "conversation_id",
        "customer_tweet_id",
        "customer_author_id",
        "customer_created_at",
        "customer_message",
        "tesco_tweet_id",
        "tesco_created_at",
        "tesco_response",
    ]

    missing = [
        column
        for column in required_columns
        if column not in corpus.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        f"Historical response pairs: "
        f"{len(corpus):,}"
    )

    corpus = corpus.dropna(
        subset=[
            "customer_message",
            "tesco_response",
        ]
    ).copy()

    corpus["customer_message"] = (
        corpus["customer_message"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    corpus["tesco_response"] = (
        corpus["tesco_response"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    corpus = corpus[
        (corpus["customer_message"] != "")
        & (corpus["tesco_response"] != "")
    ].copy()

    print(
        f"Usable retrieval records: "
        f"{len(corpus):,}"
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=80_000,
    )

    print("\nFitting retrieval TF-IDF index...")

    matrix = vectorizer.fit_transform(
        corpus["customer_message"]
    )

    print(
        f"Index matrix: "
        f"{matrix.shape[0]:,} x {matrix.shape[1]:,}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("results/tables").mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    joblib.dump(
        matrix,
        MATRIX_PATH,
    )

    records = corpus[
        [
            "conversation_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_message",
            "tesco_tweet_id",
            "tesco_created_at",
            "tesco_response",
        ]
    ].reset_index(drop=True)

    joblib.dump(
        records,
        RECORDS_PATH,
    )

    metrics = {
        "source": str(CORPUS_PATH),
        "records": len(records),
        "features": matrix.shape[1],
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.98,
        "sublinear_tf": True,
        "max_features": 80000,
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL INDEX SAVED")
    print("=" * 80)

    print(
        VECTORIZER_PATH.resolve()
    )

    print(
        MATRIX_PATH.resolve()
    )

    print(
        RECORDS_PATH.resolve()
    )

    print(
        METRICS_PATH.resolve()
    )


if __name__ == "__main__":
    main()