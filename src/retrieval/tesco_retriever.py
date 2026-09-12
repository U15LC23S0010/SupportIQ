from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


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


class TescoHistoricalRetriever:
    def __init__(
        self,
        vectorizer_path: Path = VECTORIZER_PATH,
        matrix_path: Path = MATRIX_PATH,
        records_path: Path = RECORDS_PATH,
    ) -> None:

        self.vectorizer = joblib.load(
            vectorizer_path
        )

        self.matrix = joblib.load(
            matrix_path
        )

        self.records = joblib.load(
            records_path
        )

    def retrieve(
        self,
        message: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:

        if not message or not message.strip():
            return []

        query_vector = self.vectorizer.transform(
            [message]
        )

        scores = self.matrix @ query_vector.T

        scores = np.asarray(
            scores.toarray()
        ).ravel()

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results: list[dict[str, Any]] = []

        seen_cases: set[tuple[str, str]] = set()

        for index in ranked_indices:

            score = float(scores[index])

            if score < min_score:
                break

            record = self.records.iloc[index]

            conversation_id = str(
                record["conversation_id"]
            )

            customer_tweet_id = str(
                record["customer_tweet_id"]
            )

            case_key = (
                conversation_id,
                customer_tweet_id,
            )

            if case_key in seen_cases:
                continue

            seen_cases.add(case_key)

            results.append(
                {
                    "similarity": score,
                    "conversation_id": conversation_id,
                    "customer_tweet_id": customer_tweet_id,
                    "customer_message": str(
                        record["customer_message"]
                    ),
                    "tesco_tweet_id": str(
                        record["tesco_tweet_id"]
                    ),
                    "tesco_response": str(
                        record["tesco_response"]
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results