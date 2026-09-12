from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "results/tables/supportiq_agent_gold_predictions.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32.csv"
)

TARGET_INTENTS = [
    "delivery_order",
    "pricing_payment",
    "store_staff_service",
    "product_availability",
    "product_quality",
    "product_safety_sensitive",
    "product_information_policy",
    "feedback_suggestion",
]

SAMPLES_PER_INTENT = 4


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    target = df[
        df["human_intent"].isin(TARGET_INTENTS)
    ].copy()

    samples = []

    for intent in TARGET_INTENTS:
        subset = target[
            target["human_intent"] == intent
        ].copy()

        sample_size = min(
            SAMPLES_PER_INTENT,
            len(subset),
        )

        sample = subset.sample(
            n=sample_size,
            random_state=42,
        )

        samples.append(sample)

    review = pd.concat(
        samples,
        ignore_index=True,
    )

    review = review.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    review.insert(
        0,
        "review_id",
        range(1, len(review) + 1),
    )

    review[
        [
            "conversation_id",
            "evaluation_incoming_tweet_id",
            "customer_message",
            "human_intent",
            "predicted_intent",
            "intent_confidence",
            "evidence_consistency",
            "decision",
            "decision_reason",
            "draft_reply",
            "top_retrieval_similarity",
            "evidence",
        ]
    ] = review[
        [
            "conversation_id",
            "evaluation_incoming_tweet_id",
            "customer_message",
            "human_intent",
            "predicted_intent",
            "intent_confidence",
            "evidence_consistency",
            "decision",
            "decision_reason",
            "draft_reply",
            "top_retrieval_similarity",
            "evidence",
        ]
    ]

    review["human_reply_quality"] = ""
    review["human_grounded"] = ""
    review["human_handling_decision"] = ""
    review["human_notes"] = ""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("SUPPORTIQ — HUMAN REPLY REVIEW SAMPLE")
    print("=" * 80)

    print(
        f"Review cases: {len(review)}"
    )

    print()
    print(
        review["human_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"AUTO_HANDLE: "
        f"{(review['decision'] == 'AUTO_HANDLE').sum()}"
    )

    print(
        f"ESCALATE: "
        f"{(review['decision'] == 'ESCALATE').sum()}"
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()