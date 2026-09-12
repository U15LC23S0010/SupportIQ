from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.agent.supportiq_agent import SupportIQAgent


GOLD_PATH = Path(
    "results/tables/tesco_golden_evaluation_set_172.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_agent_gold_predictions.csv"
)

METRICS_PATH = Path(
    "results/tables/supportiq_agent_evaluation_metrics.json"
)

REPORT_PATH = Path(
    "results/tables/supportiq_agent_evaluation_report.txt"
)

HUMAN_REVIEW_PATH = Path(
    "results/tables/supportiq_reply_human_review.csv"
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


def main() -> None:
    print("=" * 80)
    print("SUPPORTIQ — AGENT EVALUATION HARNESS")
    print("=" * 80)

    gold = pd.read_csv(
        GOLD_PATH
    )

    agent = SupportIQAgent()

    rows = []

    for position, row in gold.iterrows():
        message = str(
            row["evaluation_incoming_message"]
        )

        result = agent.analyze(
            message,
            top_k=5,
        )

        evidence = result["evidence"]

        top_similarity = (
            float(evidence[0]["similarity"])
            if evidence
            else 0.0
        )

        rows.append(
            {
                "conversation_id": row[
                    "conversation_id"
                ],
                "evaluation_incoming_tweet_id": row[
                    "evaluation_incoming_tweet_id"
                ],
                "customer_message": message,
                "human_intent": row[
                    "human_intent"
                ],
                "predicted_intent": result[
                    "intent"
                ],
                "intent_confidence": result[
                    "intent_confidence"
                ],
                "evidence_consistency": result[
                    "evidence_consistency"
                ],
                "intent_correct": (
                    str(row["human_intent"])
                    == str(result["intent"])
                ),
                "decision": result[
                    "decision"
                ],
                "decision_reason": result[
                    "decision_reason"
                ],
                "draft_reply": result[
                    "draft_reply"
                ],
                "top_retrieval_similarity": (
                    top_similarity
                ),
                "evidence_count": len(
                    evidence
                ),
                "evidence": json.dumps(
                    evidence,
                    ensure_ascii=False,
                ),
            }
        )

        if (position + 1) % 25 == 0:
            print(
                f"Evaluated {position + 1}/{len(gold)}"
            )

    predictions = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    target = predictions[
        predictions["human_intent"].isin(
            TARGET_INTENTS
        )
    ].copy()

    accuracy = accuracy_score(
        target["human_intent"],
        target["predicted_intent"],
    )

    macro_f1 = f1_score(
        target["human_intent"],
        target["predicted_intent"],
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        target["human_intent"],
        target["predicted_intent"],
        labels=TARGET_INTENTS,
        zero_division=0,
    )

    matrix = confusion_matrix(
        target["human_intent"],
        target["predicted_intent"],
        labels=TARGET_INTENTS,
    )

    auto_handle_count = (
        predictions["decision"]
        == "AUTO_HANDLE"
    ).sum()

    escalate_count = (
        predictions["decision"]
        == "ESCALATE"
    ).sum()

    safety_count = (
        predictions["predicted_intent"]
        == "product_safety_sensitive"
    ).sum()

    safety_escalated = (
        predictions[
            predictions["predicted_intent"]
            == "product_safety_sensitive"
        ]["decision"]
        == "ESCALATE"
    ).sum()

    metrics = {
        "gold_total_rows": len(
            predictions
        ),
        "gold_target_rows": len(
            target
        ),
        "intent_accuracy": float(
            accuracy
        ),
        "intent_macro_f1": float(
            macro_f1
        ),
        "auto_handle_count": int(
            auto_handle_count
        ),
        "escalate_count": int(
            escalate_count
        ),
        "auto_handle_rate": (
            float(auto_handle_count)
            / len(predictions)
        ),
        "escalation_rate": (
            float(escalate_count)
            / len(predictions)
        ),
        "safety_predictions": int(
            safety_count
        ),
        "safety_escalated": int(
            safety_escalated
        ),
        "safety_escalation_rate": (
            float(safety_escalated)
            / safety_count
            if safety_count
            else 0.0
        ),
        "mean_intent_confidence": float(
            predictions[
                "intent_confidence"
            ].mean()
        ),
        "median_intent_confidence": float(
            predictions[
                "intent_confidence"
            ].median()
        ),
        "mean_evidence_consistency": float(
            predictions[
                "evidence_consistency"
            ].mean()
        ),
        "median_evidence_consistency": float(
            predictions[
                "evidence_consistency"
            ].median()
        ),
        "mean_top_retrieval_similarity": float(
            predictions[
                "top_retrieval_similarity"
            ].mean()
        ),
        "median_top_retrieval_similarity": float(
            predictions[
                "top_retrieval_similarity"
            ].median()
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = (
        "SUPPORTIQ — AGENT EVALUATION HARNESS\n"
        + "=" * 80
        + "\n\n"
        + f"Gold rows: {len(predictions)}\n"
        + f"Target-intent rows: {len(target)}\n"
        + f"Intent Accuracy: {accuracy:.4f}\n"
        + f"Intent Macro-F1: {macro_f1:.4f}\n"
        + f"AUTO_HANDLE: {auto_handle_count}\n"
        + f"ESCALATE: {escalate_count}\n"
        + f"AUTO_HANDLE rate: "
        + f"{metrics['auto_handle_rate']:.4f}\n"
        + f"Escalation rate: "
        + f"{metrics['escalation_rate']:.4f}\n"
        + f"Safety predictions: "
        + f"{safety_count}\n"
        + f"Safety escalated: "
        + f"{safety_escalated}\n"
        + f"Safety escalation rate: "
        + f"{metrics['safety_escalation_rate']:.4f}\n"
        + f"Mean intent confidence: "
        + f"{metrics['mean_intent_confidence']:.4f}\n"
        + f"Median intent confidence: "
        + f"{metrics['median_intent_confidence']:.4f}\n"
        + f"Mean evidence consistency: "
        + f"{metrics['mean_evidence_consistency']:.4f}\n"
        + f"Median evidence consistency: "
        + f"{metrics['median_evidence_consistency']:.4f}\n"
        + f"Mean retrieval similarity: "
        + f"{metrics['mean_top_retrieval_similarity']:.4f}\n"
        + f"Median retrieval similarity: "
        + f"{metrics['median_top_retrieval_similarity']:.4f}\n\n"
        + "CLASSIFICATION REPORT\n"
        + "=" * 80
        + "\n"
        + report
        + "\n\n"
        + "CONFUSION MATRIX\n"
        + "=" * 80
        + "\n"
        + pd.DataFrame(
            matrix,
            index=TARGET_INTENTS,
            columns=TARGET_INTENTS,
        ).to_string()
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    review = predictions[
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
    ].copy()

    review["human_reply_quality"] = ""
    review["human_grounded"] = ""
    review["human_handling_decision"] = ""
    review["human_notes"] = ""

    review.to_csv(
        HUMAN_REVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Gold rows: {len(predictions)}"
    )

    print(
        f"Target-intent rows: {len(target)}"
    )

    print(
        f"Intent Accuracy: {accuracy:.4f}"
    )

    print(
        f"Intent Macro-F1: {macro_f1:.4f}"
    )

    print(
        f"AUTO_HANDLE: {auto_handle_count}"
    )

    print(
        f"ESCALATE: {escalate_count}"
    )

    print(
        f"Safety escalation rate: "
        f"{metrics['safety_escalation_rate']:.4f}"
    )

    print(
        f"Mean evidence consistency: "
        f"{metrics['mean_evidence_consistency']:.4f}"
    )

    print()
    print("Artifacts saved:")

    print(
        OUTPUT_PATH.resolve()
    )

    print(
        METRICS_PATH.resolve()
    )

    print(
        REPORT_PATH.resolve()
    )

    print(
        HUMAN_REVIEW_PATH.resolve()
    )


if __name__ == "__main__":
    main()