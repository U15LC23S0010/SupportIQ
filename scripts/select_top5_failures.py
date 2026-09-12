from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "results/tables/supportiq_human_review_failure_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_top5_failure_cases.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    df["intent_correct"] = (
        df["predicted_intent"] == df["human_intent"]
    )

    df["intent_confidence"] = pd.to_numeric(
        df["intent_confidence"],
        errors="coerce",
    )

    df["evidence_consistency"] = pd.to_numeric(
        df["evidence_consistency"],
        errors="coerce",
    )

    df["human_reply_quality"] = pd.to_numeric(
        df["human_reply_quality"],
        errors="coerce",
    )

    df["human_grounded"] = pd.to_numeric(
        df["human_grounded"],
        errors="coerce",
    )

    # Prefer failures that demonstrate one of the strongest
    # and most report-worthy failure patterns:
    #
    # 1. wrong intent + strong evidence + poor reply
    # 2. correct intent + strong evidence + poor reply
    # 3. wrong intent + weak evidence
    #
    # We use a transparent ranking rather than inventing examples.

    def priority(row):
        if (
            not row["intent_correct"]
            and row["human_grounded"] == 2
            and row["human_reply_quality"] == 0
        ):
            return 1

        if (
            not row["intent_correct"]
            and row["human_grounded"] == 2
        ):
            return 2

        if (
            row["intent_correct"]
            and row["human_grounded"] == 2
        ):
            return 3

        return 4

    df["priority"] = df.apply(priority, axis=1)

    # Within the same priority, prefer lower intent confidence,
    # because these are stronger examples of classifier uncertainty.
    ranked = df.sort_values(
        [
            "priority",
            "intent_confidence",
            "evidence_consistency",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).copy()

    top5 = ranked.head(5).copy()

    output_columns = [
        "review_id",
        "conversation_id",
        "customer_message",
        "human_intent",
        "predicted_intent",
        "intent_confidence",
        "evidence_consistency",
        "decision",
        "human_handling_decision",
        "human_reply_quality",
        "human_grounded",
        "failure_type",
        "draft_reply",
        "evidence",
        "human_notes",
    ]

    top5 = top5[output_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    top5.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("SUPPORTIQ — TOP 5 FAILURE CASES")
    print("=" * 80)

    for _, row in top5.iterrows():
        print()
        print(
            f"Review ID: {row['review_id']} | "
            f"Failure: {row['failure_type']}"
        )
        print(
            f"Human intent: {row['human_intent']} | "
            f"Predicted: {row['predicted_intent']}"
        )
        print(
            f"Confidence: {row['intent_confidence']:.4f} | "
            f"Evidence consistency: "
            f"{row['evidence_consistency']:.4f}"
        )
        print(
            f"Human quality: {row['human_reply_quality']} | "
            f"Human grounded: {row['human_grounded']} | "
            f"Human handling: {row['human_handling_decision']}"
        )
        print(
            f"Customer: {row['customer_message']}"
        )
        print("-" * 80)

    print()
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()