from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32_scored.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_human_review_failure_analysis.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    df["human_reply_quality"] = df["human_reply_quality"].astype(int)
    df["human_grounded"] = df["human_grounded"].astype(int)

    df["intent_correct"] = (
        df["predicted_intent"] == df["human_intent"]
    )

    df["needs_failure_review"] = (
        (~df["intent_correct"])
        | (df["human_reply_quality"] <= 1)
        | (df["human_grounded"] <= 1)
    )

    def classify_failure(row):
        if (
            not row["intent_correct"]
            and row["human_grounded"] == 2
            and row["human_reply_quality"] <= 1
        ):
            return "wrong_intent_good_evidence_poor_reply"

        if (
            row["intent_correct"]
            and row["human_grounded"] == 2
            and row["human_reply_quality"] <= 1
        ):
            return "correct_intent_good_evidence_poor_reply"

        if (
            not row["intent_correct"]
            and row["human_grounded"] <= 1
        ):
            return "wrong_intent_weak_evidence"

        if row["human_reply_quality"] <= 1:
            return "reply_quality_issue"

        return "other"

    df["failure_type"] = df.apply(
        classify_failure,
        axis=1,
    )

    selected = df[
        [
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
    ].copy()

    selected = selected[
        df["needs_failure_review"]
    ]

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("SUPPORTIQ — HUMAN REVIEW FAILURE ANALYSIS")
    print("=" * 80)

    print(f"Total reviewed: {len(df)}")
    print(
        f"Cases selected for failure analysis: "
        f"{len(selected)}"
    )

    print("\nFailure type distribution:")
    print(
        selected["failure_type"]
        .value_counts()
        .to_string()
    )

    print("\nSelected review IDs:")
    print(
        selected["review_id"]
        .tolist()
    )

    print("\nSaved:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()