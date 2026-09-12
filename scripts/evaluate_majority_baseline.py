import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEAK_LABEL_FILE = PROJECT_ROOT / "results" / "tables" / "tesco_weak_labels_a1_6.csv"
GOLD_FILE = PROJECT_ROOT / "results" / "tables" / "tesco_golden_evaluation_set_172.csv"
OUTPUT_JSON = PROJECT_ROOT / "results" / "tables" / "majority_baseline_metrics.json"
OUTPUT_TXT = PROJECT_ROOT / "results" / "tables" / "majority_baseline_report.txt"

TARGET_INTENTS = {
    "delivery_order",
    "pricing_payment",
    "store_staff_service",
    "product_availability",
    "product_quality",
    "product_safety_sensitive",
    "product_information_policy",
    "feedback_suggestion",
}


def main() -> None:
    weak = pd.read_csv(WEAK_LABEL_FILE)
    gold = pd.read_csv(GOLD_FILE)

    required_weak = {"weak_intent", "weak_label_status"}
    missing_weak = required_weak - set(weak.columns)
    if missing_weak:
        raise ValueError(
            f"Missing A1.6 columns: {sorted(missing_weak)}"
        )

    if "human_intent" not in gold.columns:
        raise ValueError(
            "Golden set must contain 'human_intent'."
        )

    weak_target = weak[
        (weak["weak_label_status"] == "accepted")
        & weak["weak_intent"].isin(TARGET_INTENTS)
    ].copy()

    gold_target = gold[
        gold["human_intent"].isin(TARGET_INTENTS)
    ].copy()

    if weak_target.empty:
        raise ValueError(
            "No accepted A1.6 target-intent rows found."
        )

    majority_intent = (
        weak_target["weak_intent"]
        .value_counts()
        .sort_values(ascending=False)
        .index[0]
    )

    y_true = gold_target["human_intent"].astype(str)
    y_pred = pd.Series(
        [majority_intent] * len(gold_target),
        index=gold_target.index,
    )

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=sorted(TARGET_INTENTS),
        average="macro",
        zero_division=0,
    )

    development_counts = (
        weak_target["weak_intent"]
        .value_counts()
        .to_dict()
    )

    metrics = {
        "baseline": "majority_class",
        "majority_intent": majority_intent,
        "accepted_development_rows": int(len(weak_target)),
        "development_label_counts": development_counts,
        "gold_target_rows": int(len(gold_target)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }

    OUTPUT_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "SUPPORTIQ — MAJORITY-CLASS BASELINE",
            "=" * 70,
            f"Majority development intent: {majority_intent}",
            f"Accepted development rows: {len(weak_target)}",
            f"Target-intent gold rows: {len(gold_target)}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro-F1: {macro_f1:.4f}",
            "",
            "Accepted development label distribution:",
            *[
                f"{label}: {count}"
                for label, count in sorted(
                    development_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        ]
    )

    OUTPUT_TXT.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)
    print()
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
