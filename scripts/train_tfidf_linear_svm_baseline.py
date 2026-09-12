from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


WEAK_PATH = Path("results/tables/tesco_weak_labels_a1_6.csv")
GOLD_PATH = Path("results/tables/tesco_golden_evaluation_set_172.csv")

MODEL_DIR = Path("results/models")
TABLE_DIR = Path("results/tables")

MODEL_PATH = MODEL_DIR / "intent_baseline_tfidf_linearsvc.joblib"
VECTORIZER_PATH = MODEL_DIR / "intent_baseline_tfidf_linearsvc_vectorizer.joblib"
METRICS_PATH = TABLE_DIR / "intent_baseline_tfidf_linearsvc_metrics.json"
REPORT_PATH = TABLE_DIR / "intent_baseline_tfidf_linearsvc_report.txt"
GOLD_PRED_PATH = TABLE_DIR / "intent_baseline_tfidf_linearsvc_gold_predictions.csv"

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
    weak = pd.read_csv(WEAK_PATH)

    weak = weak[
        weak["weak_intent"].isin(TARGET_INTENTS)
    ].copy()

    weak = weak.dropna(
        subset=["customer_message", "conversation_id"]
    )

    print("=" * 80)
    print("SUPPORTIQ — TF-IDF + LINEAR SVM SECOND BASELINE")
    print("=" * 80)

    print(f"Weak-label training rows: {len(weak):,}")

    X = weak["customer_message"].fillna("").astype(str)
    y = weak["weak_intent"].astype(str)
    groups = weak["conversation_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print()
    print(f"Training rows:          {len(X_train):,}")
    print(f"Validation rows:        {len(X_val):,}")
    print(
        f"Training conversations: {groups.iloc[train_idx].nunique():,}"
    )
    print(
        f"Validation conversations: "
        f"{groups.iloc[val_idx].nunique():,}"
    )

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
        ]
    )

    print()
    print("Fitting TF-IDF features...")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print(
        f"Training feature matrix: "
        f"{X_train_vec.shape[0]:,} x {X_train_vec.shape[1]:,}"
    )

    print(
        f"Validation feature matrix: "
        f"{X_val_vec.shape[0]:,} x {X_val_vec.shape[1]:,}"
    )

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )

    print()
    print("Training Linear SVM...")

    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_report = classification_report(
        y_val,
        val_pred,
        labels=TARGET_INTENTS,
        zero_division=0,
    )

    print()
    print("=" * 80)
    print("DEVELOPMENT VALIDATION")
    print("=" * 80)

    print(f"Accuracy:  {val_accuracy:.4f}")
    print(f"Macro-F1:  {val_macro_f1:.4f}")

    print()
    print(val_report)

    print()
    print("Evaluating frozen human gold set...")

    gold = pd.read_csv(GOLD_PATH)

    gold_target = gold[
        gold["human_intent"].isin(TARGET_INTENTS)
    ].copy()

    gold_target = gold_target.dropna(
        subset=["evaluation_incoming_message"]
    )

    X_gold = (
        gold_target["evaluation_incoming_message"]
        .fillna("")
        .astype(str)
    )

    y_gold = gold_target["human_intent"].astype(str)

    X_gold_vec = vectorizer.transform(X_gold)

    gold_pred = model.predict(X_gold_vec)

    gold_accuracy = accuracy_score(
        y_gold,
        gold_pred,
    )

    gold_macro_f1 = f1_score(
        y_gold,
        gold_pred,
        average="macro",
        zero_division=0,
    )

    gold_report = classification_report(
        y_gold,
        gold_pred,
        labels=TARGET_INTENTS,
        zero_division=0,
    )

    print()
    print("=" * 80)
    print("FROZEN 172-EXAMPLE GOLD EVALUATION")
    print("=" * 80)

    print(
        f"Gold target-intent examples: "
        f"{len(gold_target)}"
    )

    print(f"Accuracy:  {gold_accuracy:.4f}")
    print(f"Macro-F1:  {gold_macro_f1:.4f}")

    print()
    print(gold_report)

    gold_predictions = gold_target[
        [
            "conversation_id",
            "evaluation_incoming_tweet_id",
            "evaluation_incoming_message",
            "human_intent",
        ]
    ].copy()

    gold_predictions["predicted_intent"] = gold_pred

    gold_predictions["correct"] = (
        gold_predictions["human_intent"]
        == gold_predictions["predicted_intent"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    gold_predictions.to_csv(
        GOLD_PRED_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics = {
        "baseline": "tfidf_linear_svc",
        "features": {
            "word": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.98,
                "sublinear_tf": True,
                "max_features": 80000,
            },
            "char": {
                "ngram_range": [3, 5],
                "min_df": 2,
                "sublinear_tf": True,
                "max_features": 80000,
            },
        },
        "classifier": {
            "type": "LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "development": {
            "rows": len(weak),
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "accuracy": val_accuracy,
            "macro_f1": val_macro_f1,
        },
        "gold": {
            "rows_total": len(gold),
            "target_rows": len(gold_target),
            "accuracy": gold_accuracy,
            "macro_f1": gold_macro_f1,
        },
        "target_intents": TARGET_INTENTS,
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = (
        "SUPPORTIQ — TF-IDF + LINEAR SVM SECOND BASELINE\n"
        + "=" * 80
        + "\n\n"
        + "DEVELOPMENT VALIDATION\n"
        + f"Accuracy: {val_accuracy:.4f}\n"
        + f"Macro-F1: {val_macro_f1:.4f}\n\n"
        + val_report
        + "\n\n"
        + "FROZEN GOLD EVALUATION\n"
        + f"Target-intent examples: {len(gold_target)}\n"
        + f"Accuracy: {gold_accuracy:.4f}\n"
        + f"Macro-F1: {gold_macro_f1:.4f}\n\n"
        + gold_report
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ARTIFACTS SAVED")
    print("=" * 80)

    print(MODEL_DIR.resolve())
    print(METRICS_PATH.resolve())
    print(REPORT_PATH.resolve())
    print(GOLD_PRED_PATH.resolve())


if __name__ == "__main__":
    main()