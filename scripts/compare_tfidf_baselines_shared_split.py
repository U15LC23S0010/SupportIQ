from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


WEAK_PATH = Path(
    "results/tables/tesco_weak_labels_a1_6.csv"
)

GOLD_PATH = Path(
    "results/tables/tesco_golden_evaluation_set_172.csv"
)

MODEL_DIR = Path(
    "results/models"
)

TABLE_DIR = Path(
    "results/tables"
)

SPLIT_PATH = (
    TABLE_DIR
    / "tfidf_baseline_shared_split.csv"
)

COMPARISON_PATH = (
    TABLE_DIR
    / "tfidf_baseline_comparison.json"
)

COMPARISON_REPORT_PATH = (
    TABLE_DIR
    / "tfidf_baseline_comparison.txt"
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


def build_features() -> FeatureUnion:
    return FeatureUnion(
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


def main() -> None:
    print("=" * 80)
    print("SUPPORTIQ — PAIRED TF-IDF BASELINE COMPARISON")
    print("=" * 80)

    if not WEAK_PATH.exists():
        raise FileNotFoundError(
            f"A1.6 weak-label file not found: {WEAK_PATH}"
        )

    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Golden evaluation file not found: {GOLD_PATH}"
        )

    weak = pd.read_csv(
        WEAK_PATH
    )

    required_columns = {
        "weak_intent",
        "weak_label_status",
        "customer_message",
        "conversation_id",
    }

    missing_columns = (
        required_columns
        - set(weak.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required A1.6 columns: "
            f"{sorted(missing_columns)}"
        )

    weak = weak[
        (weak["weak_label_status"] == "accepted")
        & weak["weak_intent"].isin(
            TARGET_INTENTS
        )
    ].copy()

    weak = weak.dropna(
        subset=[
            "customer_message",
            "conversation_id",
        ]
    )

    if weak.empty:
        raise ValueError(
            "No accepted A1.6 target-intent rows remain "
            "after filtering."
        )

    X = (
        weak["customer_message"]
        .fillna("")
        .astype(str)
    )

    y = (
        weak["weak_intent"]
        .astype(str)
    )

    groups = (
        weak["conversation_id"]
        .astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    train_positions = set(
        train_idx
    )

    split_df = pd.DataFrame(
        {
            "row_index": weak.index.astype(int),
            "conversation_id": groups.values,
            "split": [
                "train"
                if i in train_positions
                else "validation"
                for i in range(len(weak))
            ],
        }
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_df.to_csv(
        SPLIT_PATH,
        index=False,
        encoding="utf-8",
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(
        f"Accepted A1.6 target rows: {len(weak):,}"
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_val):,}"
    )

    print(
        "Training conversations: "
        f"{groups.iloc[train_idx].nunique():,}"
    )

    print(
        "Validation conversations: "
        f"{groups.iloc[val_idx].nunique():,}"
    )

    print()
    print("Fitting shared TF-IDF features...")

    vectorizer = build_features()

    X_train_vec = vectorizer.fit_transform(
        X_train
    )

    X_val_vec = vectorizer.transform(
        X_val
    )

    print(
        "Training feature matrix: "
        f"{X_train_vec.shape[0]:,} x "
        f"{X_train_vec.shape[1]:,}"
    )

    print(
        "Validation feature matrix: "
        f"{X_val_vec.shape[0]:,} x "
        f"{X_val_vec.shape[1]:,}"
    )

    models = {
        "tfidf_logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        ),
        "tfidf_linear_svc": (
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
            )
        ),
    }

    gold = pd.read_csv(
        GOLD_PATH
    )

    if "human_intent" not in gold.columns:
        raise ValueError(
            "Golden set must contain 'human_intent'."
        )

    if "evaluation_incoming_message" not in gold.columns:
        raise ValueError(
            "Golden set must contain "
            "'evaluation_incoming_message'."
        )

    gold_target = gold[
        gold["human_intent"].isin(
            TARGET_INTENTS
        )
    ].copy()

    gold_target = gold_target.dropna(
        subset=[
            "evaluation_incoming_message"
        ]
    )

    if len(gold_target) != 151:
        raise ValueError(
            "Expected 151 target-intent golden rows, "
            f"found {len(gold_target)}"
        )

    X_gold = (
        gold_target[
            "evaluation_incoming_message"
        ]
        .fillna("")
        .astype(str)
    )

    y_gold = (
        gold_target["human_intent"]
        .astype(str)
    )

    X_gold_vec = vectorizer.transform(
        X_gold
    )

    results = {}
    reports = {}

    for name, model in models.items():

        print()
        print("-" * 80)
        print(name)
        print("-" * 80)

        model.fit(
            X_train_vec,
            y_train,
        )

        val_pred = model.predict(
            X_val_vec
        )

        gold_pred = model.predict(
            X_gold_vec
        )

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

        val_report = classification_report(
            y_val,
            val_pred,
            labels=TARGET_INTENTS,
            zero_division=0,
        )

        gold_report = classification_report(
            y_gold,
            gold_pred,
            labels=TARGET_INTENTS,
            zero_division=0,
        )

        results[name] = {
            "development_accuracy": float(
                val_accuracy
            ),
            "development_macro_f1": float(
                val_macro_f1
            ),
            "gold_accuracy": float(
                gold_accuracy
            ),
            "gold_macro_f1": float(
                gold_macro_f1
            ),
            "gold_target_rows": int(
                len(gold_target)
            ),
        }

        reports[name] = {
            "development": val_report,
            "gold": gold_report,
        }

        model_path = (
            MODEL_DIR
            / f"{name}_shared_split.joblib"
        )

        vectorizer_path = (
            MODEL_DIR
            / f"{name}_shared_split_vectorizer.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        joblib.dump(
            vectorizer,
            vectorizer_path,
        )

        gold_predictions = gold_target[
            [
                "conversation_id",
                "evaluation_incoming_tweet_id",
                "evaluation_incoming_message",
                "human_intent",
            ]
        ].copy()

        gold_predictions[
            "predicted_intent"
        ] = gold_pred

        gold_predictions[
            "correct"
        ] = (
            gold_predictions["human_intent"]
            == gold_predictions["predicted_intent"]
        )

        prediction_path = (
            TABLE_DIR
            / f"{name}_gold_predictions_shared_split.csv"
        )

        gold_predictions.to_csv(
            prediction_path,
            index=False,
            encoding="utf-8",
        )

        print(
            "Development Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            "Development Macro-F1: "
            f"{val_macro_f1:.4f}"
        )

        print(
            "Gold Accuracy: "
            f"{gold_accuracy:.4f}"
        )

        print(
            "Gold Macro-F1: "
            f"{gold_macro_f1:.4f}"
        )

    comparison = {
        "shared_split": {
            "random_state": 42,
            "test_size": 0.20,
            "group_column": "conversation_id",
            "training_rows": int(
                len(train_idx)
            ),
            "validation_rows": int(
                len(val_idx)
            ),
            "training_conversations": int(
                groups.iloc[
                    train_idx
                ].nunique()
            ),
            "validation_conversations": int(
                groups.iloc[
                    val_idx
                ].nunique()
            ),
        },
        "gold": {
            "total_rows": int(
                len(gold)
            ),
            "target_rows": int(
                len(gold_target)
            ),
        },
        "results": results,
    }

    COMPARISON_PATH.write_text(
        json.dumps(
            comparison,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "SUPPORTIQ — PAIRED TF-IDF BASELINE COMPARISON",
        "=" * 80,
        "",
        "SHARED SPLIT",
        f"Accepted A1.6 target rows: {len(weak)}",
        f"Training rows: {len(train_idx)}",
        f"Validation rows: {len(val_idx)}",
        (
            "Training conversations: "
            f"{groups.iloc[train_idx].nunique()}"
        ),
        (
            "Validation conversations: "
            f"{groups.iloc[val_idx].nunique()}"
        ),
        "Random state: 42",
        "",
        "RESULTS",
        "-" * 80,
    ]

    for name, metrics in results.items():

        lines.extend(
            [
                name,
                (
                    "  Development Accuracy: "
                    f"{metrics['development_accuracy']:.4f}"
                ),
                (
                    "  Development Macro-F1: "
                    f"{metrics['development_macro_f1']:.4f}"
                ),
                (
                    "  Gold Accuracy: "
                    f"{metrics['gold_accuracy']:.4f}"
                ),
                (
                    "  Gold Macro-F1: "
                    f"{metrics['gold_macro_f1']:.4f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "DETAILED REPORTS",
            "=" * 80,
        ]
    )

    for name, report in reports.items():

        lines.extend(
            [
                "",
                name,
                "-" * 80,
                "Development:",
                report["development"],
                "Gold:",
                report["gold"],
            ]
        )

    COMPARISON_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("COMPARISON ARTIFACTS SAVED")
    print("=" * 80)

    print(
        SPLIT_PATH.resolve()
    )

    print(
        COMPARISON_PATH.resolve()
    )

    print(
        COMPARISON_REPORT_PATH.resolve()
    )


if __name__ == "__main__":
    main()