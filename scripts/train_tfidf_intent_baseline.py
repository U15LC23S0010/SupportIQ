from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEAK_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_weak_labels_a1_6.csv"
)

GOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_evaluation_set_172.csv"
)

MODEL_DIR = PROJECT_ROOT / "results" / "models"
METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "intent_baseline_tfidf_logreg_metrics.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "intent_baseline_tfidf_logreg_report.txt"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "intent_baseline_tfidf_logreg_gold_predictions.csv"
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


def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"https?://\S+", " ", regex=True)
        .str.replace(r"www\.\S+", " ", regex=True)
        .str.replace(r"@\w+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_features():
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
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
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
        ]
    )


def main():
    weak = pd.read_csv(
        WEAK_PATH,
        dtype=str,
    )

    weak = weak[
        weak["weak_label_status"] == "accepted"
    ].copy()

    weak = weak[
        weak["weak_intent"].isin(TARGET_INTENTS)
    ].copy()

    weak["text_clean"] = normalize_text(
        weak["customer_message"]
    )

    print("=" * 80)
    print("SUPPORTIQ — TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 80)
    print(f"Weak-label training rows: {len(weak):,}")
    print()

    # ---------------------------------------------------------------
    # Development split by conversation.
    # A conversation must never appear in both train and validation.
    # ---------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            weak["text_clean"],
            weak["weak_intent"],
            groups=weak["conversation_id"],
        )
    )

    train = weak.iloc[train_idx].copy()
    val = weak.iloc[val_idx].copy()

    print(f"Training rows:          {len(train):,}")
    print(f"Validation rows:        {len(val):,}")
    print(
        f"Training conversations: "
        f"{train['conversation_id'].nunique():,}"
    )
    print(
        f"Validation conversations: "
        f"{val['conversation_id'].nunique():,}"
    )

    overlap = (
        set(train["conversation_id"])
        & set(val["conversation_id"])
    )

    if overlap:
        raise RuntimeError(
            f"Conversation leakage detected: {len(overlap)} groups"
        )

    # ---------------------------------------------------------------
    # Fit TF-IDF on development TRAIN ONLY.
    # ---------------------------------------------------------------
    vectorizer = build_features()

    X_train = vectorizer.fit_transform(
        train["text_clean"]
    )

    X_val = vectorizer.transform(
        val["text_clean"]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(
        X_train,
        train["weak_intent"],
    )

    val_predictions = classifier.predict(X_val)

    val_accuracy = accuracy_score(
        val["weak_intent"],
        val_predictions,
    )

    val_macro_f1 = f1_score(
        val["weak_intent"],
        val_predictions,
        average="macro",
        zero_division=0,
    )

    val_report = classification_report(
        val["weak_intent"],
        val_predictions,
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

    # ---------------------------------------------------------------
    # FINAL GOLD EVALUATION.
    # The classifier has NOT seen the 172 gold examples.
    # ---------------------------------------------------------------
    gold = pd.read_csv(
        GOLD_PATH,
        dtype=str,
    )

    gold_target = gold[
        gold["human_intent"].isin(TARGET_INTENTS)
    ].copy()

    gold_target["text_clean"] = normalize_text(
        gold_target["evaluation_incoming_message"]
    )

    X_gold = vectorizer.transform(
        gold_target["text_clean"]
    )

    gold_predictions = classifier.predict(X_gold)

    gold_accuracy = accuracy_score(
        gold_target["human_intent"],
        gold_predictions,
    )

    gold_macro_f1 = f1_score(
        gold_target["human_intent"],
        gold_predictions,
        labels=TARGET_INTENTS,
        average="macro",
        zero_division=0,
    )

    gold_report = classification_report(
        gold_target["human_intent"],
        gold_predictions,
        labels=TARGET_INTENTS,
        zero_division=0,
    )

    print()
    print("=" * 80)
    print("FROZEN 172-EXAMPLE GOLD EVALUATION")
    print("=" * 80)
    print(
        f"Gold target-intent examples: "
        f"{len(gold_target):,}"
    )
    print(f"Accuracy:  {gold_accuracy:.4f}")
    print(f"Macro-F1:  {gold_macro_f1:.4f}")
    print()
    print(gold_report)

    # ---------------------------------------------------------------
    # Save artifacts.
    # ---------------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tesco_intent_tfidf_vectorizer.joblib",
    )

    joblib.dump(
        classifier,
        MODEL_DIR / "tesco_intent_logreg.joblib",
    )

    gold_output = gold_target[
        [
            "conversation_id",
            "evaluation_incoming_tweet_id",
            "evaluation_incoming_message",
            "human_intent",
        ]
    ].copy()

    gold_output["predicted_intent"] = gold_predictions
    gold_output["correct"] = (
        gold_output["human_intent"]
        == gold_output["predicted_intent"]
    )

    gold_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics = {
        "model": "tfidf_word_char_logistic_regression",
        "training_examples": int(len(train)),
        "validation_examples": int(len(val)),
        "training_conversations": int(
            train["conversation_id"].nunique()
        ),
        "validation_conversations": int(
            val["conversation_id"].nunique()
        ),
        "gold_target_intent_examples": int(
            len(gold_target)
        ),
        "validation_accuracy": float(
            val_accuracy
        ),
        "validation_macro_f1": float(
            val_macro_f1
        ),
        "gold_accuracy": float(
            gold_accuracy
        ),
        "gold_macro_f1": float(
            gold_macro_f1
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        "DEVELOPMENT VALIDATION\n"
        "======================\n"
        f"Accuracy: {val_accuracy:.4f}\n"
        f"Macro-F1: {val_macro_f1:.4f}\n\n"
        f"{val_report}\n"
        "\n"
        "FROZEN GOLD EVALUATION\n"
        "======================\n"
        f"Target-intent examples: {len(gold_target)}\n"
        f"Accuracy: {gold_accuracy:.4f}\n"
        f"Macro-F1: {gold_macro_f1:.4f}\n\n"
        f"{gold_report}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ARTIFACTS SAVED")
    print("=" * 80)
    print(MODEL_DIR)
    print(METRICS_PATH)
    print(REPORT_PATH)
    print(PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
