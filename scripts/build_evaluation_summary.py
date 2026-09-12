from pathlib import Path
import json

import pandas as pd


GOLD_PATH = Path(
    "results/tables/tesco_golden_evaluation_set_172.csv"
)

AGENT_METRICS_PATH = Path(
    "results/tables/supportiq_agent_evaluation_metrics.json"
)

HUMAN_REVIEW_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32_scored.csv"
)

OUTPUT_JSON = Path(
    "results/tables/supportiq_evaluation_summary.json"
)

OUTPUT_TXT = Path(
    "results/tables/supportiq_evaluation_summary.txt"
)


def main():
    gold = pd.read_csv(GOLD_PATH)
    human = pd.read_csv(HUMAN_REVIEW_PATH)

    with AGENT_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        agent_metrics = json.load(f)

    human["intent_correct"] = (
        human["predicted_intent"]
        == human["human_intent"]
    )

    human["intent_confidence"] = pd.to_numeric(
        human["intent_confidence"],
        errors="coerce",
    )

    human["high_confidence"] = (
        human["intent_confidence"] >= 0.70
    )

    human["human_reply_quality"] = pd.to_numeric(
        human["human_reply_quality"],
        errors="coerce",
    )

    human["human_grounded"] = pd.to_numeric(
        human["human_grounded"],
        errors="coerce",
    )

    high_confidence_cases = int(
        human["high_confidence"].sum()
    )

    high_confidence_wrong = int(
        (
            human["high_confidence"]
            & (~human["intent_correct"])
        ).sum()
    )

    summary = {
        "gold_set": {
            "total_cases": int(len(gold)),
            "target_intent_cases": int(
                (
                    ~gold["human_intent"].isin(
                        [
                            "non_support_social",
                            "other_unclear",
                        ]
                    )
                ).sum()
            ),
        },
        "agent_gold_evaluation": agent_metrics,
        "human_review": {
            "cases": int(len(human)),
            "intent_accuracy": float(
                human["intent_correct"].mean()
            ),
            "mean_reply_quality": float(
                human["human_reply_quality"].mean()
            ),
            "mean_grounded": float(
                human["human_grounded"].mean()
            ),
            "human_auto_handle": int(
                (
                    human["human_handling_decision"]
                    == "AUTO_HANDLE"
                ).sum()
            ),
            "human_escalate": int(
                (
                    human["human_handling_decision"]
                    == "ESCALATE"
                ).sum()
            ),
            "high_confidence_cases": high_confidence_cases,
            "high_confidence_wrong": high_confidence_wrong,
            "high_confidence_wrong_rate": (
                high_confidence_wrong
                / high_confidence_cases
                if high_confidence_cases
                else None
            ),
        },
        "llm_judge": {
            "executed": True,
            "model": "gemini-3.5-flash-lite",
            "cases": 32,
            "reply_quality_exact_agreement": 0.1250,
            "reply_quality_quadratic_kappa": 0.0608,
            "grounded_exact_agreement": 0.46875,
            "grounded_quadratic_kappa": -0.1069,
            "handling_exact_agreement": 0.65625,
            "handling_kappa": 0.1373,
        },
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    human_summary = summary["human_review"]

    lines = [
        "SUPPORTIQ — EVALUATION SUMMARY",
        "=" * 70,
        "",
        (
            f"Frozen gold set: "
            f"{summary['gold_set']['total_cases']}"
        ),
        (
            "Target-intent gold cases: "
            f"{summary['gold_set']['target_intent_cases']}"
        ),
        "",
        "AGENT GOLD EVALUATION",
        "-" * 70,
        (
            "Intent accuracy: "
            f"{agent_metrics.get('intent_accuracy')}"
        ),
        (
            "Intent macro-F1: "
            f"{agent_metrics.get('intent_macro_f1')}"
        ),
        (
            "AUTO_HANDLE: "
            f"{agent_metrics.get('auto_handle_count')}"
        ),
        (
            "ESCALATE: "
            f"{agent_metrics.get('escalate_count')}"
        ),
        "",
        "HUMAN REPLY REVIEW",
        "-" * 70,
        (
            f"Cases: "
            f"{human_summary['cases']}"
        ),
        (
            "Intent accuracy on review sample: "
            f"{human_summary['intent_accuracy']:.4f}"
        ),
        (
            "Mean reply quality / 2: "
            f"{human_summary['mean_reply_quality']:.4f}"
        ),
        (
            "Mean groundedness / 2: "
            f"{human_summary['mean_grounded']:.4f}"
        ),
        (
            "Human AUTO_HANDLE: "
            f"{human_summary['human_auto_handle']}"
        ),
        (
            "Human ESCALATE: "
            f"{human_summary['human_escalate']}"
        ),
        "",
        "CONFIDENCE WARNING",
        "-" * 70,
        (
            "High-confidence cases (>= 0.70): "
            f"{human_summary['high_confidence_cases']}"
        ),
        (
            "High-confidence wrong: "
            f"{human_summary['high_confidence_wrong']}"
        ),
        (
            "High-confidence error rate: "
            f"{human_summary['high_confidence_wrong_rate']:.4f}"
        ),
        "",
        "LLM JUDGE",
        "-" * 70,
        "Executed: True",
        "Model: gemini-3.5-flash-lite",
        "Cases: 32",
        "Reply quality exact agreement: 0.1250",
        "Reply quality quadratic kappa: 0.0608",
        "Grounded exact agreement: 0.4688",
        "Grounded quadratic kappa: -0.1069",
        "Handling exact agreement: 0.6562",
        "Handling kappa: 0.1373",
        (
            "Agreement is descriptive because the review set "
            "contains 32 cases."
        ),
        (
            "LLM judge is treated as a secondary evaluator, "
            "not ground truth."
        ),
    ]

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )

    print(
        "\n".join(lines)
    )
    print()
    print(
        f"Saved JSON: "
        f"{OUTPUT_JSON.resolve()}"
    )
    print(
        f"Saved text: "
        f"{OUTPUT_TXT.resolve()}"
    )


if __name__ == "__main__":
    main()