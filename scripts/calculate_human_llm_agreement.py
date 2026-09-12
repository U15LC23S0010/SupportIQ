from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32_scored.csv"
)

LLM_PATH = Path(
    "results/tables/supportiq_llm_judgments_32.csv"
)

JSON_OUTPUT = Path(
    "results/tables/supportiq_human_llm_agreement.json"
)

TEXT_OUTPUT = Path(
    "results/tables/supportiq_human_llm_agreement.txt"
)


REQUIRED_HUMAN_COLUMNS = {
    "review_id",
    "human_reply_quality",
    "human_grounded",
    "human_handling_decision",
}


REQUIRED_LLM_COLUMNS = {
    "review_id",
    "llm_reply_quality",
    "llm_grounded",
    "llm_handling_decision",
}


def exact_agreement(human, llm):
    return float((human == llm).mean())


def main():
    if not HUMAN_PATH.exists():
        raise FileNotFoundError(
            f"Human review file not found: {HUMAN_PATH}"
        )

    if not LLM_PATH.exists():
        raise FileNotFoundError(
            f"LLM judgment file not found: {LLM_PATH}"
        )

    human = pd.read_csv(HUMAN_PATH)
    llm = pd.read_csv(LLM_PATH)

    missing_human = REQUIRED_HUMAN_COLUMNS - set(human.columns)
    missing_llm = REQUIRED_LLM_COLUMNS - set(llm.columns)

    if missing_human:
        raise ValueError(
            f"Missing human columns: {sorted(missing_human)}"
        )

    if missing_llm:
        raise ValueError(
            f"Missing LLM columns: {sorted(missing_llm)}"
        )

    if len(human) != 32:
        raise ValueError(
            f"Expected 32 human cases, found {len(human)}"
        )

    if len(llm) != 32:
        raise ValueError(
            f"Expected 32 LLM cases, found {len(llm)}"
        )

    human = human[
        [
            "review_id",
            "human_reply_quality",
            "human_grounded",
            "human_handling_decision",
        ]
    ].copy()

    llm = llm[
        [
            "review_id",
            "llm_reply_quality",
            "llm_grounded",
            "llm_handling_decision",
        ]
    ].copy()

    merged = human.merge(
        llm,
        on="review_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 32:
        raise ValueError(
            f"Expected 32 matched review IDs, found {len(merged)}"
        )

    quality_human = merged["human_reply_quality"].astype(int)
    quality_llm = merged["llm_reply_quality"].astype(int)

    grounded_human = merged["human_grounded"].astype(int)
    grounded_llm = merged["llm_grounded"].astype(int)

    handling_human = merged["human_handling_decision"].astype(str)
    handling_llm = merged["llm_handling_decision"].astype(str)

    results = {
        "cases": 32,
        "reply_quality": {
            "human_distribution": {
                str(k): int(v)
                for k, v in quality_human.value_counts().sort_index().items()
            },
            "llm_distribution": {
                str(k): int(v)
                for k, v in quality_llm.value_counts().sort_index().items()
            },
            "exact_agreement": exact_agreement(
                quality_human,
                quality_llm,
            ),
            "cohen_kappa": float(
                cohen_kappa_score(
                    quality_human,
                    quality_llm,
                    weights="quadratic",
                )
            ),
        },
        "grounded": {
            "human_distribution": {
                str(k): int(v)
                for k, v in grounded_human.value_counts().sort_index().items()
            },
            "llm_distribution": {
                str(k): int(v)
                for k, v in grounded_llm.value_counts().sort_index().items()
            },
            "exact_agreement": exact_agreement(
                grounded_human,
                grounded_llm,
            ),
            "cohen_kappa": float(
                cohen_kappa_score(
                    grounded_human,
                    grounded_llm,
                    weights="quadratic",
                )
            ),
        },
        "handling": {
            "human_distribution": {
                str(k): int(v)
                for k, v in handling_human.value_counts().items()
            },
            "llm_distribution": {
                str(k): int(v)
                for k, v in handling_llm.value_counts().items()
            },
            "exact_agreement": exact_agreement(
                handling_human,
                handling_llm,
            ),
            "cohen_kappa": float(
                cohen_kappa_score(
                    handling_human,
                    handling_llm,
                )
            ),
        },
    }

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    lines = [
        "SUPPORTIQ — HUMAN VS LLM JUDGE AGREEMENT",
        "=" * 60,
        "",
        f"Cases: {results['cases']}",
        "",
        "REPLY QUALITY",
        "-" * 60,
        f"Exact agreement: "
        f"{results['reply_quality']['exact_agreement']:.4f}",
        f"Quadratic weighted Cohen's kappa: "
        f"{results['reply_quality']['cohen_kappa']:.4f}",
        "",
        "GROUNDED",
        "-" * 60,
        f"Exact agreement: "
        f"{results['grounded']['exact_agreement']:.4f}",
        f"Quadratic weighted Cohen's kappa: "
        f"{results['grounded']['cohen_kappa']:.4f}",
        "",
        "HANDLING DECISION",
        "-" * 60,
        f"Exact agreement: "
        f"{results['handling']['exact_agreement']:.4f}",
        f"Cohen's kappa: "
        f"{results['handling']['cohen_kappa']:.4f}",
        "",
        "NOTE",
        "-" * 60,
        "Agreement statistics are descriptive because the review set",
        "contains only 32 cases.",
    ]

    with TEXT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print()
    print(f"Saved JSON: {JSON_OUTPUT.resolve()}")
    print(f"Saved report: {TEXT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()