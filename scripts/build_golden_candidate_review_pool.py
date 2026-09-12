from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_candidate_pool_audit.csv"
)

DUPLICATE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_incoming_duplicate_audit.csv"
)

NEAR_DUPLICATE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_near_duplicate_audit.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "tesco_golden_candidate_review_pool_360.csv"
)

RANDOM_SEED = 42

CANDIDATES_PER_INTENT = 40
UNCERTAIN_RESERVE = 40
TOTAL_REVIEW_POOL = (
    8 * CANDIDATES_PER_INTENT
    + UNCERTAIN_RESERVE
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


def load_candidates() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidate pool:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "conversation_id",
        "conversation_length",
        "customer_message_count",
        "support_message_count",
        "first_customer_message",
        "normalized_incoming_message",
        "candidate_intent",
        "candidate_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def load_duplicate_ids() -> set[int]:
    """
    Return conversation IDs that are exact duplicate incoming-message
    representatives beyond the first occurrence.

    We do NOT delete all duplicate conversations from the dataset.
    We only prevent duplicate incoming messages from being repeatedly
    selected into the candidate review pool.
    """
    if not DUPLICATE_PATH.exists():
        return set()

    duplicates = pd.read_csv(DUPLICATE_PATH)

    if duplicates.empty:
        return set()

    result: set[int] = set()

    for ids in duplicates.get(
        "example_conversation_ids",
        pd.Series(dtype=object),
    ):
        if pd.isna(ids):
            continue

        text = str(ids)

        # The audit CSV stores Python-list-like strings.
        text = (
            text.replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )

        if not text:
            continue

        parts = text.split(",")

        parsed_ids = []

        for part in parts:
            try:
                parsed_ids.append(int(part))
            except ValueError:
                continue

        # Keep first representative, flag all others.
        result.update(parsed_ids[1:])

    return result


def load_near_duplicate_ids() -> set[int]:
    """
    Return conversation IDs participating in near-duplicate pairs.

    These are ONLY flagged. They are not automatically excluded.
    """
    if not NEAR_DUPLICATE_PATH.exists():
        return set()

    near_duplicates = pd.read_csv(NEAR_DUPLICATE_PATH)

    if near_duplicates.empty:
        return set()

    ids: set[int] = set()

    for column in [
        "conversation_id_1",
        "conversation_id_2",
    ]:
        if column not in near_duplicates.columns:
            continue

        values = pd.to_numeric(
            near_duplicates[column],
            errors="coerce",
        ).dropna()

        ids.update(
            values.astype(int).tolist()
        )

    return ids


def add_length_strata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["incoming_word_count"] = (
        result["first_customer_message"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    def conversation_group(value: int) -> str:
        if value == 2:
            return "2 tweets"
        if value == 3:
            return "3 tweets"
        if value <= 5:
            return "4-5 tweets"
        if value <= 10:
            return "6-10 tweets"
        return "11+ tweets"

    def incoming_group(value: int) -> str:
        if value <= 3:
            return "0-3 words"
        if value <= 7:
            return "4-7 words"
        if value <= 15:
            return "8-15 words"
        if value <= 30:
            return "16-30 words"
        return "31+ words"

    result["conversation_length_group"] = (
        result["conversation_length"]
        .astype(int)
        .map(conversation_group)
    )

    result["incoming_length_group"] = (
        result["incoming_word_count"]
        .map(incoming_group)
    )

    return result


def mark_duplicate_risks(
    df: pd.DataFrame,
    exact_duplicate_ids: set[int],
    near_duplicate_ids: set[int],
) -> pd.DataFrame:
    result = df.copy()

    result["exact_duplicate_risk"] = (
        result["conversation_id"]
        .astype(int)
        .isin(exact_duplicate_ids)
    )

    result["near_duplicate_risk"] = (
        result["conversation_id"]
        .astype(int)
        .isin(near_duplicate_ids)
    )

    return result


def stratified_sample_group(
    group: pd.DataFrame,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Select examples while trying to preserve variation across:

    - conversation length
    - incoming message length

    This is still a candidate-review sample, not the golden set.
    """
    if len(group) <= n:
        return group.copy()

    rng = np.random.default_rng(random_state)

    group = group.copy()

    group["sampling_stratum"] = (
        group["conversation_length_group"].astype(str)
        + " | "
        + group["incoming_length_group"].astype(str)
    )

    strata = list(
        group["sampling_stratum"]
        .value_counts()
        .index
    )

    selected_parts = []

    # Initial proportional allocation.
    proportions = (
        group["sampling_stratum"]
        .value_counts(normalize=True)
    )

    allocations = {
        stratum: max(
            1,
            int(
                round(
                    proportions[stratum]
                    * n
                )
            ),
        )
        for stratum in strata
    }

    # Fix total allocation.
    while sum(allocations.values()) > n:
        removable = [
            key
            for key, value in allocations.items()
            if value > 1
        ]

        if not removable:
            break

        key = max(
            removable,
            key=lambda item: allocations[item],
        )

        allocations[key] -= 1

    while sum(allocations.values()) < n:
        key = max(
            strata,
            key=lambda item: (
                group[
                    group["sampling_stratum"]
                    == item
                ].shape[0]
                - allocations.get(item, 0)
            ),
        )

        allocations[key] += 1

    for index, stratum in enumerate(strata):
        stratum_df = group[
            group["sampling_stratum"]
            == stratum
        ]

        sample_n = min(
            allocations[stratum],
            len(stratum_df),
        )

        if sample_n > 0:
            selected_parts.append(
                stratum_df.sample(
                    n=sample_n,
                    random_state=(
                        random_state
                        + index
                    ),
                )
            )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(selected) > n:
        selected = selected.sample(
            n=n,
            random_state=random_state,
        )

    if len(selected) < n:
        remaining = group[
            ~group["conversation_id"].isin(
                selected["conversation_id"]
            )
        ]

        need = n - len(selected)

        if len(remaining) >= need:
            extra = remaining.sample(
                n=need,
                random_state=random_state,
            )

            selected = pd.concat(
                [selected, extra],
                ignore_index=True,
            )

    return selected


def select_target_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    selected_parts = []

    for index, intent in enumerate(TARGET_INTENTS):
        pool = df[
            df["candidate_intent"]
            == intent
        ].copy()

        # Prefer candidates without exact duplicate risk.
        clean = pool[
            ~pool["exact_duplicate_risk"]
        ].copy()

        if len(clean) >= CANDIDATES_PER_INTENT:
            pool = clean

        if len(pool) < CANDIDATES_PER_INTENT:
            raise RuntimeError(
                f"Not enough candidates for "
                f"{intent}: "
                f"{len(pool)} available, "
                f"{CANDIDATES_PER_INTENT} required."
            )

        sampled = stratified_sample_group(
            pool,
            n=CANDIDATES_PER_INTENT,
            random_state=(
                RANDOM_SEED + index
            ),
        )

        sampled = sampled.copy()
        sampled["review_pool_reason"] = (
            "target_intent_candidate"
        )

        selected_parts.append(sampled)

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


def select_uncertain_candidates(
    df: pd.DataFrame,
    already_selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select an uncertainty reserve.

    These examples are intentionally not assigned a presumed target
    label. They allow human annotation to catch cases where the
    heuristic screen missed an intent.
    """
    selected_ids = set(
        already_selected[
            "conversation_id"
        ].astype(int)
    )

    pool = df[
        ~df["conversation_id"]
        .astype(int)
        .isin(selected_ids)
    ].copy()

    pool = pool[
        pool["candidate_intent"].isna()
    ].copy()

    # Do not deliberately populate the uncertainty reserve with exact
    # duplicate incoming messages.
    pool = pool[
        ~pool["exact_duplicate_risk"]
    ].copy()

    if len(pool) < UNCERTAIN_RESERVE:
        raise RuntimeError(
            "Not enough uncertain candidates "
            f"for reserve: {len(pool)} available, "
            f"{UNCERTAIN_RESERVE} required."
        )

    sampled = stratified_sample_group(
        pool,
        n=UNCERTAIN_RESERVE,
        random_state=RANDOM_SEED + 100,
    )

    sampled = sampled.copy()

    sampled["review_pool_reason"] = (
        "uncertain_candidate"
    )

    return sampled


def finalize_review_pool(
    target_candidates: pd.DataFrame,
    uncertain_candidates: pd.DataFrame,
) -> pd.DataFrame:
    pool = pd.concat(
        [
            target_candidates,
            uncertain_candidates,
        ],
        ignore_index=True,
    )

    # Exactly one row per conversation.
    pool = pool.drop_duplicates(
        subset=["conversation_id"],
        keep="first",
    )

    # Stable review order.
    pool = pool.sort_values(
        [
            "review_pool_reason",
            "candidate_intent",
            "conversation_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    # Add blank human-review fields.
    pool["human_intent"] = ""
    pool["annotator_confidence"] = ""
    pool["annotation_notes"] = ""
    pool["is_multi_intent"] = ""
    pool["secondary_issue_present"] = ""
    pool["annotation_status"] = "not_annotated"

    return pool


def print_summary(pool: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("SUPPORTIQ — GOLDEN CANDIDATE REVIEW POOL")
    print("=" * 90)

    print(
        f"\nReview pool size: {len(pool):,}"
    )

    print("\nReason:")
    print(
        pool[
            "review_pool_reason"
        ]
        .value_counts()
        .to_string()
    )

    print("\nHeuristic screening signal:")
    print(
        pool[
            "candidate_intent"
        ]
        .fillna("uncertain")
        .value_counts()
        .to_string()
    )

    print("\nConversation-length distribution:")
    print(
        pool[
            "conversation_length_group"
        ]
        .value_counts()
        .reindex(
            [
                "2 tweets",
                "3 tweets",
                "4-5 tweets",
                "6-10 tweets",
                "11+ tweets",
            ]
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nIncoming-message length distribution:")
    print(
        pool[
            "incoming_length_group"
        ]
        .value_counts()
        .reindex(
            [
                "0-3 words",
                "4-7 words",
                "8-15 words",
                "16-30 words",
                "31+ words",
            ]
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print(
        "\nExact duplicate-risk candidates: "
        f"{pool['exact_duplicate_risk'].sum():,}"
    )

    print(
        "Near-duplicate-risk candidates: "
        f"{pool['near_duplicate_risk'].sum():,}"
    )

    print("\nIMPORTANT:")
    print(
        "This is a candidate review pool only."
    )
    print(
        "Heuristic candidate_intent is not a human label."
    )
    print(
        "human_intent fields are intentionally blank."
    )
    print(
        "No data/golden files were created."
    )
    print(
        "No model was trained."
    )


def main() -> None:
    print(
        f"Reading candidate pool:\n{INPUT_PATH}"
    )

    candidates = load_candidates()

    candidates = add_length_strata(
        candidates
    )

    exact_duplicate_ids = (
        load_duplicate_ids()
    )

    near_duplicate_ids = (
        load_near_duplicate_ids()
    )

    candidates = mark_duplicate_risks(
        candidates,
        exact_duplicate_ids,
        near_duplicate_ids,
    )

    target_candidates = select_target_candidates(
        candidates
    )

    uncertain_candidates = (
        select_uncertain_candidates(
            candidates,
            target_candidates,
        )
    )

    review_pool = finalize_review_pool(
        target_candidates,
        uncertain_candidates,
    )

    if len(review_pool) != TOTAL_REVIEW_POOL:
        raise RuntimeError(
            f"Expected {TOTAL_REVIEW_POOL} candidates, "
            f"got {len(review_pool)}."
        )

    review_pool.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        review_pool
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print(
        "\nNext stage:"
        "\n- inspect the candidate review pool"
        "\n- manually annotate using frozen Version 1.1"
        "\n- select the final 200"
        "\n- only then freeze data/golden/"
    )


if __name__ == "__main__":
    main()