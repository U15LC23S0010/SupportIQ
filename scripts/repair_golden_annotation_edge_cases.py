from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "twcs"
    / "twcs.csv"
)

POOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_candidate_review_pool_360.csv"
)

CONVERSATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_conversations_360.csv"
)

MESSAGE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_messages_360.csv"
)

OUTPUT_CONVERSATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_conversations_360_repaired.csv"
)

OUTPUT_MESSAGE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_messages_360_repaired.csv"
)

TESCO_ACCOUNT = "Tesco"

CHUNK_SIZE = 250_000


def load_candidate_ids() -> list[int]:
    pool = pd.read_csv(POOL_PATH)

    return (
        pool["conversation_id"]
        .dropna()
        .astype(int)
        .tolist()
    )


def load_existing_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = pd.read_csv(
        CONVERSATION_PATH
    )

    messages = pd.read_csv(
        MESSAGE_PATH
    )

    return conversations, messages


def find_missing_candidate_ids(
    candidate_ids: list[int],
    conversations: pd.DataFrame,
) -> list[int]:
    existing_ids = set(
        conversations[
            "conversation_id"
        ]
        .dropna()
        .astype(int)
    )

    return sorted(
        set(candidate_ids)
        - existing_ids
    )


def parse_response_ids(value) -> list[int]:
    """
    Parse TWCS response_tweet_id values.

    TWCS can contain:
        '1213534,1213536'

    and occasionally blanks or malformed values.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    ids = []

    for part in re.split(r"\s*,\s*", text):
        if not part:
            continue

        try:
            ids.append(int(float(part)))
        except ValueError:
            continue

    return ids


def collect_missing_root_component(
    root_id: int,
) -> pd.DataFrame:
    """
    Recover a missing-root conversation component.

    We follow BOTH directions represented by TWCS:

    1. in_response_to_tweet_id
       child -> parent

    2. response_tweet_id
       parent -> child/children

    This is necessary because a missing root can still have available
    descendants whose parent relationship points to that missing root.
    """
    frontier = {int(root_id)}
    collected: dict[int, dict] = set()

    collected_rows: dict[int, dict] = {}

    print(
        f"\nRecovering missing-parent component for "
        f"conversation {root_id}"
    )

    round_number = 0

    while frontier:
        round_number += 1

        current_ids = set(frontier)
        frontier = set()

        found_this_round = 0

        print(
            f"  Round {round_number}: searching around "
            f"{len(current_ids)} tweet IDs..."
        )

        for chunk in pd.read_csv(
            DATA_PATH,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
            dtype={
                "tweet_id": "int64",
                "author_id": "string",
                "inbound": "boolean",
                "created_at": "string",
                "text": "string",
                "response_tweet_id": "string",
                "in_response_to_tweet_id": "string",
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            parent_numeric = pd.to_numeric(
                chunk["in_response_to_tweet_id"],
                errors="coerce",
            )

            parent_mask = parent_numeric.isin(
                current_ids
            )

            tweet_mask = chunk[
                "tweet_id"
            ].isin(current_ids)

            response_mask = chunk[
                "response_tweet_id"
            ].fillna("").astype(str).apply(
                lambda value: any(
                    response_id in current_ids
                    for response_id in parse_response_ids(value)
                )
            )

            mask = (
                parent_mask
                | tweet_mask
                | response_mask
            )

            matches = chunk.loc[
                mask
            ].copy()

            if matches.empty:
                continue

            for _, row in matches.iterrows():
                tweet_id = int(
                    row["tweet_id"]
                )

                if tweet_id not in collected_rows:
                    collected_rows[
                        tweet_id
                    ] = row.to_dict()

                    found_this_round += 1

                # Follow the parent relation.
                parent = pd.to_numeric(
                    row["in_response_to_tweet_id"],
                    errors="coerce",
                )

                if not pd.isna(parent):
                    parent_id = int(parent)

                    if (
                        parent_id != tweet_id
                        and parent_id not in collected_rows
                    ):
                        frontier.add(
                            parent_id
                        )

                # Follow outgoing response relationships.
                response_ids = parse_response_ids(
                    row["response_tweet_id"]
                )

                for response_id in response_ids:
                    if (
                        response_id != tweet_id
                        and response_id not in collected_rows
                    ):
                        frontier.add(
                            response_id
                        )

        print(
            f"  New tweets recovered this round: "
            f"{found_this_round}"
        )

        if found_this_round == 0:
            break

    if not collected_rows:
        return pd.DataFrame()

    recovered = pd.DataFrame(
        list(collected_rows.values())
    )

    recovered["in_response_to_tweet_id"] = pd.to_numeric(
        recovered["in_response_to_tweet_id"],
        errors="coerce",
    ).astype("Int64")

    return recovered


def build_repaired_messages(
    root_id: int,
    recovered: pd.DataFrame,
) -> pd.DataFrame:
    if recovered.empty:
        return pd.DataFrame()

    recovered = recovered.copy()

    recovered["conversation_id"] = int(
        root_id
    )

    recovered["created_at_parsed"] = pd.to_datetime(
        recovered["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )

    recovered = recovered.sort_values(
        [
            "created_at_parsed",
            "tweet_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    recovered["speaker"] = recovered.apply(
        lambda row: (
            "TESCO"
            if (
                row["inbound"] == False
                and row["author_id"] == TESCO_ACCOUNT
            )
            else (
                "CUSTOMER"
                if row["inbound"] == True
                else "OTHER"
            )
        ),
        axis=1,
    )

    recovered["message_number"] = (
        recovered.index + 1
    )

    recovered["is_customer_message"] = (
        recovered["inbound"] == True
    )

    recovered["is_tesco_support_message"] = (
        (recovered["inbound"] == False)
        & (recovered["author_id"] == TESCO_ACCOUNT)
    )

    recovered["human_evaluation_target"] = ""

    recovered["message_review_note"] = ""

    columns = [
        "conversation_id",
        "message_number",
        "tweet_id",
        "created_at",
        "speaker",
        "author_id",
        "inbound",
        "is_customer_message",
        "is_tesco_support_message",
        "in_response_to_tweet_id",
        "response_tweet_id",
        "text",
        "human_evaluation_target",
        "message_review_note",
    ]

    return recovered[
        columns
    ]


def build_repaired_conversation_row(
    root_id: int,
    recovered_messages: pd.DataFrame,
    existing_pool: pd.DataFrame,
) -> pd.DataFrame:
    if recovered_messages.empty:
        return pd.DataFrame()

    customer = recovered_messages[
        recovered_messages[
            "is_customer_message"
        ]
    ]

    support = recovered_messages[
        recovered_messages[
            "is_tesco_support_message"
        ]
    ]

    if customer.empty or support.empty:
        return pd.DataFrame()

    customer_messages = (
        customer["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    support_messages = (
        support["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    first_customer = customer.iloc[0]

    pool_row = existing_pool[
        existing_pool[
            "conversation_id"
        ].astype(int)
        == int(root_id)
    ]

    if pool_row.empty:
        candidate_intent = ""
        candidate_score = ""
        candidate_basis = ""
        exact_duplicate_risk = ""
        near_duplicate_risk = ""
        review_pool_reason = ""
    else:
        pool_row = pool_row.iloc[0]

        candidate_intent = pool_row.get(
            "candidate_intent",
            "",
        )

        candidate_score = pool_row.get(
            "candidate_score",
            "",
        )

        candidate_basis = pool_row.get(
            "candidate_basis",
            "",
        )

        exact_duplicate_risk = pool_row.get(
            "exact_duplicate_risk",
            "",
        )

        near_duplicate_risk = pool_row.get(
            "near_duplicate_risk",
            "",
        )

        review_pool_reason = pool_row.get(
            "review_pool_reason",
            "",
        )

    return pd.DataFrame(
        [
            {
                "conversation_id": int(root_id),
                "conversation_length": int(
                    len(recovered_messages)
                ),
                "customer_message_count": int(
                    len(customer)
                ),
                "tesco_support_message_count": int(
                    len(support)
                ),
                "first_customer_tweet_id": int(
                    first_customer["tweet_id"]
                ),
                "first_customer_message": str(
                    first_customer["text"]
                ),
                "all_customer_messages": " || ".join(
                    customer_messages
                ),
                "all_tesco_support_messages": " || ".join(
                    support_messages
                ),
                "candidate_review_status": (
                    "not_annotated"
                ),
                "evaluation_incoming_tweet_id": "",
                "evaluation_incoming_message": "",
                "human_intent": "",
                "annotator_confidence": "",
                "is_multi_intent": "",
                "secondary_issue_present": "",
                "annotation_notes": "",
                "annotation_status": (
                    "not_annotated"
                ),
                "parent_tweet_missing": "yes",
                "reconstruction_note": (
                    "Conversation root "
                    f"{root_id} is referenced by an "
                    "available reply but the root tweet itself "
                    "is absent from TWCS. Available descendants "
                    "were recovered using both parent and "
                    "response relationships."
                ),
                "candidate_intent_screening": (
                    candidate_intent
                ),
                "candidate_score_screening": (
                    candidate_score
                ),
                "candidate_basis_screening": (
                    candidate_basis
                ),
                "exact_duplicate_risk": (
                    exact_duplicate_risk
                ),
                "near_duplicate_risk": (
                    near_duplicate_risk
                ),
                "review_pool_reason": (
                    review_pool_reason
                ),
            }
        ]
    )


def main() -> None:
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            POOL_PATH
        )

    if not CONVERSATION_PATH.exists():
        raise FileNotFoundError(
            CONVERSATION_PATH
        )

    if not MESSAGE_PATH.exists():
        raise FileNotFoundError(
            MESSAGE_PATH
        )

    candidate_ids = load_candidate_ids()

    conversations, messages = (
        load_existing_outputs()
    )

    pool = pd.read_csv(
        POOL_PATH
    )

    missing_ids = find_missing_candidate_ids(
        candidate_ids,
        conversations,
    )

    print(
        f"Candidate IDs: {len(candidate_ids):,}"
    )

    print(
        f"Already recovered: "
        f"{len(candidate_ids) - len(missing_ids):,}"
    )

    print(
        f"Missing edge cases: "
        f"{len(missing_ids):,}"
    )

    repaired_conversation_parts = []
    repaired_message_parts = []

    for root_id in missing_ids:
        recovered = (
            collect_missing_root_component(
                root_id
            )
        )

        if recovered.empty:
            print(
                f"  ERROR: could not recover "
                f"conversation {root_id}"
            )
            continue

        repaired_messages = (
            build_repaired_messages(
                root_id,
                recovered,
            )
        )

        customer_count = int(
            repaired_messages[
                "is_customer_message"
            ].sum()
        )

        support_count = int(
            repaired_messages[
                "is_tesco_support_message"
            ].sum()
        )

        print(
            f"  Recovered message rows: "
            f"{len(repaired_messages)}"
        )

        print(
            f"  Customer messages: "
            f"{customer_count}"
        )

        print(
            f"  Tesco support messages: "
            f"{support_count}"
        )

        if customer_count == 0 or support_count == 0:
            print(
                f"  ERROR: conversation {root_id} "
                "does not contain both customer and "
                "Tesco support messages."
            )
            continue

        repaired_conversation = (
            build_repaired_conversation_row(
                root_id,
                repaired_messages,
                pool,
            )
        )

        if repaired_conversation.empty:
            print(
                f"  ERROR: could not build "
                f"conversation row for {root_id}"
            )
            continue

        repaired_message_parts.append(
            repaired_messages
        )

        repaired_conversation_parts.append(
            repaired_conversation
        )

    if repaired_conversation_parts:
        repaired_conversations = pd.concat(
            repaired_conversation_parts,
            ignore_index=True,
        )

        conversations = pd.concat(
            [
                conversations,
                repaired_conversations,
            ],
            ignore_index=True,
        )
    else:
        repaired_conversations = (
            pd.DataFrame()
        )

    if repaired_message_parts:
        repaired_messages = pd.concat(
            repaired_message_parts,
            ignore_index=True,
        )

        messages = pd.concat(
            [
                messages,
                repaired_messages,
            ],
            ignore_index=True,
        )

    conversations = (
        conversations
        .drop_duplicates(
            subset=[
                "conversation_id"
            ],
            keep="first",
        )
        .sort_values(
            "conversation_id"
        )
        .reset_index(drop=True)
    )

    messages = (
        messages
        .drop_duplicates(
            subset=[
                "conversation_id",
                "tweet_id",
            ],
            keep="first",
        )
        .sort_values(
            [
                "conversation_id",
                "created_at",
                "tweet_id",
            ]
        )
        .reset_index(drop=True)
    )

    conversations.to_csv(
        OUTPUT_CONVERSATION_PATH,
        index=False,
    )

    messages.to_csv(
        OUTPUT_MESSAGE_PATH,
        index=False,
    )

    final_ids = set(
        conversations[
            "conversation_id"
        ].astype(int)
    )

    still_missing = (
        set(candidate_ids)
        - final_ids
    )

    print("\n" + "=" * 80)
    print("RECONSTRUCTION EDGE-CASE REPAIR")
    print("=" * 80)

    original_count = len(
        pd.read_csv(
            CONVERSATION_PATH
        )
    )

    print(
        f"\nOriginal conversations: "
        f"{original_count:,}"
    )

    print(
        f"Repaired conversations added: "
        f"{len(repaired_conversations):,}"
    )

    print(
        f"Final conversation rows: "
        f"{len(conversations):,}"
    )

    print(
        f"Final message rows: "
        f"{len(messages):,}"
    )

    print(
        f"Still-missing candidate IDs: "
        f"{len(still_missing):,}"
    )

    if still_missing:
        print(
            sorted(still_missing)
        )

    print("\nOutputs:")
    print(
        OUTPUT_CONVERSATION_PATH
    )
    print(
        OUTPUT_MESSAGE_PATH
    )

    print(
        "\nNo data/golden files were created."
    )

    print(
        "No model was trained."
    )


if __name__ == "__main__":
    main()