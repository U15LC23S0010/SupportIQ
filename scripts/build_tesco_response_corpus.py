from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
GOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_evaluation_set_172.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_historical_customer_response_corpus.csv"
)

TESCO_ACCOUNT = "Tesco"
CHUNK_SIZE = 100_000


def main():
    gold = pd.read_csv(
        GOLD_PATH,
        usecols=["conversation_id"],
        dtype=str,
    )

    excluded_roots = {
        int(value)
        for value in gold["conversation_id"].dropna()
    }

    print(f"Golden conversations excluded: {len(excluded_roots):,}")
    print()

    # ---------------------------------------------------------------
    # PASS 1: Build complete tweet -> parent relationship map.
    # ---------------------------------------------------------------
    print("Pass 1: Building tweet -> parent relationships...")

    parent_map = {}
    processed = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            usecols=[
                "tweet_id",
                "in_response_to_tweet_id",
            ],
        ),
        start=1,
    ):
        valid = df[
            df["in_response_to_tweet_id"].notna()
        ]

        for row in valid.itertuples(index=False):
            parent_map[int(row.tweet_id)] = int(
                row.in_response_to_tweet_id
            )

        processed += len(df)

        if chunk_number % 5 == 0:
            print(
                f"  Processed: {processed:,} tweets",
                end="\r",
            )

    print()
    print(
        f"Tweet -> parent relationships: "
        f"{len(parent_map):,}"
    )
    print()

    # ---------------------------------------------------------------
    # PASS 2: Collect every Tesco response and its parent tweet ID.
    # ---------------------------------------------------------------
    print("Pass 2: Collecting Tesco support responses...")

    support_rows = []
    processed = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "in_response_to_tweet_id",
            ],
            dtype={
                "tweet_id": "Int64",
                "author_id": "string",
                "inbound": "boolean",
                "created_at": "string",
                "text": "string",
                "in_response_to_tweet_id": "Int64",
            },
        ),
        start=1,
    ):
        support = df[
            (df["author_id"] == TESCO_ACCOUNT)
            & (df["inbound"] == False)
            & (df["in_response_to_tweet_id"].notna())
        ].copy()

        if not support.empty:
            support_rows.append(
                support[
                    [
                        "tweet_id",
                        "created_at",
                        "text",
                        "in_response_to_tweet_id",
                    ]
                ]
            )

        processed += len(df)

        if chunk_number % 5 == 0:
            print(
                f"  Processed: {processed:,} tweets",
                end="\r",
            )

    print()

    if support_rows:
        support = pd.concat(
            support_rows,
            ignore_index=True,
        )
    else:
        support = pd.DataFrame(
            columns=[
                "tweet_id",
                "created_at",
                "text",
                "in_response_to_tweet_id",
            ]
        )

    print(
        f"Tesco responses with parent IDs: "
        f"{len(support):,}"
    )

    parent_ids = set(
        int(value)
        for value in support[
            "in_response_to_tweet_id"
        ].dropna()
    )

    print(
        f"Unique customer parent tweet IDs: "
        f"{len(parent_ids):,}"
    )
    print()

    # ---------------------------------------------------------------
    # PASS 3: Retrieve customer messages, regardless of chunk.
    # ---------------------------------------------------------------
    print("Pass 3: Retrieving customer parent messages...")

    customer_rows = []
    processed = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
            ],
            dtype={
                "tweet_id": "Int64",
                "author_id": "string",
                "inbound": "boolean",
                "created_at": "string",
                "text": "string",
            },
        ),
        start=1,
    ):
        customers = df[
            (df["tweet_id"].isin(parent_ids))
            & (df["inbound"] == True)
        ].copy()

        if not customers.empty:
            customer_rows.append(
                customers[
                    [
                        "tweet_id",
                        "author_id",
                        "created_at",
                        "text",
                    ]
                ]
            )

        processed += len(df)

        if chunk_number % 5 == 0:
            print(
                f"  Processed: {processed:,} tweets",
                end="\r",
            )

    print()

    if customer_rows:
        customers = pd.concat(
            customer_rows,
            ignore_index=True,
        )
    else:
        customers = pd.DataFrame(
            columns=[
                "tweet_id",
                "author_id",
                "created_at",
                "text",
            ]
        )

    print(
        f"Customer parent messages found: "
        f"{len(customers):,}"
    )
    print()

    # ---------------------------------------------------------------
    # PASS 4: Join response -> customer and derive roots.
    # ---------------------------------------------------------------
    print("Pass 4: Joining pairs and excluding golden conversations...")

    customers = customers.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_message",
        }
    )

    support = support.rename(
        columns={
            "tweet_id": "tesco_tweet_id",
            "created_at": "tesco_created_at",
            "text": "tesco_response",
            "in_response_to_tweet_id": "customer_tweet_id",
        }
    )

    corpus = support.merge(
        customers,
        on="customer_tweet_id",
        how="inner",
    )

    print(
        f"Customer -> Tesco pairs before exclusion: "
        f"{len(corpus):,}"
    )

    def find_root(tweet_id):
        current = int(tweet_id)
        visited = set()

        while current in parent_map:
            if current in visited:
                break

            visited.add(current)
            current = parent_map[current]

        return current

    corpus["conversation_id"] = corpus[
        "customer_tweet_id"
    ].map(find_root)

    before = len(corpus)

    corpus = corpus[
        ~corpus["conversation_id"].isin(excluded_roots)
    ].copy()

    excluded_pairs = before - len(corpus)

    print(
        f"Golden pairs excluded: {excluded_pairs:,}"
    )

    corpus = corpus[
        [
            "conversation_id",
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_message",
            "tesco_tweet_id",
            "tesco_created_at",
            "tesco_response",
        ]
    ].sort_values(
        [
            "conversation_id",
            "customer_tweet_id",
            "tesco_tweet_id",
        ],
        kind="stable",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 80)
    print("TESCO HISTORICAL CUSTOMER -> RESPONSE CORPUS")
    print("=" * 80)
    print(f"Pairs written:         {len(corpus):,}")
    print(f"Golden pairs excluded: {excluded_pairs:,}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
