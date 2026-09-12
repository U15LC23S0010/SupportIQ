from src.retrieval.tesco_retriever import (
    TescoHistoricalRetriever,
)


def main() -> None:
    retriever = TescoHistoricalRetriever()

    test_messages = [
        "I was charged twice for the same shopping.",
        "My order was missing items from the delivery.",
        "I found a foreign object in my food.",
    ]

    for message in test_messages:
        print()
        print("=" * 80)
        print("CUSTOMER:", message)
        print("=" * 80)

        results = retriever.retrieve(
            message,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(f"RESULT {rank}")
            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )
            print(
                "Historical customer:",
                result["customer_message"],
            )
            print(
                "Historical Tesco response:",
                result["tesco_response"],
            )


if __name__ == "__main__":
    main()