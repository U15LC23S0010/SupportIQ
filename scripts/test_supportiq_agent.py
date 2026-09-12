from src.agent.supportiq_agent import SupportIQAgent


def main() -> None:
    agent = SupportIQAgent()

    test_messages = [
        "I was charged twice for the same shopping.",
        "My order was missing items from the delivery.",
        "I found a foreign object in my food.",
        "I don't know what to do.",
    ]

    for message in test_messages:
        print()
        print("=" * 80)
        print("CUSTOMER:", message)
        print("=" * 80)

        result = agent.analyze(
            message,
            top_k=3,
        )

        print(
            "Intent:",
            result["intent"],
        )

        print(
            "Intent confidence:",
            f"{result['intent_confidence']:.4f}",
        )

        print(
            "Evidence consistency:",
            f"{result['evidence_consistency']:.4f}",
        )

        print(
            "Decision:",
            result["decision"],
        )

        print(
            "Reason:",
            result["decision_reason"],
        )

        print(
            "Draft reply:",
            result["draft_reply"],
        )

        print()
        print("EVIDENCE")

        for index, evidence in enumerate(
            result["evidence"],
            start=1,
        ):
            print()
            print(f"Evidence {index}")

            print(
                "Similarity:",
                f"{evidence['similarity']:.4f}",
            )

            print(
                "Historical customer:",
                evidence["customer_message"],
            )

            print(
                "Historical Tesco response:",
                evidence["tesco_response"],
            )


if __name__ == "__main__":
    main()