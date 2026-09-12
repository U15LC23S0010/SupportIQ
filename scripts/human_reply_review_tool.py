from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32_scored.csv"
)


def is_completed(row):
    quality = row.get("human_reply_quality")
    grounded = row.get("human_grounded")
    handling = row.get("human_handling_decision")

    if pd.isna(quality) or str(quality).strip() == "":
        return False

    if pd.isna(grounded) or str(grounded).strip() == "":
        return False

    if pd.isna(handling) or str(handling).strip() == "":
        return False

    return True


def get_int_score(prompt, allowed):
    while True:
        value = input(prompt).strip()

        if value in allowed:
            return int(value)

        print(f"Enter one of: {', '.join(allowed)}")


def get_handling():
    while True:
        value = input(
            "Correct handling decision [AUTO_HANDLE/ESCALATE]: "
        ).strip().upper()

        if value in {"AUTO_HANDLE", "ESCALATE"}:
            return value

        print("Enter AUTO_HANDLE or ESCALATE.")


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def display_case(index, row):
    print()
    print("=" * 80)
    print(f"CASE {index}")
    print("=" * 80)

    print(f"Review ID: {safe_text(row.get('review_id'))}")
    print(f"Conversation ID: {safe_text(row.get('conversation_id'))}")
    print(
        f"Evaluation tweet ID: "
        f"{safe_text(row.get('evaluation_incoming_tweet_id'))}"
    )

    print("\nCUSTOMER MESSAGE")
    print("-" * 80)
    print(safe_text(row.get("customer_message")))

    print("\nHUMAN INTENT")
    print("-" * 80)
    print(safe_text(row.get("human_intent")))

    print("\nPREDICTED INTENT")
    print("-" * 80)
    print(
        f"{safe_text(row.get('predicted_intent'))} "
        f"(confidence={safe_text(row.get('intent_confidence'))})"
    )

    print("\nMODEL DECISION")
    print("-" * 80)
    print(
        f"{safe_text(row.get('decision'))} | "
        f"{safe_text(row.get('decision_reason'))}"
    )

    print("\nDRAFT REPLY")
    print("-" * 80)
    print(safe_text(row.get("draft_reply")))

    print("\nTOP RETRIEVED EVIDENCE")
    print("-" * 80)
    print(safe_text(row.get("evidence")))

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("SUPPORTIQ — HUMAN REPLY REVIEW TOOL")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, dtype=str)

    required_columns = {
        "review_id",
        "conversation_id",
        "evaluation_incoming_tweet_id",
        "customer_message",
        "human_intent",
        "predicted_intent",
        "intent_confidence",
        "decision",
        "decision_reason",
        "draft_reply",
        "evidence",
        "human_reply_quality",
        "human_grounded",
        "human_handling_decision",
        "human_notes",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, dtype=str)

        # Keep the original review sample order and rows.
        if len(existing) == len(df):
            for column in [
                "human_reply_quality",
                "human_grounded",
                "human_handling_decision",
                "human_notes",
            ]:
                if column in existing.columns:
                    df[column] = existing[column]

    # Normalize empty strings to actual missing values.
    score_columns = [
        "human_reply_quality",
        "human_grounded",
        "human_handling_decision",
        "human_notes",
    ]

    for column in score_columns:
        df[column] = df[column].replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
            }
        )

    total = len(df)

    print(f"Review cases: {total}")
    print()
    print("Scoring rubric:")
    print("Reply quality: 0 = poor, 1 = acceptable, 2 = strong")
    print("Grounded:       0 = no, 1 = partial, 2 = strong")
    print("Handling:       AUTO_HANDLE or ESCALATE")
    print()
    print(
        "The tool will resume from the first case that has not "
        "been completely scored."
    )

    completed_before = sum(
        is_completed(row)
        for _, row in df.iterrows()
    )

    print(f"Already completed: {completed_before}/{total}")

    for idx in range(total):
        row = df.iloc[idx]

        if is_completed(row):
            continue

        display_case(idx + 1, row)

        print()
        print("REVIEW")

        quality = get_int_score(
            "Reply quality [0/1/2]: ",
            {"0", "1", "2"},
        )

        grounded = get_int_score(
            "Grounded [0/1/2]: ",
            {"0", "1", "2"},
        )

        handling = get_handling()

        notes = input(
            "Notes/reason (brief): "
        ).strip()

        df.at[idx, "human_reply_quality"] = str(quality)
        df.at[idx, "human_grounded"] = str(grounded)
        df.at[idx, "human_handling_decision"] = handling
        df.at[idx, "human_notes"] = notes

        df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        completed_now = sum(
            is_completed(current_row)
            for _, current_row in df.iterrows()
        )

        print()
        print(
            f"Saved. Progress: {completed_now}/{total}"
        )

    final_completed = sum(
        is_completed(row)
        for _, row in df.iterrows()
    )

    print()
    print("=" * 80)
    print("HUMAN REVIEW STATUS")
    print("=" * 80)
    print(f"Completed cases: {final_completed}/{total}")
    print(f"Saved: {OUTPUT_PATH.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()