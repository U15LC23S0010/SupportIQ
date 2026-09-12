from pathlib import Path
import json
import pandas as pd


INPUT_PATH = Path(
    "results/tables/supportiq_reply_human_review_sample_32_scored.csv"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_llm_judge_batch_32.jsonl"
)


JUDGE_INSTRUCTIONS = """
You are an independent evaluator for an AI customer-support agent.

Evaluate ONLY the information supplied in the case.

Do NOT use:
- the human review scores
- the human review notes
- external knowledge
- assumptions about the brand beyond the supplied evidence

Evaluate three dimensions.

1. REPLY QUALITY
0 = poor, incorrect, irrelevant, unsafe, or fails to address the customer
1 = acceptable but incomplete, generic, or weakly useful
2 = strong, specific, appropriate, and useful

2. GROUNDING
0 = the draft reply is not supported by the supplied historical evidence
1 = partially supported or only loosely connected to the evidence
2 = clearly supported by the supplied historical evidence

3. HANDLING DECISION
Choose the decision you believe the agent should make:
AUTO_HANDLE = routine issue with enough evidence for an appropriate response
ESCALATE = human review is preferable because of risk, ambiguity, missing information,
or insufficiently reliable evidence

Return ONLY valid JSON with these fields:

{
  "llm_reply_quality": 0,
  "llm_grounded": 0,
  "llm_handling_decision": "AUTO_HANDLE",
  "llm_reason": "brief explanation"
}

Do not include any other fields.
""".strip()


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def build_prompt(row):
    customer_message = safe_text(row["customer_message"])
    human_intent = safe_text(row["human_intent"])
    predicted_intent = safe_text(row["predicted_intent"])
    intent_confidence = safe_text(row["intent_confidence"])
    evidence_consistency = safe_text(row["evidence_consistency"])
    model_decision = safe_text(row["decision"])
    decision_reason = safe_text(row["decision_reason"])
    draft_reply = safe_text(row["draft_reply"])
    evidence = safe_text(row["evidence"])

    return f"""
{JUDGE_INSTRUCTIONS}

CASE

Customer message:
{customer_message}

Frozen human intent label:
{human_intent}

Agent predicted intent:
{predicted_intent}

Agent intent confidence:
{intent_confidence}

Agent evidence consistency:
{evidence_consistency}

Agent handling decision:
{model_decision}

Agent decision reason:
{decision_reason}

Agent draft reply:
{draft_reply}

Historical retrieved evidence:
{evidence}
""".strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "review_id",
        "customer_message",
        "human_intent",
        "predicted_intent",
        "intent_confidence",
        "evidence_consistency",
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

    if len(df) != 32:
        raise ValueError(
            f"Expected exactly 32 review cases, found {len(df)}"
        )

    records = []

    for _, row in df.iterrows():
        record = {
            "review_id": int(row["review_id"]),
            "prompt": build_prompt(row),
        }

        records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 80)
    print("SUPPORTIQ — LLM JUDGE BATCH PREPARATION")
    print("=" * 80)
    print(f"Input cases: {len(records)}")
    print(f"Output: {OUTPUT_PATH.resolve()}")
    print()
    print("Human scores were NOT included in judge prompts.")
    print("No API calls were made.")
    print("No LLM scores were fabricated.")
    print()
    print("Prepared review IDs:")
    print([record["review_id"] for record in records])
    print("=" * 80)


if __name__ == "__main__":
    main()