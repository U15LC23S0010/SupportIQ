from pathlib import Path
import json
import os
import time

import pandas as pd
from google import genai


INPUT_PATH = Path(
    "results/tables/supportiq_llm_judge_batch_32.jsonl"
)

OUTPUT_PATH = Path(
    "results/tables/supportiq_llm_judgments_32.csv"
)

MODEL_NAME = "gemini-3.5-flash-lite"

REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3


EXPECTED_COLUMNS = [
    "review_id",
    "llm_reply_quality",
    "llm_grounded",
    "llm_handling_decision",
    "llm_reason",
]


def load_prompts():
    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 32:
        raise ValueError(
            f"Expected 32 judge prompts, found {len(records)}"
        )

    return records


def load_existing_output():
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    df = pd.read_csv(
        OUTPUT_PATH,
        dtype=str,
    )

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[EXPECTED_COLUMNS].copy()


def extract_json(text):
    text = text.strip()

    # Direct JSON response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown fences if the model adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Find first JSON object inside surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse valid JSON from Gemini response."
    )


def validate_judgment(data):
    required = {
        "llm_reply_quality",
        "llm_grounded",
        "llm_handling_decision",
        "llm_reason",
    }

    missing = required - set(data.keys())

    if missing:
        raise ValueError(
            f"Missing judge fields: {sorted(missing)}"
        )

    quality = int(data["llm_reply_quality"])
    grounded = int(data["llm_grounded"])
    handling = str(
        data["llm_handling_decision"]
    ).strip().upper()
    reason = str(
        data["llm_reason"]
    ).strip()

    if quality not in {0, 1, 2}:
        raise ValueError(
            f"Invalid llm_reply_quality: {quality}"
        )

    if grounded not in {0, 1, 2}:
        raise ValueError(
            f"Invalid llm_grounded: {grounded}"
        )

    if handling not in {
        "AUTO_HANDLE",
        "ESCALATE",
    }:
        raise ValueError(
            f"Invalid llm_handling_decision: {handling}"
        )

    if not reason:
        raise ValueError(
            "llm_reason is empty."
        )

    return {
        "llm_reply_quality": quality,
        "llm_grounded": grounded,
        "llm_handling_decision": handling,
        "llm_reason": reason,
    }


def save_output(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    prompts = load_prompts()
    df = load_existing_output()

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    completed = {
    int(row["review_id"])
    for _, row in df.iterrows()
    if (
        pd.notna(row["llm_reply_quality"])
        and str(row["llm_reply_quality"]).strip()
        and pd.notna(row["llm_grounded"])
        and str(row["llm_grounded"]).strip()
        and pd.notna(row["llm_handling_decision"])
        and str(row["llm_handling_decision"]).strip()
        and pd.notna(row["llm_reason"])
        and str(row["llm_reason"]).strip()
    )
}

    print("=" * 80)
    print("SUPPORTIQ — GEMINI LLM JUDGE")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Total cases: {len(prompts)}")
    print(f"Already completed: {len(completed)}")
    print(
        "Human scores are NOT included in the judge prompts."
    )
    print("=" * 80)

    for index, record in enumerate(
        prompts,
        start=1,
    ):
        review_id = int(record["review_id"])

        if review_id in completed:
            continue

        prompt = record["prompt"]

        print()
        print(
            f"Judging case {index}/32 "
            f"(review_id={review_id})..."
        )

        last_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )

                if not response.text:
                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                raw = extract_json(
                    response.text
                )

                judgment = validate_judgment(
                    raw
                )

                mask = (
                    df["review_id"].astype(str)
                    == str(review_id)
                )

                if not mask.any():
                    new_row = {
                    "review_id": str(review_id),
             "llm_reply_quality": str(judgment["llm_reply_quality"]),
            "llm_grounded": str(judgment["llm_grounded"]),
            "llm_handling_decision": str(
            judgment["llm_handling_decision"]
          ),
    "llm_reason": str(judgment["llm_reason"]),
      }

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                [new_row]
                            ),
                        ],
                        ignore_index=True,
                    )
                else:
                    for key, value in judgment.items():
                        df.loc[
                            mask,
                            key,
                        ] = str(value)

                df = df.sort_values(
                    "review_id"
                ).reset_index(
                    drop=True
                )

                save_output(df)

                completed.add(review_id)

                print(
                    f"Saved case {review_id}: "
                    f"quality={judgment['llm_reply_quality']}, "
                    f"grounded={judgment['llm_grounded']}, "
                    f"handling={judgment['llm_handling_decision']}"
                )

                break

            except Exception as exc:
                last_error = exc

                print(
                    f"Attempt {attempt}/{MAX_RETRIES} "
                    f"failed: {exc}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(
                        REQUEST_DELAY_SECONDS * attempt
                    )

        else:
            raise RuntimeError(
                f"Could not judge review_id={review_id}. "
                f"Last error: {last_error}"
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    final_df = pd.read_csv(
        OUTPUT_PATH,
        dtype=str,
    )

    complete_mask = (
        final_df["llm_reply_quality"].notna()
        & final_df["llm_grounded"].notna()
        & final_df["llm_handling_decision"].notna()
        & final_df["llm_reason"].notna()
        & (
            final_df["llm_reply_quality"]
            .astype(str)
            .str.strip()
            != ""
        )
        & (
            final_df["llm_grounded"]
            .astype(str)
            .str.strip()
            != ""
        )
        & (
            final_df["llm_handling_decision"]
            .astype(str)
            .str.strip()
            != ""
        )
        & (
            final_df["llm_reason"]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    print()
    print("=" * 80)
    print("GEMINI JUDGE COMPLETE")
    print("=" * 80)
    print(
        f"Completed cases: "
        f"{int(complete_mask.sum())}/32"
    )
    print(
        f"Saved: {OUTPUT_PATH.resolve()}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()