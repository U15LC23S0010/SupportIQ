from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_historical_customer_response_corpus.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_weak_labels_a1_6.csv"
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


RULES = {
    "product_safety_sensitive": [
        r"\bfood poisoning\b",
        r"\bpoisoned\b",
        r"\bpoisoning\b",
        r"\bforeign object\b",
        r"\bpiece of glass\b",
        r"\bpiece of metal\b",
        r"\bpiece of plastic\b",
        r"\bglass\b.*\b(food|meal|meat|soup|pizza|drink|product)\b",
        r"\bmetal\b.*\b(food|meal|meat|soup|pizza|drink|product)\b",
        r"\bplastic\b.*\b(food|meal|meat|soup|pizza|drink|product)\b",
        r"\bcontaminated food\b",
        r"\bcontaminated product\b",
        r"\bcontamination\b.*\b(food|product|meal)\b",
        r"\ballergic reaction\b",
        r"\bfood allergy\b",
        r"\bchoked on\b",
        r"\bchoking on\b",
        r"\bcut my\b.*\b(food|product)\b",
        r"\bburned\b.*\b(food|product)\b",
    ],

    "delivery_order": [
        r"\bwhere is my order\b",
        r"\bwhere's my order\b",
        r"\bmissing (?:my )?order\b",
        r"\blate .*delivery\b",
        r"\bdelivery .*late\b",
        r"\bdelivery slot\b",
        r"\bdelivery saver\b",
        r"\bdelivery window\b",
        r"\bdelivery date\b",
        r"\bdelivery time\b",
        r"\bhome delivery\b.*\b(?:late|missing|cancel|cancelled|slot)\b",
        r"\border\b.*\b(?:cancel|cancelled|late|missing|track)\b",
        r"\btrack .*order\b",
        r"\bclick and collect\b.*\b(?:pickup|pick up|missing|late|cancel)\b",
        r"\bcollect my order\b",
    ],

    "pricing_payment": [
        r"\bcharged twice\b",
        r"\bdouble charged\b",
        r"\bovercharged\b",
        r"\bcharged too much\b",
        r"\bcharged more than\b",
        r"\bwrong price\b",
        r"\bwrong amount\b",
        r"\bwrong charge\b",
        r"\bprice mismatch\b",
        r"\bprice\b.*\bcharged\b",
        r"\bcharged\b.*\bprice\b",
        r"\bpayment failed\b",
        r"\bpayment declined\b",
        r"\bpayment rejected\b",
        r"\bcard declined\b",
        r"\bcard payment failed\b",
        r"\bcontactless\b.*\bnot working\b",
    ],

    "store_staff_service": [
        r"\brude staff\b",
        r"\bstaff\b.*\brude\b",
        r"\bunhelpful staff\b",
        r"\bstaff\b.*\bunhelpful\b",
        r"\bstaff attitude\b",
        r"\bstaff member\b.*\b(?:rude|attitude|unhelpful)\b",
        r"\bmember of staff\b.*\b(?:rude|attitude|unhelpful)\b",
        r"\bcashier\b.*\b(?:rude|unhelpful|attitude)\b",
        r"\bcheckout staff\b",
        r"\bcheckout queue\b",
        r"\bqueue\b.*\bcheckout\b",
        r"\bstore manager\b.*\b(?:complaint|issue|problem)\b",
        r"\bbranch manager\b.*\b(?:complaint|issue|problem)\b",
        r"\bpoor service\b",
        r"\bbad service\b",
        r"\bterrible service\b",
        r"\bawful service\b",
    ],

    "product_availability": [
        r"\bout of stock\b",
        r"\bsold out\b",
        r"\bback in stock\b",
        r"\bwhen\b.*\bback in stock\b",
        r"\bwhen\b.*\brestocked\b",
        r"\bdo you stock\b",
        r"\bdo you sell\b",
        r"\bwill .* be available\b",
        r"\bis .* available\b",
        r"\bis .* in stock\b",
        r"\bwhich store has\b.*\b(?:product|item|these|this)\b",
        r"\bwhere can i find\b.*\b(?:product|item|these|this)\b",
        r"\bbring back\b.*\b(?:product|food|cheesecake|sandwich|item)\b",
        r"\bstart selling\b.*\b(?:product|food|item)\b",
        r"\bsell .* again\b",
    ],

    "product_quality": [
        r"\bdamaged product\b",
        r"\bdamaged item\b",
        r"\bbroken product\b",
        r"\bbroken item\b",
        r"\bfaulty product\b",
        r"\bfaulty item\b",
        r"\bdefective product\b",
        r"\bdefective item\b",
        r"\bexpired product\b",
        r"\bexpired food\b",
        r"\bout of date\b.*\b(?:food|product|item)\b",
        r"\bstale food\b",
        r"\bspoilt food\b",
        r"\bspoiled food\b",
        r"\bpoor quality\b",
        r"\bquality control\b",
        r"\bproduct\b.*\bnot working\b",
        r"\bitem\b.*\bnot working\b",
        r"\bproduct\b.*\bcracked\b",
        r"\bitem\b.*\bcracked\b",
        r"\bpackaging\b.*\bdamaged\b",
    ],

    "product_information_policy": [
        r"\bwhat are the ingredients\b",
        r"\bwhat ingredients\b",
        r"\bingredients list\b",
        r"\bwhich ingredients\b",
        r"\bwhat allergens\b",
        r"\bwhich allergens\b",
        r"\bnutritional information\b",
        r"\bnutrition facts\b",
        r"\bhow many calories\b",
        r"\bis .* halal\b",
        r"\bis .* vegan\b",
        r"\bis .* vegetarian\b",
        r"\bwhat is the policy\b",
        r"\bwhat are the rules\b",
        r"\bwhat are your rules\b",
        r"\bopening hours\b",
        r"\bclosing hours\b",
        r"\bwhat time\b.*\bopen\b",
        r"\bwhat time\b.*\bclose\b",
        r"\bhow long\b.*\blast\b",
        r"\bhow long\b.*\bvalid\b",
    ],
}


FEEDBACK_FALLBACK_RULES = [
    r"\bfeedback survey\b",
    r"\bonline feedback\b",
    r"\bcompleted .*feedback\b",
    r"\bi have some feedback\b",
    r"\bi have feedback\b",
    r"\bmy feedback\b",
    r"\bplease pass on.*feedback\b",
    r"\bpass on.*feedback\b",
    r"\bgive.*feedback\b",
    r"\bprovide.*feedback\b",
    r"\bmake your feedback\b",
    r"\bcustomer feedback\b",
    r"\bgeneral feedback\b",
    r"\bpass on my suggestion\b",
    r"\bplease pass on.*suggestion\b",
    r"\bmight i suggest\b",
    r"\bi suggest you change\b",
    r"\bi suggest you improve\b",
    r"\bmy suggestion\b",
    r"\bi would suggest\b.*\b(change|improve|revise)\b",
    r"\bsuggest\b.*\b(change|improve|revise)\b",
    r"\bwould love to see\b.*\b(feature|option|function)\b",
    r"\bit would be great if\b.*\b(website|service|feature|option)\b",
    r"\bplease add\b.*\b(feature|filter|function|option|button|page)\b",
    r"\bplease introduce\b.*\b(feature|option|function)\b",
    r"\bcan you add\b.*\b(feature|filter|function|option|button|page)\b",
    r"\bcan you bring back\b.*\b(feature|function|shopping list)\b",
]

CONCRETE_ISSUE_PATTERNS = [
    r"\border\b",
    r"\bparcel\b",
    r"\bdelivery\b",
    r"\bcharged\b",
    r"\bcharge\b",
    r"\bpayment\b",
    r"\brefund\b",
    r"\bstock\b",
    r"\bavailable\b",
    r"\bavailability\b",
    r"\bsell\b",
    r"\bdamaged\b",
    r"\bbroken\b",
    r"\bfaulty\b",
    r"\bexpired\b",
    r"\bwebsite\b",
    r"\bweb site\b",
    r"\bapp\b",
    r"\bapplication\b",
    r"\blogin\b",
    r"\bpassword\b",
    r"\bstaff\b",
    r"\bcashier\b",
    r"\bservice\b",
    r"\bstore\b.*\bproblem\b",
    r"\bstore\b.*\bissue\b",
    r"\berror\b",
]

COMPILED_RULES = {
    intent: [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in patterns
    ]
    for intent, patterns in RULES.items()
}

COMPILED_FEEDBACK = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in FEEDBACK_FALLBACK_RULES
]

COMPILED_CONCRETE = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in CONCRETE_ISSUE_PATTERNS
]


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def matched_intents(text):
    clean = clean_text(text)
    matches = []

    for intent in TARGET_INTENTS[:-1]:
        for pattern in COMPILED_RULES[intent]:
            if pattern.search(clean):
                matches.append(intent)
                break

    return matches


def has_feedback_fallback(text):
    clean = clean_text(text)

    return any(
        pattern.search(clean)
        for pattern in COMPILED_FEEDBACK
    )


def has_concrete_issue(text):
    clean = clean_text(text)

    return any(
        pattern.search(clean)
        for pattern in COMPILED_CONCRETE
    )


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(CORPUS_PATH)

    corpus = pd.read_csv(
        CORPUS_PATH,
        dtype=str,
    )

    corpus["customer_message_clean"] = corpus[
        "customer_message"
    ].map(clean_text)

    corpus["matched_intents"] = corpus[
        "customer_message_clean"
    ].map(matched_intents)

    corpus["weak_label_count"] = corpus[
        "matched_intents"
    ].map(len)

    corpus["weak_intent"] = "uncertain"
    corpus["weak_label_status"] = "unmatched"

    # Concrete non-feedback intents first.
    single_mask = corpus["weak_label_count"] == 1

    corpus.loc[
        single_mask,
        "weak_intent"
    ] = corpus.loc[
        single_mask,
        "matched_intents"
    ].map(lambda values: values[0])

    corpus.loc[
        single_mask,
        "weak_label_status"
    ] = "accepted"

    # Safety takes precedence.
    safety_mask = corpus["matched_intents"].map(
        lambda values: "product_safety_sensitive" in values
    )

    corpus.loc[
        safety_mask,
        "weak_intent"
    ] = "product_safety_sensitive"

    corpus.loc[
        safety_mask,
        "weak_label_status"
    ] = "accepted"

    # Feedback is a fallback only if no concrete target intent matched
    # and the wording does not contain obvious issue/remedy language.
    feedback_mask = (
        (corpus["weak_label_status"] == "unmatched")
        & corpus["customer_message_clean"].map(
            has_feedback_fallback
        )
        & ~corpus["customer_message_clean"].map(
            has_concrete_issue
        )
    )

    corpus.loc[
        feedback_mask,
        "weak_intent"
    ] = "feedback_suggestion"

    corpus.loc[
        feedback_mask,
        "weak_label_status"
    ] = "accepted"

    output = corpus[
        [
            "conversation_id",
            "customer_tweet_id",
            "customer_message",
            "customer_message_clean",
            "tesco_tweet_id",
            "tesco_response",
            "matched_intents",
            "weak_intent",
            "weak_label_count",
            "weak_label_status",
        ]
    ].copy()

    output["matched_intents"] = output[
        "matched_intents"
    ].map(
        lambda values: "|".join(values)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    accepted = output[
        output["weak_label_status"] == "accepted"
    ]

    print("=" * 80)
    print("TESCO A1.6 HIGH-PRECISION WEAK LABELING")
    print("=" * 80)
    print(f"Corpus rows:          {len(output):,}")
    print(f"Accepted:             {len(accepted):,}")
    print(
        "Unmatched:            "
        f"{(output['weak_label_status'] == 'unmatched').sum():,}"
    )
    print(
        "Conflicts:            "
        f"{(output['weak_label_status'] == 'conflict').sum():,}"
    )
    print()
    print("Accepted labels:")
    print(
        accepted["weak_intent"]
        .value_counts()
        .reindex(TARGET_INTENTS, fill_value=0)
        .to_string()
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
