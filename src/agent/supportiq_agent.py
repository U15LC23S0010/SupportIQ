from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.retrieval.tesco_retriever import TescoHistoricalRetriever


MODEL_DIR = Path("results/models")

INTENT_MODEL_PATH = (
    MODEL_DIR
    / "tfidf_logistic_regression_shared_split.joblib"
)

INTENT_VECTORIZER_PATH = (
    MODEL_DIR
    / "tfidf_logistic_regression_shared_split_vectorizer.joblib"
)

SAFETY_INTENT = "product_safety_sensitive"

AUTO_HANDLE_INTENTS = {
    "delivery_order",
    "pricing_payment",
    "store_staff_service",
    "product_availability",
    "product_quality",
    "product_information_policy",
    "feedback_suggestion",
}

INTENT_CONFIDENCE_THRESHOLD = 0.70
RETRIEVAL_SCORE_THRESHOLD = 0.25

INTENT_EVIDENCE_PATTERNS = {
    "delivery_order": [
        r"\bdeliver",
        r"\bdelivery",
        r"\border\b",
        r"\bordered\b",
        r"\bgrocery\b",
        r"\bmissing item",
        r"\bitems? missing",
        r"\bredeliver",
        r"\bslot\b",
        r"\bcourier\b",
    ],
    "pricing_payment": [
        r"\bprice\b",
        r"\bpriced\b",
        r"\bovercharg",
        r"\bcharged\b",
        r"\bpayment\b",
        r"\brefund\b",
        r"\bdiscount\b",
        r"\bpromotion\b",
        r"\bpay at pump\b",
        r"\bdebited\b",
    ],
    "store_staff_service": [
        r"\bstore\b",
        r"\bbranch\b",
        r"\bcolleague\b",
        r"\bstaff\b",
        r"\bmanager\b",
        r"\bservice\b",
        r"\bcustomer service\b",
        r"\bdriver\b",
    ],
    "product_availability": [
        r"\bin stock\b",
        r"\bout of stock\b",
        r"\bstock\b",
        r"\bsold out\b",
        r"\bavailable\b",
        r"\bavailability\b",
        r"\bcan't find\b",
        r"\bcannot find\b",
    ],
    "product_quality": [
        r"\bquality\b",
        r"\bstale\b",
        r"\brot(?:ten)?\b",
        r"\bout of date\b",
        r"\bexpired\b",
        r"\bdamaged\b",
        r"\bbroken\b",
        r"\btaste\b",
        r"\bflavour\b",
        r"\bflavor\b",
        r"\bmissing\b",
        r"\bempty\b",
    ],
    "product_safety_sensitive": [
        r"\bsafety\b",
        r"\bsafe\b",
        r"\bunsafe\b",
        r"\bforeign object\b",
        r"\bforeign body\b",
        r"\bmetal\b",
        r"\bplastic\b",
        r"\bglass\b",
        r"\bbug\b",
        r"\bworm\b",
        r"\bslug\b",
        r"\bmould\b",
        r"\bmold\b",
        r"\bchok",
        r"\binjury\b",
        r"\bhospital\b",
        r"\ba&e\b",
        r"\ballergic\b",
    ],
    "product_information_policy": [
        r"\bpolicy\b",
        r"\brules\b",
        r"\bingredients?\b",
        r"\ballergen\b",
        r"\bvegan\b",
        r"\bvegetarian\b",
        r"\bhalal\b",
        r"\bsuitable\b",
        r"\bcontain\b",
        r"\bopening hours?\b",
        r"\bwhat time\b",
        r"\bwhen can i\b",
        r"\bhow do i\b",
        r"\bhow should\b",
        r"\bwhy is\b",
        r"\bis .* supposed to\b",
    ],
    "feedback_suggestion": [
        r"\bfeedback\b",
        r"\bsuggest\b",
        r"\bsuggestion\b",
        r"\brecommend\b",
        r"\bpraise\b",
        r"\bthank(?:s| you)\b",
        r"\blove\b",
        r"\bfantastic\b",
        r"\bexcellent\b",
        r"\bgreat service\b",
        r"\bwell done\b",
        r"\bplease consider\b",
        r"\bwould be nice\b",
    ],
}


def compile_patterns(
    patterns: list[str],
) -> re.Pattern[str]:
    return re.compile(
        "|".join(
            f"(?:{pattern})"
            for pattern in patterns
        ),
        re.IGNORECASE,
    )


COMPILED_EVIDENCE_PATTERNS = {
    intent: compile_patterns(patterns)
    for intent, patterns in INTENT_EVIDENCE_PATTERNS.items()
}


class SupportIQAgent:
    def __init__(
        self,
        intent_model_path: Path = INTENT_MODEL_PATH,
        intent_vectorizer_path: Path = INTENT_VECTORIZER_PATH,
        retriever: TescoHistoricalRetriever | None = None,
    ) -> None:
        self.intent_model = joblib.load(
            intent_model_path
        )

        self.intent_vectorizer = joblib.load(
            intent_vectorizer_path
        )

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = TescoHistoricalRetriever()

    def classify_intent(
        self,
        message: str,
    ) -> dict[str, Any]:
        if not message or not message.strip():
            return {
                "intent": None,
                "confidence": 0.0,
            }

        vector = self.intent_vectorizer.transform(
            [message]
        )

        intent = self.intent_model.predict(
            vector
        )[0]

        confidence = 0.0

        if hasattr(
            self.intent_model,
            "predict_proba",
        ):
            probabilities = (
                self.intent_model
                .predict_proba(vector)[0]
            )

            confidence = float(
                np.max(probabilities)
            )

        return {
            "intent": str(intent),
            "confidence": confidence,
        }

    def retrieve_evidence(
        self,
        message: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(
            message,
            top_k=top_k,
            min_score=0.0,
        )

    def calculate_evidence_consistency(
        self,
        intent: str | None,
        evidence: list[dict[str, Any]],
    ) -> float:
        if not intent or not evidence:
            return 0.0

        pattern = COMPILED_EVIDENCE_PATTERNS.get(
            intent
        )

        if pattern is None:
            return 0.0

        matching_cases = 0

        for item in evidence[:3]:
            text = (
                str(item["customer_message"])
                + " "
                + str(item["tesco_response"])
            )

            if pattern.search(text):
                matching_cases += 1

        return matching_cases / min(
            len(evidence[:3]),
            3,
        )

    def decide_handling(
        self,
        intent: str | None,
        confidence: float,
        evidence: list[dict[str, Any]],
        evidence_consistency: float,
    ) -> dict[str, Any]:
        if not intent:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Unable to classify the customer message."
                ),
            }

        if intent == SAFETY_INTENT:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Safety-sensitive issue requires human "
                    "review before responding."
                ),
            }

        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Intent confidence is below the "
                    f"{INTENT_CONFIDENCE_THRESHOLD:.2f} threshold."
                ),
            }

        if not evidence:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "No historical Tesco handling example "
                    "was retrieved."
                ),
            }

        top_similarity = float(
            evidence[0]["similarity"]
        )

        if top_similarity < RETRIEVAL_SCORE_THRESHOLD:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Historical evidence is too weak to "
                    "support an automatic response."
                ),
            }

        if evidence_consistency < 0.34:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Retrieved historical cases do not provide "
                    "sufficient evidence for the predicted intent."
                ),
            }

        if intent not in AUTO_HANDLE_INTENTS:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Intent is outside the routine "
                    "auto-handle policy."
                ),
            }

        return {
            "decision": "AUTO_HANDLE",
            "reason": (
                "Routine intent with sufficient classifier "
                "confidence, retrieval strength, and "
                "intent-evidence consistency."
            ),
        }

    def infer_historical_action(
        self,
        evidence: list[dict[str, Any]],
    ) -> str:
        responses = " ".join(
            item["tesco_response"]
            for item in evidence[:3]
        ).lower()

        if re.search(
            r"\bdm\b|\bdirect message\b",
            responses,
        ):
            return (
                "Similar Tesco cases were commonly followed "
                "by a request for additional customer or order details."
            )

        if re.search(
            r"\breturn\b.*\bstore\b"
            r"|\bback to (?:the )?store\b",
            responses,
        ):
            return (
                "Similar Tesco cases were commonly handled by "
                "asking the customer to return the item to a store."
            )

        if re.search(
            r"\brefund\b|\bmoneycard\b|\brefund you\b",
            responses,
        ):
            return (
                "Similar Tesco cases were commonly handled through "
                "a refund or compensation process after the purchase "
                "details were checked."
            )

        if re.search(
            r"\breceipt\b|\bbarcode\b|\bproof of purchase\b",
            responses,
        ):
            return (
                "Similar Tesco cases commonly involved checking "
                "the receipt, barcode, or other purchase details."
            )

        if re.search(
            r"\bstore\b.*\bmanager\b"
            r"|\bmanagement team\b",
            responses,
        ):
            return (
                "Similar cases were commonly passed to the relevant "
                "store or management team for follow-up."
            )

        if re.search(
            r"\blook into this\b"
            r"|\binvestigate\b"
            r"|\binvestigation\b",
            responses,
        ):
            return (
                "Similar Tesco cases were handled by investigating "
                "the issue and requesting the relevant details."
            )

        return (
            "Similar historical Tesco cases were handled by "
            "reviewing the issue and gathering the relevant details."
        )

    def draft_reply(
        self,
        intent: str | None,
        evidence: list[dict[str, Any]],
        decision: str,
    ) -> str:
        if decision == "ESCALATE":
            if intent == SAFETY_INTENT:
                return (
                    "Thanks for contacting Tesco. "
                    "Because your message concerns a potentially "
                    "safety-sensitive product issue, a colleague "
                    "should review this before any further action."
                )

            return (
                "Thanks for contacting Tesco. "
                "We need a colleague to review your message "
                "before providing the appropriate response."
            )

        historical_action = self.infer_historical_action(
            evidence
        )

        if intent == "pricing_payment":
            return (
                "Sorry to hear about the pricing or payment issue. "
                "We'd like to check this for you. "
                "Please provide the relevant purchase details "
                "and, where available, your receipt or order "
                "information. "
                f"{historical_action}"
            )

        if intent == "delivery_order":
            return (
                "Sorry to hear about the issue with your delivery. "
                "Please provide your order details and let us know "
                "which item or items were affected so we can check "
                f"this. {historical_action}"
            )

        if intent == "product_availability":
            return (
                "Thanks for getting in touch. "
                "Please let us know the product and the store or "
                "area you are looking for, so we can check the "
                f"availability. {historical_action}"
            )

        if intent == "product_quality":
            return (
                "Sorry to hear that the product did not meet "
                "expectations. Please provide the product and "
                "purchase details so this can be looked into. "
                f"{historical_action}"
            )

        if intent == "store_staff_service":
            return (
                "Sorry to hear about your experience in store. "
                "Please provide the store and details of what "
                "happened so the relevant team can review it. "
                f"{historical_action}"
            )

        if intent == "product_information_policy":
            return (
                "Thanks for your question. "
                "We'd like to check the relevant Tesco product "
                "or policy information before giving you an "
                f"answer. {historical_action}"
            )

        if intent == "feedback_suggestion":
            return (
                "Thanks for taking the time to share your feedback. "
                "We'll make sure your comments are recorded for "
                f"review. {historical_action}"
            )

        return (
            "Thanks for contacting Tesco. "
            "We'll review your message and the relevant details."
        )

    def analyze(
        self,
        message: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        classification = self.classify_intent(
            message
        )

        evidence = self.retrieve_evidence(
            message,
            top_k=top_k,
        )

        evidence_consistency = (
            self.calculate_evidence_consistency(
                classification["intent"],
                evidence,
            )
        )

        handling = self.decide_handling(
            intent=classification["intent"],
            confidence=classification["confidence"],
            evidence=evidence,
            evidence_consistency=evidence_consistency,
        )

        reply = self.draft_reply(
            intent=classification["intent"],
            evidence=evidence,
            decision=handling["decision"],
        )

        evidence_summary = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            evidence_summary.append(
                {
                    "rank": index,
                    "similarity": item["similarity"],
                    "conversation_id": item[
                        "conversation_id"
                    ],
                    "customer_message": item[
                        "customer_message"
                    ],
                    "tesco_response": item[
                        "tesco_response"
                    ],
                }
            )

        return {
            "message": message,
            "intent": classification["intent"],
            "intent_confidence": classification[
                "confidence"
            ],
            "evidence_consistency": evidence_consistency,
            "decision": handling["decision"],
            "decision_reason": handling["reason"],
            "draft_reply": reply,
            "evidence": evidence_summary,
        }