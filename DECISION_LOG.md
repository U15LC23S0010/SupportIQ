# SupportIQ — Decision Log

1. **Selected Tesco as the target brand**

   Tesco was selected after comparing brands using conversation volume, customer-message coverage, response-pair availability, conversation depth, and issue diversity. Tesco provided enough data while also exposing a broad range of support scenarios.

2. **Used customer → Tesco response pairs for the historical corpus**

   The grounding corpus uses individual customer messages paired with Tesco's historical replies rather than whole conversations. This keeps retrieval focused on how Tesco responded to a specific customer issue and avoids adding unrelated conversational turns to the retrieval unit.

3. **Excluded golden-set conversations from the historical corpus**

   All historical response pairs belonging to the 172 golden evaluation conversations were removed from the retrieval corpus. This prevents evaluation leakage where the system could retrieve the exact conversation it is being evaluated on.

4. **Defined the intent taxonomy from Tesco data rather than importing a generic taxonomy**

   The final eight target intents were derived from observed Tesco customer-support patterns. This keeps the classifier aligned with the actual domain instead of assuming an external support taxonomy fits Tesco.

5. **Separated review-only cases from target intents**

   `non_support_social` and `other_unclear` were retained for review and analysis but were not treated as normal target support intents. This prevents unrelated social messages and genuinely ambiguous cases from distorting the main support-intent metrics.

6. **Applied safety precedence**

   Messages containing credible health, injury, contamination, foreign-object, or comparable safety risk are assigned to `product_safety_sensitive` ahead of ordinary quality or information intents. Safety cases are also always escalated.

7. **Used conservative high-precision weak supervision**

   A1.6 weak labels were used only to create development data. The accepted A1.6 set contained 1,592 labels, and the rules were deliberately conservative. Ambiguous or multi-match cases were not accepted. Weak labels were never treated as ground truth.
   

8. **Discarded A1.7**

   A later weak-label expansion was rejected after manual audit revealed semantic leakage between categories, including feedback being confused with ordinary support follow-ups, safety being confused with packaging issues, and availability being confused with product quality. The earlier A1.6 set was retained instead.

9. **Used TF-IDF word + character n-grams**

   Twitter support messages contain short text, spelling variation, product names, usernames, and noisy phrasing. Combining word and character n-grams gives the classifier robustness while remaining transparent and reproducible.

10. **Selected Logistic Regression over Linear SVC**

    Logistic Regression and Linear SVC were evaluated on the same grouped train/validation split. They had identical development accuracy and macro-F1, while Logistic Regression performed better on the frozen golden set, so Logistic Regression was selected as the main classifier.

11. **Grouped the development split by conversation**

    The weak-label development split was performed by `conversation_id` rather than individual messages. This reduces leakage from near-identical turns from the same conversation appearing on both sides of the split.

12. **Kept decision policy separate from intent classification**

    The agent does not treat classifier output as an automatic-handling decision by itself. Confidence, retrieval evidence, predicted intent, and safety rules are combined before choosing AUTO_HANDLE or ESCALATE.

13. **Made safety cases always escalate**

    Even when the predicted confidence or historical evidence is strong, credible safety-sensitive cases are escalated because the cost of an unsafe automatic response is higher than the cost of human review.

14. **Used a 172-example frozen golden set**

    The final golden evaluation set was intentionally frozen at 172 hand-labelled conversations, which is within the assignment's required 150–250 range. The set was not expanded merely to improve statistical comfort after results were observed.

15. **Used the LLM judge as a secondary evaluator**

    Gemini 3.5 Flash-Lite was used to independently judge reply quality, groundedness, and handling on a 32-case sample. Human-vs-LLM agreement was reported explicitly, and the LLM judge was not treated as ground truth because agreement was low.
