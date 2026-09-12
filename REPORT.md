# SupportIQ — Evidence-Grounded AI Customer Support Agent

## 1. Problem framing

SupportIQ is an evidence-grounded customer-support agent built on the Kaggle Customer Support on Twitter (TWCS) dataset.

The agent has three responsibilities:

1. classify a customer message into a small brand-specific intent taxonomy;
2. retrieve historically similar customer → Tesco response examples;
3. decide whether the case should be AUTO_HANDLE or ESCALATE and draft a response grounded in historical handling.

### Why Tesco?

Tesco was selected after comparing major support accounts on conversation volume, customer-message volume, support-message volume, conversation depth, and issue diversity.

Tesco provided a useful balance of sufficient historical volume and varied support situations, including delivery, pricing, products, stores, complaints, refunds, availability, and operational questions.

The selected Tesco corpus contains 16,722 reconstructed customer-support conversations and 38,468 valid customer → Tesco response relationships before golden-set exclusion.

## 2. Data and evaluation design

The full TWCS dataset contains 2,811,774 tweets.

To avoid evaluation leakage, the historical response corpus was constructed from customer → Tesco response pairs and all conversations belonging to the frozen 172-case golden evaluation set were excluded from retrieval.

The resulting historical response corpus contains:

* 37,855 usable historical customer → Tesco pairs
* 0 golden conversation IDs remaining after leakage filtering
* 0 rows missing customer or Tesco response text

### Intent taxonomy

The target taxonomy contains eight Tesco-specific intents:

* delivery_order
* pricing_payment
* store_staff_service
* product_availability
* product_quality
* product_safety_sensitive
* product_information_policy
* feedback_suggestion

Two review-only categories were retained for annotation analysis:

* non_support_social
* other_unclear

The taxonomy and annotation guidelines were frozen before the final golden set was created.

### Golden evaluation set

The final frozen golden set contains **172 hand-labelled conversations**.

Of these:

* 151 belong to target intents
* 21 are review-only cases

The golden set was intentionally frozen at 172 because the assignment requires 150–250 hand-labelled examples.

### Development data

A conservative high-precision weak-supervision process produced A1.6 with:

* 1,592 accepted weak labels
* 36,263 unmatched messages discarded

Weak labels were used for model development only and were not treated as ground truth.

## 3. Model and retrieval architecture

### Intent classifier

The primary classifier is:

**TF-IDF word + character n-grams + Logistic Regression with balanced class weights**

The choice was motivated by:

* short and noisy Twitter text;
* strong lexical signals;
* class imbalance;
* transparency and reproducibility.

A second baseline used LinearSVC with the same general TF-IDF feature design.

Both models were evaluated on the same GroupShuffleSplit partition.

### Retrieval

Historical responses are indexed using TF-IDF and cosine similarity.

Retrieval is performed over customer messages and returns historical Tesco responses associated with the most similar customer cases.

Results are deduplicated by historical conversation/customer-message identity to avoid allowing repeated responses from the same case to dominate the evidence set.

### Decision policy

The agent currently escalates when:

* predicted intent is safety-sensitive;
* intent confidence is below 0.70;
* no usable evidence is retrieved;
* retrieval similarity is too weak;
* evidence consistency is too weak;
* the predicted intent is outside the routine auto-handle set.

Otherwise it may AUTO_HANDLE.

This policy is deliberately conservative for safety-sensitive cases.

## 4. Results

### Baseline comparison

The evaluation uses the trivial majority-class baseline required by the assignment, a simple learned baseline, and the final selected model.

The majority-class baseline always predicts `delivery_order`, which was the most frequent accepted intent in the A1.6 development labels.

| Model                        | Dev Accuracy | Dev Macro-F1 | Gold Accuracy | Gold Macro-F1 |
| ---------------------------- | -----------: | -----------: | ------------: | ------------: |
| Majority-class baseline      |            — |            — |        17.22% |         3.67% |
| TF-IDF + LinearSVC           |       93.53% |       80.83% |        44.37% |        34.57% |
| TF-IDF + Logistic Regression |       93.53% |       80.83% |    **47.68%** |    **37.53%** |

The majority-class baseline provides a trivial reference point. LinearSVC is the simple learned baseline. Logistic Regression was selected because it performed best on the frozen golden set.

On the 172-case frozen evaluation set:

* intent accuracy: **47.68%**
* intent macro-F1: **37.53%**
* AUTO_HANDLE: **6**
* ESCALATE: **166**
* safety-sensitive escalation rate: **100%**
* mean evidence consistency: **0.2791**

The large escalation count shows that the current policy is conservative and that the confidence threshold is a major bottleneck.

## 5. Human reply-quality review

A stratified sample of 32 golden cases was manually reviewed, with four cases from each target intent.

The human reviewer scored:

* reply quality from 0–2;
* grounding from 0–2;
* preferred handling decision.

Results:

* mean reply quality: **0.9688 / 2**
* mean groundedness: **1.8125 / 2**
* human preferred AUTO_HANDLE: **12 / 32**
* human preferred ESCALATE: **20 / 32**
* intent correct: **13 / 32 (40.63%)**
* intent incorrect: **19 / 32 (59.38%)**

The human review showed that useful historical evidence did not necessarily lead to a good final reply, particularly when the classifier intent was wrong.

Historical evidence was often judged useful even when the classifier intent was incorrect, but the deterministic reply generator frequently failed to turn that evidence into a specific customer-facing answer.

## 6. LLM judge and human-vs-LLM agreement

An independent LLM judge was executed on the same 32 cases used for human review using Gemini 3.5 Flash-Lite.

The judge received the customer message, the agent's predicted intent and decision, the draft reply, and retrieved historical evidence. Human review scores were withheld from the judge.

| Dimension         | Exact Agreement |  Kappa |
| ----------------- | --------------: | -----: |
| Reply quality     |           12.5% |  0.061 |
| Groundedness      |           46.9% | -0.107 |
| Handling decision |           65.6% |  0.137 |

Reply quality and groundedness use quadratic weighted Cohen's kappa because their labels are ordinal. Handling decision uses ordinary Cohen's kappa.

The agreement is low, so the LLM judge is not treated as ground truth. The statistics are descriptive because the review set contains only 32 cases and the human ratings are highly concentrated. In particular, 31 of 32 human reply-quality ratings were 1.

The disagreement itself is informative: automated judging of customer-support reply quality and grounding is sensitive to rubric interpretation, and human evaluation remains important for this task.

LLM-judge artifacts:

* `results/tables/supportiq_llm_judgments_32.csv`
* `results/tables/supportiq_human_llm_agreement.json`
* `results/tables/supportiq_human_llm_agreement.txt`

## 7. Top five failure examples

### Failure 1 — High-confidence wrong intent with excellent evidence

Case 8 asked when Christmas delivery slots would become available for Delivery Saver subscribers.

The classifier predicted `delivery_order` with **0.8999 confidence**, while the human intent was `product_information_policy`.

The retrieved evidence was excellent and contained directly matching Tesco cases answering Christmas-slot timing.

The failure was therefore not lack of evidence; it was the inability to correctly frame the customer request and then use the retrieved evidence in the final reply.

### Failure 2 — Product quality misclassification

Case 12 asked whether Tesco crisps were supposed to taste stale.

The model predicted `delivery_order`.

A related historical stale-product example existed in the retrieved evidence, but the final response remained a generic escalation message.

### Failure 3 — Availability misclassified as safety

Case 14 asked whether Tesco had discontinued sweet chilli tortilla wraps.

The classifier predicted `product_safety_sensitive`.

Historical examples showed Tesco handling similar discontinued-product questions through availability checks.

The emotional wording and emoji appear to have contributed to a poor intent boundary.

### Failure 4 — Positive feedback misclassified as safety

Case 25 was positive feedback recommending Tesco vegan wine.

The classifier predicted `product_safety_sensitive`.

This is a particularly important failure because an ordinary praise message can incorrectly enter a safety escalation path.

### Failure 5 — Pricing misclassified as availability

Case 11 asked why identical products had different prices.

The classifier predicted `product_availability`.

The historical evidence included a closely matching Tesco case about different prices for the same product, showing that retrieval contained a useful signal despite the incorrect classifier decision.

## 8. What is misleading about my headline number?

The headline frozen-gold intent accuracy is **47.68%**, but this number alone does not describe the full agent.

The 32-case human review showed:

* mean reply quality: **0.9688 / 2**
* mean groundedness: **1.8125 / 2**
* 13/32 intent predictions correct
* 19/32 intent predictions incorrect

This demonstrates that intent classification, evidence retrieval, and response generation are separate failure points.

A particularly important warning is classifier confidence.

Only three human-reviewed cases had confidence ≥0.70, and two of those three were wrong. That corresponds to a **66.67% error rate among high-confidence cases in this small sample**.

This statistic is only a warning signal because the sample contains three high-confidence cases, but it demonstrates why confidence alone should not be interpreted as reliability.

The current system can therefore retrieve useful historical evidence even when its headline intent prediction is wrong. Conversely, strong retrieval does not guarantee a strong generated reply.

## 9. One-week improvement plan

### Days 1–2: improve intent classification

Move from conservative keyword-heavy development labels toward better labelled training data, with additional examples for:

* feedback vs safety
* availability vs information
* product quality vs pricing
* delivery problem vs delivery information
* store/service vs ordinary store-related questions

Use confusion-matrix-driven sampling rather than broad keyword expansion.

### Days 3–4: improve evidence-aware response generation

Replace the current deterministic templates with a constrained evidence-grounded response generator that:

* explicitly answers the customer request;
* identifies the historical action pattern;
* avoids inventing policy;
* requests the same information Tesco historically requested when needed;
* escalates when evidence is insufficient.

Evaluate whether the 0.70 confidence threshold is appropriate.

Calibrate confidence on held-out labelled data rather than tuning directly on the frozen golden set.

Measure precision of AUTO_HANDLE separately from overall intent accuracy.

Expand the human review set using the failure categories already identified and continue tracking disagreements between human and LLM judges.

Test whether larger and more diverse review samples change the observed agreement patterns, and use disagreement cases to refine the evaluation rubric.


## 10. Limitations

The frozen golden set contains 172 conversations, with 151 target-intent examples, so per-intent estimates are noisy for small classes.

Weak supervision was intentionally conservative and is not ground truth.

The retrieval system is lexical rather than a semantic embedding retriever.

The current response generator is deterministic and therefore underuses the rich historical evidence.

The escalation policy is intentionally conservative, which results in only six AUTO_HANDLE decisions in the current frozen-gold evaluation.

Human-vs-LLM agreement was measured on 32 reviewed cases, but the agreement was low and the sample was small, so the LLM judge is treated only as a secondary evaluation signal rather than ground truth.

Despite these limitations, the project provides a complete pipeline from noisy support data to intent classification, historical evidence retrieval, response drafting, handling decisions, human evaluation, and independent LLM judging.

The main priority throughout the project was to make the evaluation honest and the system easy to inspect.
