# SupportIQ — Evidence-Grounded AI Customer Support Agent

SupportIQ is a small AI customer-support system built on the Customer Support on Twitter (TWCS) dataset.

I selected Tesco as the target brand and built the system around three things:

1. understanding what the customer is asking,
2. finding similar cases from Tesco's past support interactions, and
3. deciding whether the message is safe to handle automatically or should be sent to a human.

The main goal was not to build the most complex model possible. I focused on making the system easy to inspect, leakage-safe, reproducible, and honest about where it works and where it fails.

## 1. Project Structure

supportiq/

│
├── data/
│   └── raw/
│       └── twcs/
│           └── twcs.csv                 # local dataset, not committed
│
├── results/
│   ├── models/                          # trained runtime artifacts
│   └── tables/                          # selected evaluation outputs
│
├── scripts/
│   ├── analyze_human_review_failures.py
│   ├── build_evaluation_summary.py
│   ├── build_golden_candidate_review_pool.py
│   ├── build_tesco_response_corpus.py
│   ├── build_tesco_retrieval_index.py
│   ├── build_tesco_weak_labels_a1_6.py
│   ├── calculate_human_llm_agreement.py
│   ├── compare_tfidf_baselines_shared_split.py
│   ├── evaluate_majority_baseline.py
│   ├── evaluate_supportiq_agent.py
│   ├── human_reply_review_tool.py
│   ├── prepare_human_reply_review.py
│   ├── prepare_llm_judge_batch.py
│   ├── repair_golden_annotation_edge_cases.py
│   ├── run_gemini_llm_judge.py
│   ├── select_top5_failures.py
│   ├── test_supportiq_agent.py
│   ├── test_tesco_retriever.py
│   ├── train_tfidf_intent_baseline.py
│   └── train_tfidf_linear_svm_baseline.py
│
├── src/
│   ├── agent/
│   └── retrieval/
│
├── annotation_guidelines.md
├── brand_selection.md
├── data_dictionary.md
├── DECISION_LOG.md
├── REPORT.md
├── requirements.txt
└── README.md


The repository contains only the scripts used by the final pipeline and evaluation. Earlier exploratory scripts were removed from the submission to keep the repository focused.

## 2. Setup

The project was developed with Python 3.12.

Install the dependencies:

python -m pip install -r requirements.txt

Then set the project root on `PYTHONPATH`:

$env:PYTHONPATH = (Get-Location).Path

The original TWCS dataset should be available locally at:

data/raw/twcs/twcs.csv

The raw dataset is intentionally not committed to Git because of its size.

The submitted repository contains the trained classifier and retrieval artifacts needed to run the agent, so the reviewer does not need to rebuild everything from the full dataset just to test the system.

## 3. Dataset

The project uses the Kaggle dataset:

`thoughtvector/customer-support-on-twitter`

The full dataset contains:

* 2,811,774 tweets
* 702,777 unique authors
* timestamps from 2008 to 2017

The dataset contains customer-support conversations between customers and many different brands, which made it useful for testing the pipeline on noisy, short, multi-turn messages.

More information about the dataset columns is in:

data_dictionary.md

## 4. Why I Chose Tesco

I did not pick the brand only because it had a large number of tweets.

I compared major support accounts using:

* customer-message volume
* support-message volume
* valid customer-to-support relationships
* conversation depth
* diversity of customer issues

Tesco gave me a good balance between enough historical data and a wide range of support situations.

The Tesco subset contained:

* 16,722 reconstructed conversations
* 34,228 customer messages
* 38,573 Tesco support messages
* 38,468 valid customer → Tesco relationships

The brand-selection analysis is documented in:

brand_selection.md

## 5. Intent Taxonomy

After looking through Tesco conversations, I defined the following eight target intents:

delivery_order
pricing_payment
store_staff_service
product_availability
product_quality
product_safety_sensitive
product_information_policy
feedback_suggestion

I also kept two review-only categories:

non_support_social
other_unclear


These were useful during annotation because not every tweet is a normal support request. For example, a casual thank-you should not be forced into one of the normal support categories.

The annotation rules were frozen before the final golden-set evaluation and are documented in:

annotation_guidelines.md

Some important rules were:

* genuine safety concerns take priority over ordinary product-quality complaints
* trying to find or obtain a product is availability
* asking about ingredients, rules, specifications, or policies is information/policy
* simply mentioning a store does not automatically make something a store/service issue
* Tesco mentioning the word "feedback" does not automatically make the customer's intent feedback

## 6. Golden Evaluation Set

The final golden evaluation set contains:

* 172 hand-labelled conversations
* 151 target-intent cases
* 21 review-only cases

The assignment requires 150–250 hand-labelled examples, so I intentionally stopped at 172 rather than changing the set after seeing the results.

The frozen set is:

results/tables/tesco_golden_evaluation_set_172.csv

Once frozen, these labels were not changed during model comparison or evaluation.

## 7. Historical Evidence

The agent uses Tesco's previous support interactions as evidence.

I built the historical corpus as customer message → Tesco response pairs. This keeps the retrieval unit focused on a particular customer problem and how Tesco responded to it.

The development corpus contained:

* 37,855 historical customer → Tesco response pairs
* 0 golden conversation IDs retained
* 0 missing customer messages
* 0 missing Tesco responses

The historical corpus was built separately from the final golden set, and golden conversations were removed before the retrieval index was created.

The full development corpus was used during development but is not required for runtime inference because the submitted repository includes the pre-built retrieval index.

## 8. Development Labels

I used a conservative weak-supervision approach to create development labels.

The final accepted A1.6 set contained:

* 1,592 accepted weak labels
* 36,263 unmatched examples

These labels were used for development only. They were never treated as ground truth.

I also tried a broader A1.7 rule set and then discarded it after manual auditing showed that it was introducing semantic leakage between categories. I kept A1.6 instead.

The final A1.6 development labels are retained at:

results/tables/tesco_weak_labels_a1_6.csv

## 9. Baselines and Intent Classifier

The assignment asks for a trivial baseline and a simple learned baseline. I used both, then compared the final model against them.

### Trivial baseline

The trivial baseline always predicts the most frequent accepted A1.6 development intent.

That intent was:

delivery_order

The majority baseline was evaluated on the same 151 target-intent golden cases.

Results:

Accuracy: 17.22%
Macro-F1: 3.67%

The baseline is implemented in:

scripts/evaluate_majority_baseline.py

Its results are saved in:

results/tables/majority_baseline_metrics.json
results/tables/majority_baseline_report.txt


### Simple learned baseline

The simple learned baseline is:

TF-IDF word + character n-grams
+
LinearSVC


### Final model

The final classifier is:

TF-IDF word + character n-grams
+
Logistic Regression
+
balanced class weights

I chose this because the messages are short and noisy, the classes are imbalanced, and I wanted a model that was still easy to understand and reproduce.

Both learned models used the same grouped development split.

### Results

| Model                        | Dev Accuracy | Dev Macro-F1 | Gold Accuracy | Gold Macro-F1 |
| ---------------------------- | -----------: | -----------: | ------------: | ------------: |
| Majority-class baseline      |            — |            — |        17.22% |         3.67% |
| TF-IDF + LinearSVC           |       93.53% |       80.83% |        44.37% |        34.57% |
| TF-IDF + Logistic Regression |       93.53% |       80.83% |        47.68% |        37.53% |

The majority-class baseline provides a trivial reference point. LinearSVC is the simple learned baseline, and Logistic Regression was selected because it performed best on the frozen golden set.

## 10. Retrieval

The retrieval component uses TF-IDF cosine similarity over historical Tesco customer messages.

The trained runtime artifacts are:

results/models/tesco_retrieval_tfidf_vectorizer.joblib
results/models/tesco_retrieval_tfidf_matrix.joblib
results/models/tesco_retrieval_records.joblib


Run the retrieval smoke test with:
python .\scripts\test_tesco_retriever.py

The retriever returns:

* similar historical customer messages
* Tesco's historical responses
* similarity scores
* conversation IDs

I kept the retrieval layer intentionally simple and inspectable rather than using a more complex black-box retrieval service.

## 11. Running the Agent

The submitted repository includes the trained intent model and retrieval index needed for runtime inference.

Run:

python .\scripts\test_supportiq_agent.py

The agent produces:

* intent
* intent confidence
* evidence consistency
* decision
* decision reason
* draft reply
* historical evidence

The current decision policy is deliberately conservative:

* safety-sensitive cases are always escalated
* low-confidence predictions are escalated
* weak or missing evidence is escalated
* only routine cases with sufficient confidence and evidence can be auto-handled

## 12. Frozen-Gold Evaluation

Run:
python .\scripts\evaluate_supportiq_agent.py

The main outputs are:

results/tables/supportiq_agent_gold_predictions.csv
results/tables/supportiq_agent_evaluation_metrics.json
results/tables/supportiq_agent_evaluation_report.txt

Current frozen-gold results:

Intent accuracy: 47.68%
Intent macro-F1: 37.53%
AUTO_HANDLE: 6
ESCALATE: 166
Safety escalation rate: 100%
Mean evidence consistency: 0.2791

The low AUTO_HANDLE count is expected because the current policy is intentionally biased toward escalation when the model is uncertain.

## 13. Human Review

I manually reviewed 32 cases, with four examples from each target intent.

The review file is:
results/tables/supportiq_reply_human_review_sample_32_scored.csv


I used the following rubric:

### Reply quality

0 = poor
1 = acceptable
2 = strong

### Groundedness

0 = unsupported
1 = partial
2 = strong


### Handling

AUTO_HANDLE
ESCALATE

Results:
Cases: 32

Mean reply quality: 0.9688 / 2
Mean groundedness: 1.8125 / 2
Human AUTO_HANDLE: 12
Human ESCALATE: 20

One of the main things this review showed was that good historical evidence does not necessarily help if the intent classifier is wrong. It also showed that the current reply generator does not always use the available evidence specifically enough.

## 14. Failure Analysis

The manual review produced this breakdown:

Wrong intent + good evidence + poor reply: 14
Correct intent + good evidence + poor reply: 12
Wrong intent + weak evidence: 5
Reply-quality issue: 1

The current top failure cases are stored in:

results/tables/supportiq_top5_failure_cases.csv

The main failure patterns were:

* high-confidence wrong intent predictions
* product-quality messages classified as delivery
* availability messages classified as safety-sensitive
* positive feedback classified as safety-sensitive
* pricing questions classified as availability

These failures suggest that improving the classifier and reply-generation stage would have more value than simply making the retrieval system more complicated.

## 15. What Is Misleading About the Headline Number?

The main headline number is:

47.68% intent accuracy

That number is useful, but it is also easy to misinterpret.

The agent has multiple stages:

classification
↓
retrieval
↓
reply generation
↓
handling decision

A single intent-accuracy number does not measure all of them.

In the 32-case human review:

Intent accuracy: 40.63%
Mean reply quality: 0.9688 / 2
Mean groundedness: 1.8125 / 2

Only three reviewed cases had confidence ≥ 0.70, and two of those three were wrong:

High-confidence cases: 3
High-confidence wrong: 2
High-confidence error rate: 66.67%

That 66.67% figure is only a warning signal because there were just three high-confidence cases.

The main lesson is that the headline intent score should not be presented as the overall quality of the support agent.

## 16. LLM-as-Judge

I also evaluated the 32 human-reviewed cases with an independent LLM judge using:

Gemini 3.5 Flash-Lite

The judge was not shown the human scores.

Output:

results/tables/supportiq_llm_judgments_32.csv


### Human vs LLM agreement

| Dimension         | Exact Agreement |  Kappa |
| ----------------- | --------------: | -----: |
| Reply quality     |           12.5% |  0.061 |
| Groundedness      |           46.9% | -0.107 |
| Handling decision |           65.6% |  0.137 |

Reply quality and groundedness use quadratic weighted Cohen's kappa because their labels are ordinal. Handling decision uses ordinary Cohen's kappa.

The human reply-quality distribution was also highly concentrated:

0 = 1 case
1 = 31 cases
2 = 0 cases

The agreement artifacts are:

results/tables/supportiq_human_llm_agreement.json
results/tables/supportiq_human_llm_agreement.txt

The complete evaluation summary is:

results/tables/supportiq_evaluation_summary.json
results/tables/supportiq_evaluation_summary.txt

To run the Gemini evaluation again, set `GEMINI_API_KEY` and run:

python .\scripts\prepare_llm_judge_batch.py
python .\scripts\run_gemini_llm_judge.py
python .\scripts\calculate_human_llm_agreement.py

The completed judge results are already included in the repository, so a reviewer does not need an API key just to inspect the completed evaluation.

## 17. One-week improvement plan

The project was completed in four days, from September 9 to September 12, 2026.

### Days 1–2: improve intent classification

I would collect more labelled examples for the main confusion pairs:

* feedback vs safety
* availability vs information
* product quality vs pricing
* delivery issue vs delivery information
* store/service vs simple store mentions

I would use the confusion matrix to drive the sampling instead of adding more keyword rules.

### Days 3–4: improve reply generation

The current deterministic response templates are the biggest weakness.

I would replace them with constrained generation that:

* answers the actual customer question
* uses the retrieved Tesco examples as evidence
* avoids inventing policies
* asks for information Tesco historically requested
* escalates when the retrieved evidence is insufficient

### Day 5: confidence and handling calibration

I would evaluate whether the current 0.70 confidence threshold is appropriate.

More importantly, I would measure AUTO_HANDLE precision separately from overall classification accuracy.

I would keep the frozen golden set separate from threshold tuning.

### Days 6–7: evaluation improvement

I would expand the human review set using the failure categories already identified and continue tracking disagreements between human and LLM judges.

I would also test whether larger and more diverse review samples change the observed agreement patterns and use disagreement cases to refine the evaluation rubric.

## 18. Decision Log

The main engineering decisions are documented in:

DECISION_LOG.md

The log covers the non-obvious choices around:

* brand selection
* retrieval design
* leakage prevention
* taxonomy design
* safety precedence
* weak supervision
* rejecting A1.7
* model and baseline selection
* grouped validation
* escalation policy
* golden-set freezing
* human review
* LLM judging

## 19. Limitations

This is a prototype, not a production customer-support system.

The main limitations are:

* the golden evaluation set contains 172 conversations
* 151 of those belong to the target intents
* some smaller intents have relatively few evaluation examples
* retrieval is lexical TF-IDF rather than embedding-based semantic retrieval
* the current reply generator is deterministic
* the escalation policy is intentionally conservative
* the LLM judge has low agreement with the human reviewer

Despite these limitations, the project provides a complete pipeline from noisy support data to intent classification, historical evidence retrieval, response drafting, handling decisions, human evaluation, and independent LLM judging.

The main priority throughout the project was to make the evaluation honest and the system easy to inspect.
