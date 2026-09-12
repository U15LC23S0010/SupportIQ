# SupportIQ — Tesco Annotation Guidelines

## Version

if still tied, prefer the issue requiring greater support intervention or carrying greater support risk;

## Annotation objective

Label the customer's underlying support intent, not the response channel, Tesco's administrative action, or the wording used by Tesco.

## Target intents

1. `delivery_order`
   Delivery, missing items from a delivery, late orders, delivery slots, substitutions, order problems, or other problems primarily concerning an order or delivery.

2. `pricing_payment`
   Prices, charges, payment problems, duplicate charges, checkout/payment disputes, price mismatches, billing, or related monetary issues.

3. `store_staff_service`
   Concrete complaints or questions about a named store, staff behaviour, service quality, accessibility, parking, opening/operational store issues, or in-store experience.

4. `product_availability`
   Whether a product can be obtained, located, stocked, or found in a store or online.

5. `product_quality`
   Defective, damaged, stale, expired, spoiled, poor-quality, or otherwise unsatisfactory products when there is no credible safety risk.

6. `product_safety_sensitive`
   Credible health, injury, contamination, foreign-object, poisoning, dangerous-product, or comparable safety risk.

7. `product_information_policy`
   Product specifications, ingredients, allergens, dietary information, usage information, policies, rules, timings, or other product/policy details where the primary issue is information rather than obtaining the product.

8. `feedback_suggestion`
   Genuine suggestions, praise, criticism, opinion, or general feedback intended as support feedback rather than a concrete operational problem.

## Review-only labels

### `non_support_social`

Use for casual social conversation, thanks, greetings, unrelated remarks, or messages without a substantive customer-support issue.

### `other_unclear`

Use only when the available evidence is genuinely insufficient to assign one of the target intents.

## General principles

### 1. Label the underlying customer problem

Do not label based on Tesco's response wording.

For example, if Tesco replies that it will issue a refund, that does not automatically make the original customer message `pricing_payment`. The customer's original problem determines the intent.

### 2. Multi-intent messages

Use this order:

- the clearest underlying main issue;
- if two issues are equally tied, choose the first/main customer request;
- if still tied, prefer the issue requiring greater support intervention;
- for credible safety risk, `product_safety_sensitive` takes precedence.

### 3. Safety precedence

A concrete health, injury, contamination, foreign-object, or comparable safety concern is `product_safety_sensitive` even if the customer also complains about product quality.

Environmental, packaging, recycling, or sustainability complaints are not automatically safety cases. They require an actual health, injury, contamination, foreign-object, or comparable safety concern.

### 4. Availability vs information

Use `product_availability` when the customer is trying to obtain, locate, or find a product.

Use `product_information_policy` when the customer is asking for rules, specifications, ingredients, allergens, policies, timing, or other details.

### 5. Store mention does not imply store intent

A message mentioning a Tesco store is not automatically `store_staff_service`.

Use `store_staff_service` only when the store, staff, service, accessibility, parking, or in-store experience is itself the support issue.

### 6. Feedback vs concrete issue

Use `feedback_suggestion` only when the message is genuinely feedback, praise, criticism, or a suggestion.

Tesco mentioning words such as “feedback” does not change the customer's underlying intent.

### 7. Social messages

Casual thanks, greetings, or unrelated social messages should be `non_support_social` unless they contain a genuine support issue. In that case, classify the actual support issue.

### 8. Website/app/technical issues

Website, app, login, or other technical issues outside the frozen eight-target taxonomy should normally be `other_unclear` for this evaluation.

## Frozen edge cases

The following reviewed cases were explicitly resolved before the final golden set:

- `486989` → `store_staff_service`
- `2885399` → `product_information_policy`
- `2768474` → `other_unclear`
- `445775` → `other_unclear`

A reconstructed candle-lighter safety conversation was labelled `product_safety_sensitive` because the customer reported a credible safety risk.

## Golden-set rule

The 172-example golden set is frozen and must not be changed after observing evaluation results.

Weak labels are for development only and are never treated as ground truth.
