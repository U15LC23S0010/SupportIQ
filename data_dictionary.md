# Data Dictionary — Customer Support on Twitter

Dataset:

Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)

Main file:

`data/raw/twcs/twcs.csv`

## Dataset Overview

The dataset contains Twitter customer-support interactions from multiple brands and customer accounts. Each row represents a tweet and includes identifiers, message direction, timestamp, tweet text, and relationships to other tweets.

SupportIQ uses these tweet-level relationships to reconstruct customer-support conversations and identify customer → support response pairs for the selected Tesco subset.

The raw dataset contains 2,811,774 tweets and spans timestamps from 2008 to 2017.

## Columns

| Column                    | Observed Type                | Description                                                                                                                                                                                                                                                                                                               |
| ------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tweet_id`                | integer                      | Unique identifier for a tweet.                                                                                                                                                                                                                                                                                            |
| `author_id`               | string                       | Identifier of the account that authored the tweet. This may represent a customer account or a support/brand account.                                                                                                                                                                                                      |
| `inbound`                 | boolean                      | Indicates the direction of the tweet. `True` represents an inbound customer message; `False` represents an outbound support or brand message.                                                                                                                                                                             |
| `created_at`              | string                       | Tweet creation timestamp as stored in the source CSV. It is parsed during data profiling and conversation analysis when temporal ordering is required.                                                                                                                                                                    |
| `text`                    | string                       | Text content of the tweet. The profiling audit found no empty text rows in the full dataset.                                                                                                                                                                                                                              |
| `response_tweet_id`       | float after pandas loading   | Identifier of the tweet that responds to the current tweet when such a relationship is available. Missing values are represented as `NaN` after pandas loading. Although loaded as a floating-point column because of missing values, the field represents tweet identifiers rather than continuous numeric measurements. |
| `in_response_to_tweet_id` | integer after pandas loading | Identifier of the tweet to which the current tweet is responding. This field is used to connect a tweet to its parent message when reconstructing conversations.                                                                                                                                                          |

## Conversation Relationships

The fields `response_tweet_id` and `in_response_to_tweet_id` provide tweet-level relationships that can be used to reconstruct conversation threads.

The full dataset was audited for relationship quality before the final Tesco pipeline was defined. The audit identified valid customer → support relationships as well as missing-parent, missing-tweet, customer → customer, and support → support relationships.

For the final SupportIQ pipeline, the reconstruction focuses on valid customer → Tesco support relationships rather than treating every tweet relationship as a valid support interaction.

The selected Tesco data contains:

* 16,722 reconstructed customer-support conversations
* 34,228 customer messages
* 38,573 Tesco support messages
* 38,468 valid customer → Tesco relationships

## Important Data Handling Notes

* The raw CSV is source data and is not modified directly.
* The raw dataset is intentionally not committed to Git because of its size.
* The repository contains the trained runtime artifacts and selected evaluation tables needed to inspect and run the final system without rebuilding the full dataset.
* Missing relationship fields are handled during conversation reconstruction rather than being treated as valid support links automatically.
* Conversation-level grouping is used for development splitting so that messages from the same conversation are not arbitrarily distributed across development partitions.
* The frozen golden evaluation conversations are excluded from the historical retrieval corpus to prevent evaluation leakage.
* Duplicate and near-duplicate content was audited as part of data and leakage analysis.
* The final historical retrieval corpus contains 37,855 usable customer → Tesco response pairs after the required filtering.

## Terminology

A **customer message** refers to an inbound tweet from a customer.

A **support message** refers to an outbound tweet from a brand or support account.

A **customer → Tesco response pair** refers to a customer message paired with the Tesco support message that directly responds to it according to the reconstructed tweet relationship.

A **conversation** refers to a reconstructed sequence of related tweets belonging to the same interaction.

A **historical response** refers to an observed Tesco support reply in the development retrieval corpus.

A historical response is not automatically considered a **resolution**. The term "resolution" is used only when the available conversation evidence supports the conclusion that the customer's issue was actually resolved.

A **golden evaluation case** refers to one of the 172 hand-labelled conversations frozen for final evaluation.

A **development weak label** refers to a label generated through the conservative A1.6 weak-supervision process. These labels are used for development only and are not treated as ground truth.
