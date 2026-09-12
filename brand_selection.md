# Brand Selection — Why Tesco?

I did not choose the target brand only by tweet count. I compared the major support accounts using four things that matter for this assignment:

- customer-message volume
- Tesco/brand support-message volume
- valid customer-to-support relationships
- variety and depth of support conversations

I wanted a brand with enough historical support interactions to build a useful retrieval corpus, but also enough variety to make the intent-classification problem meaningful.

## Comparison

Some of the larger candidates were:

| Brand account | Conversations | Customer messages | Support messages |
| ------------- | ------------: | ----------------: | ---------------: |
| AmazonHelp    |        82,556 |           203,598 |          169,840 |
| AppleSupport  |        80,717 |           131,764 |          106,860 |
| Uber_Support  |        41,923 |            72,154 |           56,270 |
| SpotifyCares  |        28,280 |            48,543 |           43,265 |
| Delta         |        26,168 |            45,296 |           42,253 |
| **Tesco**     |    **16,722** |        **34,228** |       **38,573** |

Tesco had less raw volume than Amazon or Apple, but the conversations were still large enough for the task and covered a broad set of support situations.

I found examples involving:

- deliveries and orders
- prices and payments
- product availability
- product quality
- product information
- store and staff service
- complaints and feedback
- product-safety concerns
- refunds and related support actions

The Tesco conversations also contained multi-turn exchanges rather than only isolated customer messages, which was useful when reconstructing customer-support relationships.

## Why not simply choose the largest brand?

AmazonHelp had substantially more data, but the extra volume was not by itself a reason to prefer it. I wanted a domain where the customer issues were varied enough to define a useful, compact intent taxonomy and where historical replies could provide meaningful evidence for response drafting.

AppleSupport was also large, but its support interactions were more concentrated around technical/device-related issues.

Uber_Support had good volume and varied issues, but many interactions were strongly channel- and workflow-oriented.

Tesco gave me a more balanced problem for this assignment: enough data for retrieval and development, while still covering a wide range of customer-support scenarios.

## Final choice

I therefore selected **Tesco** as the target brand.

The final Tesco subset used for the project contained:
16,722 reconstructed conversations
34,228 customer messages
38,573 Tesco support messages
38,468 valid customer → Tesco relationships

The selection was made before building the final intent taxonomy and golden evaluation set.

The detailed brand comparison and supporting analysis were kept separate from the model evaluation so that the target brand was not chosen based on the final model results.
