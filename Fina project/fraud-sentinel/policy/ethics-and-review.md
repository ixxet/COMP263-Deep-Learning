# Ethics And Analyst Review

Fraud detection creates asymmetric harm. A false negative can allow financial loss, while a false positive can block a legitimate customer at a sensitive moment. Fraud Sentinel therefore routes high-risk and uncertain cases through a human review gate.

The LLM-generated brief must remain grounded in retrieved policy and model documents. It may explain model outputs, thresholds, and limitations, but it must not claim that the LLM detected fraud or that fraud is proven.

Analysts should record one of three decisions: approve, escalate, or dismiss. Every decision is audit logged with the reviewer identity, rationale, model version, and case state.

