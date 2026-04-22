# Fraud Response Playbook

When a case is flagged, the analyst should inspect the risk score, anomaly score, threshold, model version, and available transaction fields. Because the Kaggle dataset is anonymized, feature-level explanations are limited.

Recommended actions:

- Approve when the case is low impact and evidence is weak.
- Escalate when both the supervised score and anomaly score are elevated.
- Dismiss when the case is caused by a known data issue, malformed transaction, or clearly weak model evidence.

Any customer-impacting action should occur outside Fraud Sentinel in a system with identity, account history, and business controls.

