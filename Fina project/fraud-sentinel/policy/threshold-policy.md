# Threshold Policy

Risk bands are assigned from the supervised fraud score and the autoencoder anomaly score.

Low-risk transactions are stored for audit and monitoring. Uncertain transactions require review because the model score is near the decision threshold or the anomaly signal is elevated. High-risk transactions require analyst review and may be escalated depending on the operational policy.

Thresholds are model artifacts, not hard-coded constants. They are generated during training from validation metrics and loaded by the API with the model version.

