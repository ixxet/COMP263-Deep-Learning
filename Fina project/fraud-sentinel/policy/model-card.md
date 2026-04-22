# Fraud Sentinel Model Card

## Intended Use

Fraud Sentinel scores anonymized credit card transactions and helps analysts prioritize suspicious activity. It is a decision-support system, not an automated denial system.

## Inputs

The model accepts the Kaggle credit card fraud schema: `Time`, `Amount`, and anonymized PCA features `V1` through `V28`. The training label `Class` is used only during training and evaluation.

## Output

The API returns a risk score, an autoencoder anomaly score, a risk band, and the model artifact version. High and uncertain cases are routed to analyst review.

## Limits

The dataset is anonymized and historical. It does not contain merchant category, geography, customer history, or current fraud tactics. Scores should be monitored for drift and should not be used as the sole reason for blocking a cardholder.

