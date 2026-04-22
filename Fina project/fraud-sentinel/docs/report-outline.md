# COMP263 Report Outline

## 1. Problem Understanding

Fraud Sentinel addresses credit card fraud detection using real historical transaction data. The business risk is imbalanced: fraudulent transactions are rare but expensive, while false alarms can harm legitimate customers.

## 2. Dataset

The project uses the Kaggle credit card fraud dataset with `Time`, `Amount`, anonymized PCA features `V1`-`V28`, and the binary label `Class`.

## 3. Model Design

The primary model is a PyTorch dense neural network trained as a supervised binary classifier. A PyTorch autoencoder is trained on normal transactions to provide a supporting anomaly score.

## 4. Evaluation

Accuracy is not the main metric because the dataset is highly imbalanced. The report should present PR-AUC, recall, precision, F1-score, and confusion matrix. Recall is emphasized because missed fraud is costly.

## 5. Deployment

The system deploys to Talos k3s using Flux-ready Kustomize manifests. The trainer writes artifacts to a PVC; the API loads the artifact bundle; Prometheus scrapes API/model metrics; Grafana visualizes readiness, request rate, risk bands, and case volume.

## 6. Agentic Review

LangGraph routes uncertain and high-risk cases through a durable review workflow. RAG retrieves model-card, threshold, ethics, and fraud-response policy. vLLM generates a grounded analyst brief. Human reviewers approve, escalate, or dismiss the case.

## 7. Ethics

The model does not prove fraud. It prioritizes review. False positives and false negatives have different harms, so customer-impacting action should require additional operational context and human review.

