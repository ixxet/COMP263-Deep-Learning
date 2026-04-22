# Fraud Sentinel

Deployment-ready COMP263 final project: credit-card fraud detection with a PyTorch model, FastAPI inference service, SvelteKit dashboard, LangGraph case-review workflow, RAG-grounded analyst briefs, Prometheus metrics, and Flux/Kustomize manifests for a Talos k3s platform.

## What It Builds

| Layer | Technology | Purpose |
| --- | --- | --- |
| Model | PyTorch | Dense fraud classifier plus autoencoder anomaly signal |
| API | FastAPI | Prediction, batch upload, cases, reviews, health, metrics |
| Agent | LangGraph | Durable high-risk case review with human approval gates |
| RAG | TEI + Qdrant + vLLM | Grounded analyst brief from model/policy documents |
| UI | SvelteKit | Uploads, live predictions, case queue, review actions |
| Data | Kaggle credit card fraud dataset | Real historical fraud data for training/evaluation |
| Storage | Postgres | Predictions, cases, model registry, audit events |
| GitOps | Kustomize + Flux | Talos k3s deployment package |
| Observability | Prometheus + Grafana | API, model, case, and agent metrics |

The LLM never decides fraud. The PyTorch model produces the score; the agent explains high-risk or uncertain cases and pauses for human review.

## Local Development

```bash
cd "Fina project/fraud-sentinel"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the API with local in-memory storage after model artifacts exist:

```bash
export FRAUD_MODEL_DIR=./artifacts/model
uvicorn fraud_sentinel.api.main:app --app-dir backend --reload
```

Train from an existing Kaggle CSV:

```bash
python -m fraud_sentinel.cli.train \
  --csv data/creditcard.csv \
  --output-dir artifacts/model \
  --min-pr-auc 0.70 \
  --min-recall 0.80
```

Download and train through Kaggle credentials:

```bash
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
python -m fraud_sentinel.cli.train --download-if-missing --output-dir artifacts/model
```

Run the SvelteKit UI:

```bash
cd frontend
npm install
npm run dev
```

## Deployment Shape

The Talos overlay assumes existing platform services are reachable through internal DNS and credentials are provided by a SOPS-managed Secret:

```bash
kustomize build k8s/overlays/talos
```

Images are expected at:

- `ghcr.io/ixxet/fraud-sentinel-api:a3e80b5`
- `ghcr.io/ixxet/fraud-sentinel-trainer:280d31a`
- `ghcr.io/ixxet/fraud-sentinel-ui:280d31a`

The trainer `Job` writes model artifacts to the `fraud-model-artifacts` PVC. The API refuses readiness until a valid artifact bundle is present.
The trainer image is built for `linux/amd64` from a CUDA-enabled PyTorch runtime and the Kubernetes trainer jobs request `nvidia.com/gpu: "1"` for the 3090 node.
The Talos overlay also starts a Cloudflare quick tunnel in front of `fraud-ui` for smoke testing and public demo access.

## UI Design Handoff

For product/design handoff, see `docs/ui-design-handoff.md`. It explains the stack, prediction workflow, batch workflow, case-review workflow, ideal demo uses, and design guardrails without locking the UI into the current layout.

## Test Suites

```bash
PYTHONPATH=backend python3 -m unittest discover backend/tests
python3 -m py_compile ci/e2e_api.py
bash -n ci/smoke_cluster.sh
./ci/smoke_cluster.sh
```

Use `REQUIRE_MODEL_READY=true ./ci/smoke_cluster.sh` after the Kaggle trainer has produced model artifacts. Before training, the smoke suite verifies the expected not-ready state instead of failing the deployment.

## Gates

| Gate | What must pass |
| --- | --- |
| G1 Data | Kaggle schema valid, class imbalance reported, no target leakage |
| G2 Model | PR-AUC, recall, precision, F1, confusion matrix, thresholds generated |
| G3 API | Prediction, batch upload, case creation, review, health, metrics |
| G4 Agent | Risk routing, RAG retrieval, grounding gate, human interrupt/resume |
| G5 UI | Prediction form, CSV upload, case queue, case review |
| G6 Deploy | Images build, manifests validate, pods ready, metrics scraped |
